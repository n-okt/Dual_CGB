# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

# Encoder-Guided Denoising Block
class EGDBlock(nn.Module):
    def __init__(self, c, max_threshold=2.0):
        super().__init__()
        self.mask_generator = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c),
            nn.Conv2d(c, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.norm = LayerNorm2d(c)
        self.base_threshold = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.max_threshold_scale = max_threshold

    def forward(self, x_dec, x_enc, noise_tolerance=0.5):
        """
        Parameters:
            x_dec: Decoder level-1 output.
            x_enc: Encoder level-1 output.
            noise_tolerance: Tolerance level for noise, 
                            ranging from 0.0 (strictly remove all suspicious regions) to 1.0 (allow all regions).
        """

        # Morphological opening to remove small background noise
        x_enc = -F.max_pool2d(-x_enc, kernel_size=3, stride=1, padding=1)
        x_enc = F.max_pool2d(x_enc, kernel_size=3, stride=1, padding=1)
        
        # Generate denoising mask from encoder output
        mask = self.mask_generator(x_enc)
    
        # Apply mask
        x_dec_masked = self.norm(x_dec) * mask
        
        # Get threshold
        shift = self.max_threshold_scale * (1.0 - noise_tolerance)
        dynamic_threshold = self.base_threshold + shift

        return F.leaky_relu(x_dec_masked - dynamic_threshold, negative_slope=0.1)

class NormSkip(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dec_norm = LayerNorm2d(c)
        self.skip_norm = LayerNorm2d(c)

    def forward(self, x_dec, enc_skip):
        x_dec_norm = self.dec_norm(x_dec)
        enc_skip_norm = self.skip_norm(enc_skip)
        return x_dec_norm + enc_skip_norm

class NAFNet_EGD(nn.Module):
    def __init__(self, segmentation, img_channel=3, num_classes=1, width=48, middle_blk_num=2, enc_blk_nums=[1,1,2,4], dec_blk_nums=[1,1,1,1]):
        super().__init__()

        self.segmentation = segmentation

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        if self.segmentation:
            self.head = nn.Conv2d(in_channels=img_channel, out_channels=num_classes, kernel_size=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.norm_skips = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.norm_skips.append(NormSkip(chan))
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

        self.egdblock = EGDBlock(width)

    def forward(self, inp, noise_tolerance=0.5):
        """
        noise_tolerance: Adjustable from 0.0 to 1.0 during inference.
        """
        B, C, H, W = inp.shape
        padded_inp = self.check_image_size(inp)

        x = self.intro(padded_inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, norm_skip, up, enc_skip in zip(self.decoders, self.norm_skips, self.ups, encs[::-1]):
            x = up(x)
            x = norm_skip(x, enc_skip)
            x = decoder(x)

        # Apply EGDBlock to the final decoder output
        x = self.egdblock(x_dec=x, x_enc=encs[0], noise_tolerance=noise_tolerance)

        x = self.ending(x)
        x = x + padded_inp

        if self.segmentation:
            x = self.head(x)
            
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x