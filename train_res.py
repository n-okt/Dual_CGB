import os
import time
import math
import torch
import random
import logging
import warnings
import datetime
import argparse
import yaml
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.dataset import Datasets_res
from torch.utils.data import DataLoader
from utils.get_model import create_model
from topolosses.losses import CLDiceLoss
from utils.topoloss_pytorch import getTopoLoss
from utils.adversarial_loss import AdversarialLoss
warnings.simplefilter('ignore')

############################## Config Loading & Args ##############################
parser = argparse.ArgumentParser(description="Train GAN Model with CV")
parser.add_argument(
    "--config", 
    type=str, 
    required=True, 
    help="Path to the config yaml file (e.g., config.yaml)"
)
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

generator_name = config["generator"]["name"]
generator_args = config["generator"]["args"]

discriminator_name = config["discriminator"]["name"]
discriminator_args = config["discriminator"]["args"]

dataset_name = config["dataset"]["name"]
csv_dir_path = config["dataset"]["csv_dir_path"]
input_col = config["dataset"]["input_col"]
target_col = config["dataset"]["target_col"]

epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
accumulation_steps = config["training"]["accumulation_steps"]
num_classes = config["training"]["num_classes"]
num_folds = config["training"]["num_folds"]
save_start_epoch = config["training"].get("save_start_epoch", 1)
finetune = config["training"]["finetune"]
pretrained_dir = config["training"].get("pretrained_dir", None)

if finetune:
    topo_size = config["loss"]["topo_size"]
    topo_loss_weight = config["loss"]["topo_loss_weight"]
dis_loss_weight = config["loss"]["dis_loss_weight"]
gan_loss_weight = config["loss"]["gan_loss_weight"]
BCE_loss_weight = config["loss"]["BCE_loss_weight"]
BCE_pos_weight = config["loss"]["BCE_pos_weight"]
gan_loss_type = config["loss"]["gan_loss_type"]

gen_optimizer_conf = config["gen_optimizer"]
dis_optimizer_conf = config["dis_optimizer"]
gen_scheduler_conf = config["gen_scheduler"]
dis_scheduler_conf = config["dis_scheduler"]
####################################################################

def print_and_logging(message):
    print(message)
    logging.info(message)

def custom_collate_fn(batch):
    gts, inps, topo_datas, file_names = zip(*batch)
    gt_batch = torch.stack(gts, dim=0)
    inp_batch = torch.stack(inps, dim=0)
    topo_batch = list(topo_datas)
    file_name_batch = list(file_names)
    return gt_batch, inp_batch, topo_batch, file_name_batch

if finetune:
    assert pretrained_dir is not None, "Set the pretrained_dir for fine-tuning."

base_output_dir = os.path.join('results_res', dataset_name, generator_name)
if pretrained_dir:
    base_output_dir += "_finetune"

if os.path.exists(base_output_dir):
    version = 2
    while os.path.exists(f"{base_output_dir}_v{version}"):
        version += 1
    base_output_dir = f"{base_output_dir}_v{version}"
os.makedirs(base_output_dir)


####################################################################
### Cross-Validation Loop (Training only)
####################################################################
for fold in range(1, num_folds + 1):
    print(f"\n{'='*20} Starting Fold {fold} {'='*20}")

    ####################################################################
    ### Set seed
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = False

    output_dir = os.path.join(base_output_dir, f"fold_{fold}")
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(csv_dir_path, f"fold_{fold}_paths.csv")

    log_path = os.path.join(output_dir, f"{generator_name}_fold{fold}.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(message)s", filename=log_path)
    
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    logging.info(now.strftime('%Y/%m/%d %H:%M\n'))
    logging.info(f"Dataset CSV Dir: {csv_dir_path}")
    logging.info(f"Fold: {fold}")
    logging.info(f"Generator: {generator_name}")
    logging.info(f"Discriminator: {discriminator_name}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Batch size (per step): {batch_size}")
    logging.info(f"gan_loss_weight: {gan_loss_weight}")
    logging.info(f"dis_loss_weight: {dis_loss_weight}")
    logging.info(f"BCE_loss_weight: {BCE_loss_weight}")
    logging.info(f"BCE_pos_weight: {BCE_pos_weight}")
    logging.info(f"gan_loss_type: {gan_loss_type}")
    logging.info(f"Effective batch size: {batch_size * accumulation_steps}")
    if finetune:
        logging.info(f"\n*** FINE-TUNING MODE ***")
        logging.info(f"Loading Pretrained Weights from: {pretrained_dir}")
        logging.info(f"topo_loss_weight: {topo_loss_weight}")
        logging.info(f"topo_size: {topo_size}")
    
    checkpoint_latest_path = os.path.join(output_dir, "model_latest.pth")
    dis_latest_path = os.path.join(output_dir, "dis_latest.pth")

    ####################################################################
    ### Create dataloaders
    train_dataset = Datasets_res(
        csv_file, 
        finetune,
        phase='train', 
        input_col=input_col, 
        target_col=target_col
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        drop_last=False, 
        pin_memory=True,
        collate_fn=custom_collate_fn)
    
    print_and_logging(f'===> Number of training images: {len(train_dataset)}')

    ####################################################################
    ### Init params
    num_iters = len(train_loader)
    num_update_steps_per_epoch = math.ceil(num_iters / accumulation_steps)
    total_update_steps = epochs * num_update_steps_per_epoch

    generator = create_model(generator_name, generator_args)
    discriminator = create_model(discriminator_name, discriminator_args)

    generator.cuda()
    discriminator.cuda()

    if finetune:
        gen_weight_path = os.path.join(pretrained_dir, f"fold_{fold}", "model_latest.pth")
        dis_weight_path = os.path.join(pretrained_dir, f"fold_{fold}", "dis_latest.pth")
        
        if os.path.exists(gen_weight_path):
            generator.load_state_dict(torch.load(gen_weight_path))
            print_and_logging(f"===> Loaded Pretrained Generator: {gen_weight_path}")
        else:
            print_and_logging(f"===> WARNING: Pretrained Generator NOT FOUND: {gen_weight_path}")
            
        if os.path.exists(dis_weight_path):
            discriminator.load_state_dict(torch.load(dis_weight_path))
            print_and_logging(f"===> Loaded Pretrained Discriminator: {dis_weight_path}")
        else:
            print_and_logging(f"===> WARNING: Pretrained Discriminator NOT FOUND (Training Dis from scratch)")

    gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=gen_optimizer_conf["lr"], weight_decay=gen_optimizer_conf["weight_decay"], betas=gen_optimizer_conf["betas"])
    dis_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=dis_optimizer_conf["lr"], weight_decay=dis_optimizer_conf["weight_decay"], betas=dis_optimizer_conf["betas"])

    if gen_scheduler_conf["flg"]:
        gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=total_update_steps, eta_min=gen_scheduler_conf["eta_min"])
    if dis_scheduler_conf["flg"]:
        dis_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dis_optimizer, T_max=total_update_steps, eta_min=dis_scheduler_conf["eta_min"])

    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 / BCE_pos_weight)]).cuda())
    adversarial_loss = AdversarialLoss(type=gan_loss_type)

    ####################################################################
    ### Training
    print("===> Generator:", generator_name, ", Discriminator:", discriminator_name)
    print(f"===> Total epochs: {epochs}, Effective Batch size: {batch_size * accumulation_steps}\n")

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        epoch_loss = {
            "gen": 0.0,
            "gen_gan": 0.0,
            "gen_topo": 0.0,
            "gen_bce": 0.0,
            "dis": 0.0,
        }
        epoch_start_time = time.time()

        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()

        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"), 1):
            gt = data[0].cuda()
            inp = data[1].cuda()
            gt_topo_data = data[2]
            
            out = generator(inp)
            out_prob = torch.sigmoid(out)

            gen_loss = 0
            dis_loss = 0

            # discriminator loss
            dis_input_real = gt
            dis_input_fake = out_prob.detach()
            
            dis_real, dis_real_feat = discriminator(dis_input_real)
            dis_fake, dis_fake_feat = discriminator(dis_input_fake)
            
            dis_real_loss = adversarial_loss(dis_real, True, True)
            dis_fake_loss = adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            dis_loss = dis_loss * dis_loss_weight

            # generator adversarial loss
            gen_input_fake = out_prob
            gen_fake, gen_fake_feat = discriminator(gen_input_fake)
            
            gen_gan_loss = adversarial_loss(gen_fake, True, False)
            gen_gan_loss = gen_gan_loss * gan_loss_weight
            gen_loss += gen_gan_loss

            # generator topo loss
            gen_topo_loss = 0
            if finetune:
                for k in range(out.shape[0]):
                    logit_k = out[k] 
                    target_k = gt[k]
                    topo_k = gt_topo_data[k]
                    gen_topo_loss += getTopoLoss(logit_k, topo_k, topo_size, n_jobs=8)
                    
                gen_topo_loss = (gen_topo_loss / out.shape[0]) * topo_loss_weight
            gen_loss += gen_topo_loss

            # generator bce
            gen_bce_loss = bce_loss(out, gt) * BCE_loss_weight * BCE_pos_weight
            gen_loss += gen_bce_loss

            epoch_loss["gen_gan"] += gen_gan_loss.item()
            epoch_loss["gen_topo"] += float(gen_topo_loss)
            epoch_loss["gen_bce"] += gen_bce_loss.item()

            # Gradient Accumulation
            gen_loss = gen_loss / accumulation_steps
            dis_loss = dis_loss / accumulation_steps

            gen_loss.backward()
            dis_loss.backward()

            if i % accumulation_steps == 0 or i == len(train_loader):
                gen_optimizer.step()
                dis_optimizer.step()
                gen_optimizer.zero_grad()
                dis_optimizer.zero_grad()

                if gen_scheduler_conf["flg"]:
                    gen_scheduler.step()
                if dis_scheduler_conf["flg"]:
                    dis_scheduler.step()

            epoch_loss["gen"] += gen_loss.item() * accumulation_steps
            epoch_loss["dis"] += dis_loss.item() * accumulation_steps

        for k in epoch_loss:
            epoch_loss[k] /= num_iters

        print_and_logging(
            f"Epoch: {epoch}   Time: {int((time.time()-epoch_start_time)//60)}m   "
            f"dis loss: {epoch_loss['dis']:.4f}  "
            f"gen loss: {epoch_loss['gen']:.4f} "
            f"[gan: {epoch_loss['gen_gan']:.4f}, "
            f"bce: {epoch_loss['gen_bce']:.4f}, "
            f"topo: {epoch_loss['gen_topo']:.4f}]"
        )

        # Save latest (Gen & Dis)
        torch.save(generator.state_dict(), checkpoint_latest_path)
        torch.save(discriminator.state_dict(), dis_latest_path)

        # Save current
        if epoch >= save_start_epoch:
            curr_ckpt_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
            torch.save(generator.state_dict(), curr_ckpt_path)

    print_and_logging(f"Fold {fold} Training Finished.")

print("\nAll Folds Training Finished! Results saved to", base_output_dir)