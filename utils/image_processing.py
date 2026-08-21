import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure, segmentation, morphology, color
from skimage.morphology import skeletonize
from scipy.stats import gaussian_kde
from scipy.ndimage import convolve

def get_crop_positions(total_size, crop_size, stride):
    """Calculate sliding window cropping coordinates for inference."""
    positions = list(range(0, total_size - crop_size + 1, stride))
    if positions[-1] != total_size - crop_size:
        positions.append(total_size - crop_size)
    return positions

def check_scale_bar_flag(file_name, left, top, crop_size):
    """Check if the current crop index contains a scale bar to skip processing."""
    if "20231120_4340_x50-01" in file_name or "20231120_4340_x50-02" in file_name:
        if left + crop_size > 3800 and top + crop_size > 3000:
            return True
    return False

def get_overlap_color_img(gt, out):
    """Generate an overlap visualization mapping TP, FP, TN, and FN to distinct colors."""
    color_map = torch.tensor([
        [0, 0, 0],         # True Negative (Black)
        [255, 0, 0],       # False Positive (Red)
        [0, 255, 255],     # False Negative (Cyan)
        [255, 255, 255]    # True Positive (White)
    ], dtype=torch.float32, device="cuda")

    B, H, W = gt.shape
    overlap_indices = out + (gt * 2)
    flat_indices = overlap_indices.flatten()
    color_overlap = color_map[flat_indices.long()]
    return color_overlap.reshape(B, H, W, 3)

def save_single_img(img, save_path, scale=255):
    """Scale and save a generic NumPy array as an image."""
    img_save = (img * scale).clip(0, 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR) if img_save.ndim == 3 else img_save)

def extract_grains_and_boundaries(mask_np, min_area=100, boundary_thickness=1):
    """Extract morphological features (grains and boundaries) from a binary mask."""
    threshold = mask_np.max() / 2.0
    if threshold == 0:
        foreground_mask = np.zeros_like(mask_np, dtype=bool)
    else:
        foreground_mask = mask_np > threshold

    labeled_image = measure.label(foreground_mask, connectivity=2)
    labeled_image_cleared = segmentation.clear_border(labeled_image)

    regions = measure.regionprops(labeled_image_cleared)
    valid_regions = [region for region in regions if region.area >= min_area]

    diameters = [prop.equivalent_diameter_area for prop in valid_regions]
    areas = [np.pi * (d / 2.0)**2 for d in diameters]
    
    valid_labels = [region.label for region in valid_regions]
    mask = np.isin(labeled_image_cleared, valid_labels)
    filtered_labeled_image = labeled_image_cleared * mask
    
    boundary_mask = segmentation.find_boundaries(filtered_labeled_image, mode='inner')
    if boundary_thickness > 1:
        footprint = morphology.disk(boundary_thickness)
        boundary_mask = morphology.dilation(boundary_mask, footprint)
        
    return diameters, areas, boundary_mask

def overlay_boundaries(image_gray, boundary_mask):
    """Overlay calculated boundary masks in green onto the target grayscale image."""
    display_img = (image_gray * 255).astype(np.uint8) if image_gray.max() <= 1.0 else image_gray.astype(np.uint8)
    
    if display_img.ndim == 2:
        final_image = color.gray2rgb(display_img)
    elif display_img.ndim == 3 and display_img.shape[-1] == 1:
        final_image = color.gray2rgb(display_img[:, :, 0])
    else:
        final_image = display_img.copy()
        
    final_image[boundary_mask] = [0, 255, 0]
    return final_image

def skeletonize_and_dilate(pred, dilation_kernel_size):
    pred = pred.squeeze()
    
    # Invert logic: assuming Boundary=0, Background=1
    boundary_mask = 1.0 - pred
    skel = skeletonize(boundary_mask > 0.5).astype(np.uint8)
    
    if dilation_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        dilated_boundary = cv2.dilate(skel, kernel, iterations=1)
    else:
        dilated_boundary = skel
        
    normalized_out = 1.0 - dilated_boundary
    return np.expand_dims(normalized_out, axis=0) # Shape: (1, H, W)

def remove_dead_ends(pred_np, base_np, dilation_kernel_size=None):
    base_b = (1.0 - base_np) > 0.5
    pred_b = (1.0 - pred_np) > 0.5

    skel_pred = skeletonize(pred_b)
    protected_mask = base_b

    dy = [-1, -1, -1,  0, 0,  1, 1, 1]
    dx = [-1,  0,  1, -1, 1, -1, 0, 1]

    pruned = skel_pred.copy()
    
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbors = convolve(pruned.astype(int), kernel, mode='constant', cval=0)
    
    endpoints = pruned & (neighbors <= 1) & (~protected_mask)
    y_idx, x_idx = np.where(endpoints)
    
    queue = list(zip(y_idx, x_idx))
    H, W = pruned.shape
    
    while queue:
        y, x = queue.pop(0)
        
        if not pruned[y, x] or protected_mask[y, x]:
            continue
            
        n_count = 0
        next_y, next_x = -1, -1
        for i in range(8):
            ny, nx = y + dy[i], x + dx[i]
            if 0 <= ny < H and 0 <= nx < W:
                if pruned[ny, nx]:
                    n_count += 1
                    next_y, next_x = ny, nx
        
        if n_count <= 1:
            pruned[y, x] = False
            if n_count == 1 and not protected_mask[next_y, next_x]:
                queue.append((next_y, next_x))

    if dilation_kernel_size is not None and dilation_kernel_size > 1:
        cv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        dilated_pruned = cv2.dilate(pruned.astype(np.uint8), cv_kernel, iterations=1)
    else:
        dilated_pruned = pruned.astype(np.uint8)

    return (1.0 - dilated_pruned).astype(np.float32)

def macro_avg(score_array):
    """Calculate the macro average, safely ignoring NaN values."""
    valid_scores = [s for s in score_array if not np.isnan(s)]
    return sum(valid_scores) / len(valid_scores) if valid_scores else np.nan

def visualization(gt, gt_bound, gt_diams, gt_areas, inp, pred, pred_bound, pred_diams, pred_areas, img_dir, analyzed_dir, kde_dir, file_name, idx):
    # Masks preservation
    save_single_img(gt.squeeze(0), os.path.join(img_dir, "gt.png"), 255)
    save_single_img(inp, os.path.join(img_dir, "inp.png"), 1 if inp.max() > 1 else 255)
    save_single_img(pred.squeeze(0), os.path.join(img_dir, f"out.png"), 255)
    
    color_overlap = get_overlap_color_img(torch.tensor(gt).cuda(), torch.tensor(pred).cuda())
    save_single_img(color_overlap.cpu().numpy().squeeze(0), os.path.join(img_dir, "overlap.png"), 1)

    # Overlay Validation Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    axes[0].imshow(overlay_boundaries(inp, gt_bound))
    axes[0].set_title(f'Ground Truth (particles: {len(gt_diams)}, avg: {np.mean(gt_diams) if gt_diams else 0:.2f})')
    axes[0].axis('off')
    
    axes[1].imshow(overlay_boundaries(inp, pred_bound))
    axes[1].set_title(f'Prediction (particles: {len(pred_diams)}, avg: {np.mean(pred_diams) if pred_diams else 0:.2f})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analyzed_dir, f"analyzed_{file_name}.png"), dpi=300)
    plt.close(fig)

    # High-Resolution Publication-ready KDE Plot
    all_areas = gt_areas + pred_areas
    if all_areas and max(all_areas) > min(all_areas):
        fig_kde, ax_kde = plt.subplots(figsize=(8, 6))
        x_eval = np.linspace(min(all_areas), max(all_areas), 1000)
        
        def plot_kde(areas, label, color):
            if len(areas) > 1 and max(areas) > min(areas):
                try:
                    kde = gaussian_kde(areas)
                    y_eval = kde(x_eval)
                    ax_kde.plot(x_eval, y_eval, label=label, color=color, linewidth=2.0)
                    ax_kde.fill_between(x_eval, y_eval, alpha=0.3, color=color)
                except np.linalg.LinAlgError:
                    pass

        plot_kde(gt_areas, 'Ground Truth', 'black')
        plot_kde(pred_areas, 'Prediction', 'red')
        
        ax_kde.set_title(f'Area Distribution KDE Plot (Test image {idx})', fontweight='bold', pad=10)
        ax_kde.set_xlabel('Area (pixels)')
        ax_kde.set_ylabel('Density')
        
        ax_kde.grid(True, linestyle='--', alpha=0.6)
        ax_kde.legend(loc='upper right', framealpha=0.9, edgecolor='black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(kde_dir, f"kde_{file_name}.png"), dpi=600, bbox_inches='tight')
        plt.close(fig_kde)

def visualization_res(gt, gt_bound, gt_diams, gt_areas, inp, pred_seg, pred_seg_bound, pred_seg_diams, pred_seg_areas, 
                      pred_res, pred_res_bound, pred_res_diams, pred_res_areas, img_dir, analyzed_dir, kde_dir, file_name, idx):
    save_single_img(gt.squeeze(0), os.path.join(img_dir, "gt.png"), 255)
    save_single_img(inp, os.path.join(img_dir, "inp.png"), 1 if inp.max()>1 else 255)
    save_single_img(pred_seg, os.path.join(img_dir, "out_seg.png"), 255)
    save_single_img(pred_res, os.path.join(img_dir, f"out_res.png"), 255)
    
    overlap = get_overlap_color_img(torch.tensor(gt).cuda(), torch.tensor(pred_res).cuda())
    save_single_img(overlap.cpu().numpy().squeeze(), os.path.join(img_dir, "overlap.png"), 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    axes[0].imshow(overlay_boundaries(inp, gt_bound))
    axes[0].set_title(f'GT (particles: {len(gt_diams)}, avg: {np.mean(gt_diams) if gt_diams else 0:.2f})')
    axes[0].axis('off')
    
    axes[1].imshow(overlay_boundaries(inp, pred_seg_bound))
    axes[1].set_title(f'Before Restoration (particles: {len(pred_seg_diams)}, avg: {np.mean(pred_seg_diams) if pred_seg_diams else 0:.2f})')
    axes[1].axis('off')
    
    axes[2].imshow(overlay_boundaries(inp, pred_res_bound))
    axes[2].set_title(f'After Restoration (particles: {len(pred_res_diams)}, avg: {np.mean(pred_res_diams) if pred_res_diams else 0:.2f})')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(analyzed_dir, f"analyzed_{file_name}.png"), dpi=300)
    plt.close(fig)

    all_areas = gt_areas + pred_seg_areas + pred_res_areas
    if all_areas and max(all_areas) > min(all_areas):
        fig_kde, ax_kde = plt.subplots(figsize=(8, 6))
        x_eval = np.linspace(min(all_areas), max(all_areas), 1000)
        
        def plot_kde(areas, label, color):
            if len(areas) > 1 and max(areas) > min(areas):
                try:
                    kde = gaussian_kde(areas)
                    y_eval = kde(x_eval)
                    ax_kde.plot(x_eval, y_eval, label=label, color=color, linewidth=2.0)
                    ax_kde.fill_between(x_eval, y_eval, alpha=0.3, color=color)
                except np.linalg.LinAlgError:
                    pass

        plot_kde(gt_areas, 'Ground Truth', 'black')
        plot_kde(pred_seg_areas, 'Before Restoration', 'blue')
        plot_kde(pred_res_areas, 'After Restoration', 'red')
        
        ax_kde.set_title(f'Area Distribution KDE Plot (Test image {idx})', fontweight='bold', pad=10)
        ax_kde.set_xlabel('Area (pixels)')
        ax_kde.set_ylabel('Density')
        
        ax_kde.grid(True, linestyle='--', alpha=0.6)
        ax_kde.legend(loc='upper right', framealpha=0.9, edgecolor='black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(kde_dir, f"kde_{file_name}.png"), dpi=600, bbox_inches='tight')
        plt.close(fig_kde)