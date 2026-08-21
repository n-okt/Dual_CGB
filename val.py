import os
import yaml
import warnings
import shutil
import logging
import datetime
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import numpy as np
import pandas as pd
from skimage import measure, segmentation
from scipy.stats import wasserstein_distance

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.get_model import create_model
from utils.metrics import calc_stats, calc_relative_error
from utils.image_processing import extract_grains_and_boundaries
from utils.dataset import Datasets_seg

from utils.image_processing import *
from utils.util import *

def process_batch(inputs, coords, model, flip_rot_augment_flg, label_counts, overlap_counts, noise_tolerance=None):
    """Conduct localized batch processing, including Test-Time Augmentations."""
    b, c, crop_h, crop_w = inputs.shape
    batch_accum = torch.zeros((b, crop_h, crop_w), device='cuda')
    batch_overlap = torch.zeros((b, crop_h, crop_w), device='cuda')
    
    inputs = inputs.cuda()

    if flip_rot_augment_flg:
        current_inputs = inputs.clone()
        for flip_flg in range(2):
            if flip_flg == 1:
                current_inputs = torch.flip(current_inputs, dims=[-1])
            for k in range(4):
                with torch.no_grad():
                    rotated_inp = torch.rot90(current_inputs, k=k, dims=[-2, -1])
                    if noise_tolerance is not None:
                        rotated_out = torch.sigmoid(model(rotated_inp, noise_tolerance=noise_tolerance))
                    else:
                        rotated_out = torch.sigmoid(model(rotated_inp))
                    
                    out = torch.rot90(rotated_out, k=-k, dims=[-2, -1])
                    if flip_flg == 1:
                        out = torch.flip(out, dims=[-1])
                    
                    out = (out >= 0.5).int().squeeze(1)
                    batch_accum += out
                    batch_overlap += 1
    else:
        with torch.no_grad():
            if noise_tolerance is not None:
                out = torch.sigmoid(model(inputs, noise_tolerance=noise_tolerance))
            else:
                out = torch.sigmoid(model(inputs))
            out = (out >= 0.5).int().squeeze(1)
            batch_accum += out
            batch_overlap += 1

    for i, (t, l) in enumerate(coords):
        label_counts[:, t:t + crop_h, l:l + crop_w] += batch_accum[i]
        overlap_counts[:, t:t + crop_h, l:l + crop_w] += batch_overlap[i]

# Predict
def predict(model, inp, crop_coords_list, current_crop_h, current_crop_w, stride, test_batch_size, 
            tau, test_data_aug, dilation_kernel_size, dataset_name, file_name, pbar_desc, noise_tolerance=None):
    _, _, h, w = inp.shape

    label_counts = torch.zeros((1, h, w), device='cuda')
    overlap_counts = torch.zeros((1, h, w), device='cuda')
    batch_inputs, batch_coords = [], []
    for idx, (top, left) in enumerate(tqdm(crop_coords_list, desc=pbar_desc, leave=False)):
        if dataset_name == "4340_Steel" and check_scale_bar_flag(file_name, left, top, current_crop_w):
            continue

        cropped_inp = inp[:, :, top:top + current_crop_h, left:left + current_crop_w]
        batch_inputs.append(cropped_inp)
        batch_coords.append((top, left))
        
        if len(batch_inputs) >= test_batch_size:
            batched_tensor = torch.cat(batch_inputs, dim=0)
            process_batch(batched_tensor, batch_coords, model, test_data_aug, label_counts, overlap_counts, noise_tolerance)
            batch_inputs, batch_coords = [], []

    if len(batch_inputs) > 0:
        batched_tensor = torch.cat(batch_inputs, dim=0)
        process_batch(batched_tensor, batch_coords, model, test_data_aug, label_counts, overlap_counts, noise_tolerance)

    pred = ((label_counts / (overlap_counts + 1e-8)) >= tau).float()
    pred[overlap_counts == 0] = 1.0
    
    if dilation_kernel_size is not None:
        pred_np = pred.cpu().detach().numpy()
        pred_np = skeletonize_and_dilate(pred_np, dilation_kernel_size)
        pred = torch.from_numpy(pred_np).float().to(inp.device)

    return pred.unsqueeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Val Dual-CGB")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---------------- Parameters Loading ----------------
    model_name = config["model"]["name"]
    model_args = config["model"]["args"]
    res_model_name = config["res_model"]["name"]
    res_model_args = config["res_model"]["args"]

    dataset_name = config["dataset"]["name"]
    csv_dir_path = config["dataset"]["csv_dir_path"]
    input_col = config["dataset"]["input_col"]
    target_col = config["dataset"]["target_col"]
    test_data_aug = config["dataset"]["test_data_aug"]
    num_folds = config["training"]["num_folds"]

    seg_weight_template = config["test"]["weight_path"]
    res_weight_template = config["res_test"]["weight_path"]
    noise_tolerances = config["test"]["noise_tolerances"]
    max_res_iters = config["test"]["max_res_iters"]
    
    stride = config["test"]["stride"]
    tau = config["test"]["tau"]
    test_batch_size = config["test"]["test_batch_size"]
    test_crop_size = config["test"]["test_crop_size"]
    
    val_start_epoch = config["test"].get("val_start_epoch", None)

    MIN_PARTICLE_AREA = 100
    BOUNDARY_THICKNESS = 1

    # Setup Logging
    log_dir = f"val_logs/{dataset_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"Dual_CGB.log")
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(message)s", filename=log_path)

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), "JST"))
    print_and_logging(f"=== Validation Dual-CGB - {now.strftime('%Y/%m/%d %H:%M')} ===")
    print_and_logging("Searching optimal combination of: [Epoch] x [Noise Tolerance] x [Res Iterations]")
    
    # ---------------- Cross-Validation Testing Loop ----------------
    for fold in range(1, num_folds + 1):
        print_and_logging(f"\n{'='*20} Starting Validation for FOLD {fold} {'='*20}")

        csv_file = os.path.join(csv_dir_path, f"fold_{fold}_paths.csv")
        val_dataset = Datasets_seg(csv_file=csv_file, phase='val', input_col=input_col, target_col=target_col)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        # 1. Initialize and Load Restoration Model (Fixed checkpoint per fold)
        model_res = create_model(res_model_name, res_model_args)
        model_res.cuda()
        model_res.eval()
        
        current_res_weight_path = res_weight_template.format(fold=fold)
        if os.path.exists(current_res_weight_path):
            model_res.load_state_dict(torch.load(current_res_weight_path))
            print_and_logging(f"-> Loaded Restoration Network fixed weights: {current_res_weight_path}")
        else:
            print_and_logging(f"-> Error: Restoration Network weights NOT FOUND at {current_res_weight_path}. Skipping fold.")
            continue

        # 2. Initialize Seg Model
        model_seg = create_model(model_name, model_args)
        model_seg.cuda()
        
        sample_seg_path = seg_weight_template.format(fold=fold, epoch="dummy")
        seg_checkpoint_dir = os.path.dirname(sample_seg_path)
        checkpoints = [f for f in os.listdir(seg_checkpoint_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]
        checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

        if val_start_epoch is not None:
            checkpoints = [f for f in checkpoints if int(f.split("_")[2].split(".")[0]) >= val_start_epoch]

        # Track global bests for this fold
        fold_best_error = float('inf')
        fold_best_epoch_file = None
        fold_best_nt = None
        fold_best_iter = None

        # 3. Epoch Loop
        for cp_file in checkpoints:
            cp_path = os.path.join(seg_checkpoint_dir, cp_file)
            model_seg.load_state_dict(torch.load(cp_path))
            model_seg.eval()
            
            print_and_logging(f"\n[Evaluating Checkpoint: {cp_file}]")
            
            epoch_best_error = float('inf')
            epoch_best_nt = None
            epoch_best_iter = None

            # 4. Noise Tolerance Loop
            for nt in noise_tolerances:
                dataset_iter_errors = {}
                for i in range(1, max_res_iters+1):
                    dataset_iter_errors[i] = 0.0
                valid_images = 0

                # 5. Dataset Loop
                for gt, inp, file_name in tqdm(val_loader, desc=f" NT={nt} Validation", leave=False):
                    _, _, h, w = inp.shape

                    gt_np = gt.squeeze(1).cpu().detach().numpy()
                    gt_diams, gt_areas, gt_bound = extract_grains_and_boundaries(gt_np.squeeze(), MIN_PARTICLE_AREA, BOUNDARY_THICKNESS)
                    gt_c, gt_a, gt_s = calc_stats(gt_diams)

                    # Determine dynamic crops
                    if test_crop_size is None:
                        top_positions, left_positions = [0], [0]
                        current_crop_h, current_crop_w = h, w
                    else:
                        top_positions = get_crop_positions(h, test_crop_size, stride)
                        left_positions = get_crop_positions(w, test_crop_size, stride)
                        current_crop_h, current_crop_w = test_crop_size, test_crop_size

                    crop_coords_list = [(t, l) for t in top_positions for l in left_positions]
                    current_file_name = file_name if isinstance(file_name, str) else file_name[0]
                    dilation_kernel_size = get_dilation_kernel_size(current_file_name, dataset_name)

                    # --- Stage 1: Segmentation ---
                    seg_mask = predict(model_seg, inp, crop_coords_list, current_crop_h, current_crop_w, stride, test_batch_size, tau, 
                                    test_data_aug, dilation_kernel_size, dataset_name, current_file_name, pbar_desc="Cropping [Seg]", noise_tolerance=nt)
                    seg_mask_np = seg_mask.cpu().detach().numpy().squeeze()

                    # --- Stage 2: Restoration ---
                    temp_errors = {}
                    has_nan = False
                    current_res = seg_mask
                    for iter_idx in range(1, max_res_iters+1):
                        current_res = predict(model_res, current_res, crop_coords_list, current_crop_h, current_crop_w, stride, test_batch_size, tau, 
                                            test_data_aug, dilation_kernel_size, dataset_name, current_file_name, pbar_desc=f"Cropping [Res iter{iter_idx}/{max_res_iters}]")

                        # --- Pruning process ---
                        raw_pred_np = current_res.cpu().detach().numpy().squeeze()
                        pred_np = remove_dead_ends(raw_pred_np, seg_mask_np, dilation_kernel_size)

                        pred_diams, pred_areas, pred_bound = extract_grains_and_boundaries(pred_np.squeeze(), MIN_PARTICLE_AREA, BOUNDARY_THICKNESS)
                        pred_c, pred_a, pred_s = calc_stats(pred_diams)

                        # Calculate Error
                        if gt_c > 0:
                            err_c = calc_relative_error(pred_c, gt_c) / 100.0
                            err_a = calc_relative_error(pred_a, gt_a) / 100.0
                            
                            if gt_s == 0:
                                err_s = 0.0 if pred_s == 0 else 1.0
                            else:
                                err_s = calc_relative_error(pred_s, gt_s) / 100.0
                        else:
                            err_c = float(pred_c)
                            err_a = 0.0 if pred_c == 0 else 1.0
                            err_s = 0.0 if pred_c == 0 else 1.0

                        if gt_c > 0 and pred_c > 0:
                            ws_dist = wasserstein_distance(gt_areas, pred_areas)
                            gt_mean_area = np.mean(gt_areas)
                            norm_ws_dist = ws_dist / gt_mean_area if gt_mean_area > 0 else 1.0
                        elif gt_c == 0 and pred_c == 0:
                            norm_ws_dist = 0.0
                        else:
                            norm_ws_dist = 1.0

                        total_error = err_c + err_a + err_s + norm_ws_dist

                        if np.isnan(total_error):
                            has_nan = True
                            break
                            
                        temp_errors[iter_idx] = total_error

                    if not has_nan:
                        for idx, err in temp_errors.items():
                            dataset_iter_errors[idx] += err
                        valid_images += 1

                # Average errors across the dataset for this (Epoch, NT)
                avg_iter_errors = {k: v / valid_images for k, v in dataset_iter_errors.items() if valid_images > 0}
                
                best_iter_for_nt = min(avg_iter_errors, key=avg_iter_errors.get)
                min_err_for_nt = avg_iter_errors[best_iter_for_nt]

                print_and_logging(f"   -> NT: {nt} | Best Iter: {best_iter_for_nt} | Score: {min_err_for_nt:.4f}")

                if min_err_for_nt < epoch_best_error:
                    epoch_best_error = min_err_for_nt
                    epoch_best_nt = nt
                    epoch_best_iter = best_iter_for_nt

            print_and_logging(f"   *** Best Setup for {cp_file} --> NT: {epoch_best_nt}, Iterations: {epoch_best_iter}, Score: {epoch_best_error:.4f}")

            if epoch_best_error < fold_best_error:
                fold_best_error = epoch_best_error
                fold_best_epoch_file = cp_file
                fold_best_nt = epoch_best_nt
                fold_best_iter = epoch_best_iter

        # 7. Save results for the fold
        if fold_best_epoch_file:
            print_and_logging(f"\n{'*'*40}")
            print_and_logging(f"WINNER FOR FOLD {fold}:")
            print_and_logging(f"Checkpoint: {fold_best_epoch_file}")
            print_and_logging(f"Noise Tolerance: {fold_best_nt}")
            print_and_logging(f"Res Iterations: {fold_best_iter}")
            print_and_logging(f"Final Score: {fold_best_error:.4f}")
            print_and_logging(f"{'*'*40}\n")

            best_cp_path = os.path.join(seg_checkpoint_dir, fold_best_epoch_file)
            best_save_path = os.path.join(seg_checkpoint_dir, "model_best.pth")
            shutil.copy(best_cp_path, best_save_path)
            
            params_txt_path = os.path.join(seg_checkpoint_dir, "best_params.txt")
            with open(params_txt_path, "w", encoding="utf-8") as f:
                f.write(f"Best Checkpoint: {fold_best_epoch_file}\n")
                f.write(f"Noise Tolerance: {fold_best_nt}\n")
                f.write(f"Res Iterations: {fold_best_iter}\n")
                f.write(f"Best Score: {fold_best_error:.4f}\n")

    print_and_logging("\nAll Folds Completed Successfully.")