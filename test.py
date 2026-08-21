import os
import yaml
import warnings
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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.get_model import create_model
from utils.metrics import compute_scores
from utils.dataset import Datasets_seg

from utils.image_processing import *
from utils.util import *

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.0
warnings.simplefilter('ignore')

def load_best_params(params_file):
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"best_params.txt not found in {params_file}. Run validation first.")
    
    best_epoch_file, best_nt, best_iter = "model_best.pth", 0.0, 1
    with open(params_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Best Checkpoint:"):
                best_epoch_file = line.split(":", 1)[1].strip()
            elif line.startswith("Noise Tolerance:"):
                val = line.split(":", 1)[1].strip()
                best_nt = float(val) if val != "None" else None
            elif line.startswith("Res Iterations:"):
                best_iter = int(line.split(":", 1)[1].strip())
    return best_epoch_file, best_nt, best_iter

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
            tau, test_data_aug, dilation_kernel_size, dataset_name, pbar_desc, noise_tolerance=None):
    _, _, h, w = inp.shape

    label_counts = torch.zeros((1, h, w), device='cuda')
    overlap_counts = torch.zeros((1, h, w), device='cuda')
    batch_inputs, batch_coords = [], []
    for idx, (top, left) in enumerate(tqdm(crop_coords_list, desc=pbar_desc, leave=False)):
        if dataset_name == "4340_Steel" and check_scale_bar_flag(current_file_name, left, top, current_crop_w):
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
    parser = argparse.ArgumentParser(description="Test Dual-CGB")
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
    num_classes = config["training"]["num_classes"]

    best_params_file_template = config["test"]["best_params_file_path"]
    seg_weight_template = config["test"]["weight_path"]
    res_weight_template = config["res_test"]["weight_path"]
    stride = config["test"]["stride"]
    tau = config["test"]["tau"]
    test_batch_size = config["test"]["test_batch_size"]
    save_img_flg = config["test"]["save_img_flg"]
    test_crop_size = config["test"]["test_crop_size"]

    MIN_PARTICLE_AREA = 100
    BOUNDARY_THICKNESS = 1

    # ---------------- Setup Directories ----------------
    base_output_dir = os.path.join("test_results", dataset_name, f"Dual_CGB")
    if os.path.exists(base_output_dir):
        version = 2
        while os.path.exists(f"{base_output_dir}_v{version}"):
            version += 1
        base_output_dir = f"{base_output_dir}_v{version}"
    os.makedirs(base_output_dir)

    # ---------------- Logging Setup ----------------
    log_path = os.path.join(base_output_dir, "test.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(message)s", filename=log_path)

    print_and_logging(f"=== Test Dual-CGB - {datetime.datetime.now().strftime('%Y/%m/%d %H:%M')} ===")
    print_and_logging(f"Dataset: {dataset_name} | Folds: {num_folds}")
    print_and_logging(f"Output Directory: {base_output_dir}\n")

    cv_pixel_results = []
    grain_analysis_results = []
    
    # Global index to track processing order across sequential folds
    global_img_idx = 1

    # ---------------- Cross-Validation Testing Loop ----------------
    for fold in range(1, num_folds + 1):
        print_and_logging(f"\n{'='*20} Starting Test for Fold {fold} {'='*20}")
        
        fold_out_dir = os.path.join(base_output_dir, f"fold_{fold}")
        preds_dir = os.path.join(fold_out_dir, "preds")
        analyzed_dir = os.path.join(fold_out_dir, "analyzed_images")
        kde_dir = os.path.join(fold_out_dir, "kde_plots")
        for d in [preds_dir, analyzed_dir, kde_dir]:
            os.makedirs(d, exist_ok=True)

        # DataLoader Setup
        csv_file = os.path.join(csv_dir_path, f"fold_{fold}_paths.csv")
        test_dataset = Datasets_seg(csv_file=csv_file, phase='test', input_col=input_col, target_col=target_col)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        # Load Best Params from Validation
        best_params_file = best_params_file_template.format(fold=fold)
        seg_weight_path = seg_weight_template.format(fold=fold)
        res_weight_path = res_weight_template.format(fold=fold)
        try:
            _, best_nt, best_iter = load_best_params(best_params_file)
        except FileNotFoundError as e:
            print_and_logging(f"Error: {e}. Skipping Fold {fold}.")
            continue
        print_and_logging(f"-> Using params: Noise Tol: {best_nt}, Res Iter: {best_iter}")

        # Initialize Models
        model_seg = create_model(model_name, model_args).cuda().eval()
        try:
            model_seg.load_state_dict(torch.load(seg_weight_path))
        except FileNotFoundError as e:
            print_and_logging(f"Seg Model Checkpoint missing: {e}. Skipping fold.")
            continue
        
        model_res = create_model(res_model_name, res_model_args).cuda().eval()
        try:
            model_res.load_state_dict(torch.load(res_weight_path))
        except FileNotFoundError as e:
            print_and_logging(f"Restoration Model Checkpoint missing: {e}. Skipping fold.")
            continue

        scores_accum = {
            "precision": [], 
            "recall": [], 
            "f1": [], 
            "b_iou": [[] for _ in range(num_classes)], 
            "hd95": [[] for _ in range(num_classes)]
        }

        for gt, inp, file_name in tqdm(test_loader, desc=f"Fold {fold} Testing", leave=False):
            _, _, h, w = inp.shape

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
                               test_data_aug, dilation_kernel_size, dataset_name, pbar_desc="Cropping [Seg]", noise_tolerance=best_nt)

            # --- Stage 2: Restoration ---
            current_res = seg_mask
            for iter_idx in range(1, best_iter + 1):
                current_res = predict(model_res, current_res, crop_coords_list, current_crop_h, current_crop_w, stride, test_batch_size, tau, 
                                      test_data_aug, dilation_kernel_size, dataset_name, pbar_desc=f"Cropping [Res iter{iter_idx}/{best_iter}]")

            # --- Pruning process ---
            seg_mask_np = seg_mask.cpu().detach().numpy().squeeze()
            raw_pred_np = current_res.cpu().detach().numpy().squeeze()
            pred_np = remove_dead_ends(raw_pred_np, seg_mask_np, dilation_kernel_size)

            gt_np = gt.squeeze(1).cpu().detach().numpy()
            inp_np = inp.cpu().detach().numpy().squeeze(0)
            if inp_np.ndim == 3:
                if inp_np.shape[0] == 1:
                    inp_np = inp_np.squeeze(0)
                else:
                    inp_np = inp_np.transpose(1, 2, 0)
            
            # --- Pixel Metrics Calculation ---
            precision, recall, f1, b_iou, hd95 = compute_scores(pred_np, gt_np, num_classes)
            scores_accum["precision"].append(precision)
            scores_accum["recall"].append(recall)
            scores_accum["f1"].append(f1)
            
            for cls_ in range(num_classes):
                if not np.isnan(b_iou[cls_]): 
                    scores_accum["b_iou"][cls_].append(b_iou[cls_])
                if not np.isnan(hd95[cls_]): 
                    scores_accum["hd95"][cls_].append(hd95[cls_])

            # --- Grain Analysis ---
            gt_diams, gt_areas, gt_bound = extract_grains_and_boundaries(gt_np.squeeze(), MIN_PARTICLE_AREA, BOUNDARY_THICKNESS)
            pred_seg_diams, pred_seg_areas, pred_seg_bound = extract_grains_and_boundaries(seg_mask_np.squeeze(), MIN_PARTICLE_AREA, BOUNDARY_THICKNESS)
            pred_res_diams, pred_res_areas, pred_res_bound = extract_grains_and_boundaries(pred_np.squeeze(), MIN_PARTICLE_AREA, BOUNDARY_THICKNESS)

            grain_analysis_results.append({
                "fold": f"fold_{fold}", "image": current_file_name,
                "gt_areas": gt_areas, "pred_seg_areas": pred_seg_areas, "pred_res_areas": pred_res_areas,
                "gt_diams": gt_diams, "pred_seg_diams": pred_seg_diams, "pred_res_diams": pred_res_diams
            })

            # --- Visualizations & Data Preservation ---
            if save_img_flg:
                img_dir = os.path.join(preds_dir, current_file_name)
                os.makedirs(img_dir, exist_ok=True)
                visualization_res(gt_np, gt_bound, gt_diams, gt_areas, inp_np, seg_mask_np, pred_seg_bound, pred_seg_diams, pred_seg_areas,
                                  pred_np, pred_res_bound, pred_res_diams, pred_res_areas, img_dir, analyzed_dir, kde_dir, current_file_name, global_img_idx)
            global_img_idx += 1

        # Record Fold Averages
        p_avg, r_avg, f_avg = np.nanmean(scores_accum["precision"], axis=0), np.nanmean(scores_accum["recall"], axis=0), np.nanmean(scores_accum["f1"], axis=0)
        b_avg = [np.mean(l) if len(l)>0 else np.nan for l in scores_accum["b_iou"]]
        h_avg = [np.mean(l) if len(l)>0 else np.nan for l in scores_accum["hd95"]]
        
        cls_names = ["grain_boundary", "background"]
        for idx, cls_name in enumerate(cls_names):
            cv_pixel_results.append({
                "fold": f"fold_{fold}", "class": cls_name,
                "precision": p_avg[idx], "recall": r_avg[idx], "f1": f_avg[idx], "b_iou": b_avg[idx], "hd95": h_avg[idx]
            })
        cv_pixel_results.append({
            "fold": f"fold_{fold}", "class": "overall",
            "precision": macro_avg(p_avg), "recall": macro_avg(r_avg), "f1": macro_avg(f_avg), "b_iou": macro_avg(b_avg), "hd95": macro_avg(h_avg)
        })

    final_test_report(cv_pixel_results, grain_analysis_results, base_output_dir, model_name, args.config, with_res=True)
    print_and_logging("\nAll Evaluation Processes Completed Successfully.")