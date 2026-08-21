import os
import logging
import numpy as np
import pandas as pd
from .metrics import calc_stats, calc_relative_error

def print_and_logging(message):
    """Print to standard output and log simultaneously."""
    print(message)
    logging.info(message)

def get_dilation_kernel_size(file_name, dataset_name):
    dilation_kernel_size = None
    if isinstance(file_name, str):
        if dataset_name == "4340_Steel":
            dilation_kernel_size = 4

            prop_a_bases_4340_steel = [
                "20231120_4340_x50-01",
                "20231120_4340_x50-02",
                "20231213_4340_x50-03",
                "20231213_4340_x50-04",
                "20231213_4340_x50-05"
            ]
            for prop_a_base in prop_a_bases_4340_steel:
                if prop_a_base in file_name:
                    dilation_kernel_size = 11
                    break

        elif dataset_name == "316L_Grains":
            dilation_kernel_size = 8
        elif dataset_name == "TBM":
            dilation_kernel_size = 3
    return dilation_kernel_size

def final_test_report(cv_pixel_results, grain_analysis_results, base_output_dir, model_name, config, with_res=False):
    print_and_logging("\n" + "="*50)
    print_and_logging("Generating Aggregated Analysis Reports")
    print_and_logging("="*50)

    # --- 1. Pixel-level Metric Validations ---
    df_cv = pd.DataFrame(cv_pixel_results)
    summary_pixel = df_cv.groupby("class", as_index=False).agg({
        "precision": ["mean", "std"], "recall": ["mean", "std"], "f1": ["mean", "std"],
        "b_iou": ["mean", "std"], "hd95": ["mean", "std"]
    })
    summary_pixel.columns = [f"{col}_{stat}" if stat else col for col, stat in summary_pixel.columns]
    summary_pixel.to_csv(os.path.join(base_output_dir, "cv_pixel_summary.csv"), index=False, float_format="%.4f")

    with open(os.path.join(base_output_dir, "cv_pixel_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== Segmentation Pixel-Level Cross Validation Summary ===\n")
        f.write(f"Model: {model_name} | Configuration: {config}\n\n")
        f.write("[ Fold Component Scores ]\n")
        f.write(df_cv.to_string(index=False, na_rep="-"))
        f.write("\n\n[ Macro Fold Averages (Mean ± Std) ]\n")
        
        disp_df = pd.DataFrame({"class": summary_pixel["class"]})
        for m in ["precision", "recall", "f1", "b_iou", "hd95"]:
            disp_df[m] = summary_pixel.apply(lambda r: f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}" if pd.notnull(r[f'{m}_mean']) else "-", axis=1)
        f.write(disp_df.to_string(index=False))

    # --- 2. Grain Features & Relative Errors Reporting ---
    fold_errors = []
    df_grain = pd.DataFrame(grain_analysis_results)
    
    with open(os.path.join(base_output_dir, "cv_grain_analysis_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Extracted Grain Analytics Summary ===\n\n")
        
        for fold_name, group in df_grain.groupby("fold"):
            f.write(f"\n[{fold_name.upper()}]\n")
            
            fold_gt_diams, fold_pred_diams = [], []
            img_err_c, img_err_a, img_err_s = [], [], []
            if with_res:
                fold_pred_seg_diams = []
                img_err_seg_c, img_err_seg_a, img_err_seg_s = [], [], []

            for _, row in group.iterrows():
                f.write(f"  ### Image: {row['image']} ###\n")
                gt_c, gt_a, gt_s = calc_stats(row['gt_diams'])
                if with_res:
                    pred_seg_c, pred_seg_a, pred_seg_s = calc_stats(row['pred_seg_diams'])
                    pred_c, pred_a, pred_s = calc_stats(row['pred_res_diams'])
                else:
                    pred_c, pred_a, pred_s = calc_stats(row['pred_diams'])
                
                # Image-level errors
                err_c = calc_relative_error(pred_c, gt_c)
                err_a = calc_relative_error(pred_a, gt_a)
                err_s = calc_relative_error(pred_s, gt_s)
                img_err_c.append(err_c)
                img_err_a.append(err_a)
                img_err_s.append(err_s)

                if with_res:
                    err_seg_c = calc_relative_error(pred_seg_c, gt_c)
                    err_seg_a = calc_relative_error(pred_seg_a, gt_a)
                    err_seg_s = calc_relative_error(pred_seg_s, gt_s)
                    img_err_seg_c.append(err_seg_c)
                    img_err_seg_a.append(err_seg_a)
                    img_err_seg_s.append(err_seg_s)

                    f.write(f"  {'Metric':<25} | {'Ground Truth':<15} | {'Before Rest.':<20} | {'After Rest.'}\n")
                    f.write("  " + "-" * 85 + "\n")
                    f.write(f"  {'Total Particles':<25} | {gt_c:<15} | {pred_seg_c:<10} ({err_seg_c:5.2f}%) | {pred_c:<10} ({err_c:5.2f}%)\n")
                    f.write(f"  {'Avg Diameter':<25} | {gt_a:<15.2f} | {pred_seg_a:<10.2f} ({err_seg_a:5.2f}%) | {pred_a:<10.2f} ({err_a:5.2f}%)\n")
                    f.write(f"  {'Std Diameter':<25} | {gt_s:<15.2f} | {pred_seg_s:<10.2f} ({err_seg_s:5.2f}%) | {pred_s:<10.2f} ({err_s:5.2f}%)\n\n")
                
                else:
                    f.write(f"  {'Metric':<25} | {'Ground Truth':<15} | {'Prediction':<15} | {'Error (%)'}\n")
                    f.write("  " + "-" * 75 + "\n")
                    f.write(f"  {'Total Particles':<25} | {gt_c:<15} | {pred_c:<15} | {err_c:.2f}%\n")
                    f.write(f"  {'Avg Diameter':<25} | {gt_a:<15.2f} | {pred_a:<15.2f} | {err_a:.2f}%\n")
                    f.write(f"  {'Std Diameter':<25} | {gt_s:<15.2f} | {pred_s:<15.2f} | {err_s:.2f}%\n\n")
                    
                fold_gt_diams.extend(row['gt_diams'])
                if with_res:
                    fold_pred_seg_diams.extend(row['pred_seg_diams'])
                    fold_pred_diams.extend(row['pred_res_diams'])
            
            # Record fold average errors
            f_err_c = np.nanmean(img_err_c)
            f_err_a = np.nanmean(img_err_a)
            f_err_s = np.nanmean(img_err_s)

            if with_res:
                f_err_seg_c = np.nanmean(img_err_seg_c)
                f_err_seg_a = np.nanmean(img_err_seg_a)
                f_err_seg_s = np.nanmean(img_err_seg_s)
                fold_errors.append({
                    "fold": fold_name,
                    "error_count_seg": f_err_seg_c,
                    "error_mean_seg": f_err_seg_a,
                    "error_std_seg": f_err_seg_s,
                    "error_count": f_err_c,
                    "error_mean": f_err_a,
                    "error_std": f_err_s
                })
            else:
                fold_errors.append({
                    "fold": fold_name,
                    "error_count": f_err_c,
                    "error_mean": f_err_a,
                    "error_std": f_err_s
                })
            
            # Aggregate Subtotal Statistics per Fold Base
            f.write(f"  >>> {fold_name.upper()} OVERALL <<<\n")
            if with_res:
                f_gt_c, f_gt_a, f_gt_s = calc_stats(fold_gt_diams)
                f_pred_seg_c, f_pred_seg_a, f_pred_seg_s = calc_stats(fold_pred_seg_diams)
                f_pred_c, f_pred_a, f_pred_s = calc_stats(fold_pred_diams)
                f.write(f"  {'Total Particles Detected':<25} | {f_gt_c:<15} | {f_pred_seg_c} | {f_pred_c}\n")
                f.write(f"  {'Average Area Diameter':<25} | {f_gt_a:<15.2f} | {f_pred_seg_a:.2f} | {f_pred_a:.2f}\n")
                f.write(f"  {'Standard Deviation':<25} | {f_gt_s:<15.2f} | {f_pred_seg_s:.2f} | {f_pred_s:.2f}\n")
            else:
                f_gt_c, f_gt_a, f_gt_s = calc_stats(fold_gt_diams)
                f_pred_c, f_pred_a, f_pred_s = calc_stats(fold_pred_diams)
                f.write(f"  {'Total Particles Detected':<25} | {f_gt_c:<15} | {f_pred_c}\n")
                f.write(f"  {'Average Area Diameter':<25} | {f_gt_a:<15.2f} | {f_pred_a:.2f}\n")
                f.write(f"  {'Standard Deviation':<25} | {f_gt_s:<15.2f} | {f_pred_s:.2f}\n")
            f.write("="*80 + "\n")

    # --- 3. Final Overall Relative Errors Summary ---
    df_err = pd.DataFrame(fold_errors)
    with open(os.path.join(base_output_dir, "cv_grain_error_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Relative Error Summary (Cross-Validation) ===\n\n")
        f.write("[ Fold Average Relative Errors (%) ]\n")
        
        if with_res:
            # Header formatting
            f.write(f"{'Fold':<10} | {'Count Error (Before / After)':<35} | {'Avg Diam Error (Before / After)':<35} | {'Std Diam Error (Before / After)'}\n")
            f.write("-" * 120 + "\n")
            
            for _, row in df_err.iterrows():
                c_str = f"{row['error_count_seg']:.2f} / {row['error_count']:.2f}"
                a_str = f"{row['error_mean_seg']:.2f} / {row['error_mean']:.2f}"
                s_str = f"{row['error_std_seg']:.2f} / {row['error_std']:.2f}"
                f.write(f"{row['fold'].upper():<10} | {c_str:<35} | {a_str:<35} | {s_str}\n")
            
            # Overall Mean and Std
            mean_seg_c, std_seg_c = df_err["error_count_seg"].mean(), df_err["error_count_seg"].std()
            mean_c, std_c = df_err["error_count"].mean(), df_err["error_count"].std()
            
            mean_seg_a, std_a = df_err["error_mean_seg"].mean(), df_err["error_mean_seg"].std()
            mean_a, std_a = df_err["error_mean"].mean(), df_err["error_mean"].std()
            
            mean_seg_s, std_s = df_err["error_std_seg"].mean(), df_err["error_std_seg"].std()
            mean_s, std_s = df_err["error_std"].mean(), df_err["error_std"].std()
            
            f.write("\n[ Overall Cross-Validation Performance (Mean ± Std) % ]\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Metric':<25} | {'Before Restoration':<30} | {'After Restoration'}\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Particle Count Error':<25} | {mean_seg_c:6.2f} ± {std_seg_c:6.2f} | {mean_c:6.2f} ± {std_c:6.2f}\n")
            f.write(f"{'Avg Diameter Error':<25} | {mean_seg_a:6.2f} ± {std_a:6.2f} | {mean_a:6.2f} ± {std_a:6.2f}\n")
            f.write(f"{'Std Diameter Error':<25} | {mean_seg_s:6.2f} ± {std_s:6.2f} | {mean_s:6.2f} ± {std_s:6.2f}\n")

        else:
            # Header formatting
            f.write(f"{'Fold':<15} | {'Particle Count':<20} | {'Avg Diameter':<20} | {'Std Diameter'}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in df_err.iterrows():
                f.write(f"{row['fold'].upper():<15} | {row['error_count']:<20.2f} | {row['error_mean']:<20.2f} | {row['error_std']:.2f}\n")
            
            # Overall Mean and Std
            mean_err_c = df_err["error_count"].mean()
            std_err_c = df_err["error_count"].std()
            
            mean_err_a = df_err["error_mean"].mean()
            std_err_a = df_err["error_mean"].std()
            
            mean_err_s = df_err["error_std"].mean()
            std_err_s = df_err["error_std"].std()
            
            f.write("\n[ Overall Cross-Validation Performance (Mean ± Std) ]\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Particle Count Error':<25} : {mean_err_c:.2f} ± {std_err_c:.2f} %\n")
            f.write(f"{'Avg Diameter Error':<25} : {mean_err_a:.2f} ± {std_err_a:.2f} %\n")
            f.write(f"{'Std Diameter Error':<25} : {mean_err_s:.2f} ± {std_err_s:.2f} %\n")