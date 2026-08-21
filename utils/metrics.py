import numpy as np
import cv2
from scipy.ndimage import binary_erosion, distance_transform_edt

def calc_stats(values):
    """Compute length, mean, and standard deviation for statistical summaries."""
    count = len(values)
    avg_val = np.mean(values) if count > 0 else 0.0
    std_val = np.std(values) if count > 0 else 0.0
    return count, avg_val, std_val

def calc_relative_error(v_pred, v_gt):
    """Calculate relative error percentage: |V_pred - V_gt| / V_gt * 100."""
    if v_gt == 0:
        return np.nan  # Return NaN to prevent division by zero
    return abs(v_pred - v_gt) / v_gt * 100.0

def compute_hd95(pred, gt):
    """
    Calculate the 95% Hausdorff Distance (HD95).
    Effectively evaluates the spatial deviation of boundaries.
    """
    if np.count_nonzero(pred) == 0 or np.count_nonzero(gt) == 0:
        return np.nan

    # Extract borders using XOR with eroded masks
    pred_border = pred ^ binary_erosion(pred, structure=np.ones((3, 3)))
    gt_border = gt ^ binary_erosion(gt, structure=np.ones((3, 3)))

    if np.count_nonzero(pred_border) == 0 or np.count_nonzero(gt_border) == 0:
        return np.nan

    # Distance transform (shortest distance to the border)
    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)

    # Get distances to each other's borders
    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    # Return the 95th percentile distance
    return max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))

def compute_boundary_iou(pred, gt, dilation_ratio=0.02):
    """
    Calculate Boundary Intersection over Union (Boundary IoU).
    """
    if np.count_nonzero(pred) == 0 and np.count_nonzero(gt) == 0:
        return np.nan 
    if np.count_nonzero(pred) == 0 or np.count_nonzero(gt) == 0:
        return 0.0

    pred_uint8 = pred.astype(np.uint8)
    gt_uint8 = gt.astype(np.uint8)

    h, w = pred_uint8.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Extract boundaries (inner contours) using erosion
    pred_erode = cv2.erode(pred_uint8, kernel, iterations=dilation)
    gt_erode = cv2.erode(gt_uint8, kernel, iterations=dilation)

    pred_b = np.logical_and(pred_uint8, np.logical_not(pred_erode))
    gt_b = np.logical_and(gt_uint8, np.logical_not(gt_erode))

    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()

    if union == 0:
        return 0.0
    return intersection / union

def compute_relaxed_metrics(pred_mask, gt_mask, tolerance=5):
    """
    Calculate Precision, Recall, and F1-score with spatial tolerance (Relaxed Metrics).
    Args:
        tolerance: Maximum allowable pixel deviation.
    """
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    if np.count_nonzero(pred_bool) == 0 and np.count_nonzero(gt_bool) == 0:
        return 1.0, 1.0, 1.0
    if np.count_nonzero(pred_bool) == 0 or np.count_nonzero(gt_bool) == 0:
        return 0.0, 0.0, 0.0

    if tolerance > 0:
        # Create a circular kernel to define the tolerance zone
        kernel_size = tolerance * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilate GT and prediction to create allowable zones
        gt_dilated = cv2.dilate(gt_bool.astype(np.uint8), kernel).astype(bool)
        pred_dilated = cv2.dilate(pred_bool.astype(np.uint8), kernel).astype(bool)
    else:
        gt_dilated = gt_bool
        pred_dilated = pred_bool

    # Precision: Ratio of predicted pixels within the GT tolerance zone
    tp_prec = np.logical_and(pred_bool, gt_dilated).sum()
    precision = tp_prec / pred_bool.sum()

    # Recall: Ratio of GT pixels within the prediction tolerance zone
    tp_rec = np.logical_and(gt_bool, pred_dilated).sum()
    recall = tp_rec / gt_bool.sum()

    # Calculate F1 Score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def compute_scores(pred, gt, num_classes, eps=1e-6, threshold=0.5, tolerance=5):
    """
    Compute evaluation metrics for each class.
    Args:
        tolerance: Allowable pixel shift for relaxed metrics (default: 5).
    """
    # Binarize while maintaining 2D spatial structure
    pred_2d = (np.squeeze(pred) > threshold).astype(int)
    gt_2d = np.squeeze(gt).astype(int)

    precision_list = []
    recall_list = []
    f1_list = []
    b_iou_list = []
    hd95_list = []

    for cls_ in range(num_classes):
        pred_mask = (pred_2d == cls_)
        gt_mask = (gt_2d == cls_)
        
        # 1. Relaxed Precision, Recall, F1
        prec, rec, f1 = compute_relaxed_metrics(pred_mask, gt_mask, tolerance=tolerance)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        
        # 2. HD95
        hd = compute_hd95(pred_mask, gt_mask)
        hd95_list.append(hd if not np.isnan(hd) else np.nan)

        # 3. Boundary IoU (Calculated only for the foreground class)
        if cls_ == 1: 
            b_iou = compute_boundary_iou(pred_mask, gt_mask)
            b_iou_list.append(b_iou)
        else:
            b_iou_list.append(np.nan)
            
    return np.array(precision_list), np.array(recall_list), np.array(f1_list), np.array(b_iou_list), np.array(hd95_list)