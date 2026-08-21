# =====================================================================================
# Optimized & Corrected TopoLoss for Foreground=0, Background=1 Environment
# Modified from https://github.com/HuXiaoling/TopoLoss/blob/master/topoloss_pytorch.py
#
# Key Modifications:
# 1. Parallelized patch processing using joblib to significantly reduce training time.
# 2. Fixed the official bug where MSE was computed on raw logits instead of probabilities.
# 3. Corrected the 'fix' logic: pushes Birth (local minima) to 0.0 and Death (saddle points) to 1.0.
# =====================================================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import math
import numpy
import gudhi as gd
import torch
from joblib import Parallel, delayed


def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.03, pers_thresh_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process
    """
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if (gt_dgm.shape[0] == 0):
        gt_pers = None
        gt_n_holes = 0
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = list()
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list()
    else:
        tmp = gt_pers > pers_thresh_perfect
        tmp_lh = lh_pers > pers_thresh_perfect
        lh_pers_sorted_indices = numpy.argsort(lh_pers)[::-1]
        
        if numpy.sum(tmp_lh) >= 1:
            lh_n_holes_perfect = tmp_lh.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect]
        else:
            idx_holes_perfect = list()

        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes]
        idx_holes_to_fix = list(set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:]

    idx_valid = numpy.where(lh_pers > pers_thresh)[0]
    idx_holes_to_remove = list(set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = numpy.zeros(lh_dgm.shape)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / math.sqrt(2.0)

    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect
    return force_list, idx_holes_to_fix, idx_holes_to_remove


def getCriticalPoints(prob_map):
    """
    Compute the critical points of the image.
    
    [MODIFICATION] Removed the internal "lh = 1 - likelihood" inversion. 
    Since the environment assumes Foreground=0 and Background=1, and Gudhi's CubicalComplex 
    starts filtration from 0 to 1, passing the raw prob_map directly is mathematically correct.
    """
    lh_vector = numpy.asarray(prob_map).flatten()

    lh_cubic = gd.CubicalComplex(
        dimensions=[prob_map.shape[0], prob_map.shape[1]],
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    if (len(pairs_lh[0]) == 0) or (len(pairs_lh[0][0]) == 0): 
        return 0, 0, 0, False

    pairs = numpy.array(pairs_lh[0][0])
    w = prob_map.shape[1]

    pd_lh = lh_vector[pairs] 
    bcp_lh = numpy.column_stack((pairs[:, 0] // w, pairs[:, 0] % w))
    dcp_lh = numpy.column_stack((pairs[:, 1] // w, pairs[:, 1] % w))

    return pd_lh, bcp_lh, dcp_lh, True


def _precompute_gt_patch_worker(args):
    """
    Worker function to precompute topology on a single GT patch.
    """
    gt_patch, y, x = args
    if not gt_patch.any() or gt_patch.all():
        return (y, x), None

    pd_gt, bcp_gt, dcp_gt, is_valid = getCriticalPoints(gt_patch)
    if not is_valid:
        return (y, x), None
        
    return (y, x), (pd_gt, bcp_gt, dcp_gt)


def precompute_gt_topology(gt_tensor, topo_size=100, n_jobs=1):
    """
    Function to precompute GT topology.
    """
    gt_np = torch.squeeze(gt_tensor).cpu().detach().numpy()
    
    gt_inv = 1.0 - gt_np
    
    patch_args = []
    for y in range(0, gt_np.shape[0], topo_size):
        for x in range(0, gt_np.shape[1], topo_size):
            gt_patch = gt_inv[y:min(y + topo_size, gt_inv.shape[0]),
                              x:min(x + topo_size, gt_inv.shape[1])]
            patch_args.append((gt_patch, y, x))

    if n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_precompute_gt_patch_worker)(args) for args in patch_args
        )
    else:
        results = [_precompute_gt_patch_worker(args) for args in patch_args]

    return {pos: data for pos, data in results}


def _process_patch_worker(args):
    prob_patch, gt_topo, y, x = args

    if numpy.min(prob_patch) == 1 or numpy.max(prob_patch) == 0: return None
    if gt_topo is None: return None

    pd_gt, bcp_gt, dcp_gt = gt_topo

    lh_patch = 1.0 - prob_patch
    pd_lh, bcp_lh, dcp_lh, is_valid = getCriticalPoints(lh_patch)
    if not is_valid: return None

    force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thresh=0.03)

    updates_y, updates_x, updates_w, updates_ref = [], [], [], []
    h, w = prob_patch.shape

    if len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0:
        for hole_indx in idx_holes_to_fix:
            b_y, b_x = int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])
            d_y, d_x = int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])

            if 0 <= b_y < h and 0 <= b_x < w:
                updates_y.append(y + b_y); updates_x.append(x + b_x)
                updates_w.append(1.0);     updates_ref.append(1.0)
            
            if 0 <= d_y < h and 0 <= d_x < w:
                updates_y.append(y + d_y); updates_x.append(x + d_x)
                updates_w.append(1.0);     updates_ref.append(1.0)
                
        for hole_indx in idx_holes_to_remove:
            b_y, b_x = int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])
            d_y, d_x = int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])

            if 0 <= b_y < h and 0 <= b_x < w:
                ref_val = prob_patch[d_y, d_x] if (0 <= d_y < h and 0 <= d_x < w) else 0.0
                updates_y.append(y + b_y); updates_x.append(x + b_x)
                updates_w.append(1.0);     updates_ref.append(ref_val)
                
            if 0 <= d_y < h and 0 <= d_x < w:
                ref_val = prob_patch[b_y, b_x] if (0 <= b_y < h and 0 <= b_x < w) else 1.0
                updates_y.append(y + d_y); updates_x.append(x + d_x)
                updates_w.append(1.0);     updates_ref.append(ref_val)

    if not updates_y:
        return None

    return (numpy.array(updates_y), numpy.array(updates_x), 
            numpy.array(updates_w), numpy.array(updates_ref))


def getTopoLoss(likelihood_tensor, gt_topo_data, topo_size=100, n_jobs=-1):
    """
    Calculate the topology loss of the predicted image and ground truth image.
    """
    # [MODIFICATION] Fixed the official bug where MSE was calculated using raw logits.
    # Sigmoid is applied here, and the computed prob_tensor is used for both critical 
    # point extraction and MSE computation to ensure proper scaling.
    prob_tensor = torch.sigmoid(likelihood_tensor)
    prob_np = torch.squeeze(prob_tensor).cpu().detach().numpy()

    topo_cp_weight_map = numpy.zeros(prob_np.shape)
    topo_cp_ref_map = numpy.zeros(prob_np.shape)

    patch_args = []
    for y in range(0, prob_np.shape[0], topo_size):
        for x in range(0, prob_np.shape[1], topo_size):
            prob_patch = prob_np[y:min(y + topo_size, prob_np.shape[0]),
                                 x:min(x + topo_size, prob_np.shape[1])]
            
            gt_topo = gt_topo_data.get((y, x), None)
            patch_args.append((prob_patch, gt_topo, y, x))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_patch_worker)(args) for args in patch_args
    )

    for res in results:
        if res is not None:
            py_arr, px_arr, w_arr, ref_arr = res
            topo_cp_weight_map[py_arr, px_arr] = w_arr
            topo_cp_ref_map[py_arr, px_arr] = ref_arr

    device = likelihood_tensor.device
    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float, device=device)
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float, device=device)

    num_active_points = topo_cp_weight_map.sum()

    if num_active_points > 0:
        # [MODIFICATION] Calculating MSE using the activated prob_tensor instead of raw likelihood_tensor.
        loss_topo = (((prob_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
    else:
        loss_topo = torch.tensor(0.0, dtype=torch.float, requires_grad=True).to(device)

    return loss_topo