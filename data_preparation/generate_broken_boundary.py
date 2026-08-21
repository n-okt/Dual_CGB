import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def degrade_grain_boundary_mixed(
    gt_mask: np.ndarray,
    mix_ratios: tuple = None,           # (Mod, Large) ratio
    max_thick_ratio: float = 1.6,       # Upper limit multiplier for thickness
    min_gap_dist: int = 80,             # Base minimum distance between gap centers (px)
    breakage_ratio: float = 1.0
) -> tuple[np.ndarray, tuple, int, int]:
    """
    1. Auto-detect original boundary thickness and modulate to uniform thickness.
    2. Place gap centers with randomized spacing to avoid unnatural uniform patterns.
    3. Mix Mod and Large gap profiles and perform a sharp binary cut.
    """
    # 1. Background color auto-detection
    is_white_bg = np.mean(gt_mask) > 127
    if is_white_bg:
        binary = (gt_mask < 127).astype(np.uint8)
    else:
        binary = (gt_mask > 127).astype(np.uint8)

    # 2. Skeletonization (extract centerlines)
    skel = skeletonize(binary).astype(np.uint8)
    skel_ys, skel_xs = np.where(skel > 0)
    if len(skel_xs) == 0:
        return gt_mask.copy(), (0.5, 0.5), 0, 0

    # 3. Auto-estimate grain boundary thickness from original GT image
    dist_gt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    fg_dist = dist_gt[binary > 0]
    if len(fg_dist) > 0:
        estimated_radius = np.percentile(fg_dist, 70)
        orig_thickness = max(2, int(round(estimated_radius * 2)))
    else:
        orig_thickness = 4

    # 4. Determine uniform thickness for the whole image
    max_thickness = max(orig_thickness + 1, int(round(orig_thickness * max_thick_ratio)))
    chosen_thickness = np.random.randint(orig_thickness, max_thickness + 1)

    k_size = chosen_thickness if chosen_thickness % 2 == 1 else chosen_thickness + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    uniform_thick_binary = cv2.dilate(skel, kernel, iterations=1)

    # 5. Define gap specifications
    profiles = {
        "moderate":  {"size": (15, 35), "base_count": (8, 15)},
        "few_large": {"size": (40, 70), "base_count": (2, 5)}
    }

    if mix_ratios is None:
        ratios = np.random.dirichlet([1.0, 1.0])
    else:
        ratios = np.array(mix_ratios, dtype=float)
        ratios /= np.sum(ratios)

    r_mod, r_large = ratios
    n_mod = int(round(np.random.randint(*profiles["moderate"]["base_count"]) * r_mod * 2.0 * breakage_ratio))
    n_large = int(round(np.random.randint(*profiles["few_large"]["base_count"]) * r_large * 2.0 * breakage_ratio))

    gap_specs = []
    for count, (min_s, max_s) in [
        (n_mod, profiles["moderate"]["size"]),
        (n_large, profiles["few_large"]["size"])
    ]:
        for _ in range(count):
            rx = np.random.randint(min_s, max_s) // 2
            ry = np.random.randint(min_s, max_s) // 2
            angle = np.random.randint(0, 360)
            gap_specs.append((rx, ry, angle))

    # 6. Select gap positions 
    base_dist = max(10, int(round(min_gap_dist / max(0.1, breakage_ratio))))
    candidate_indices = np.random.permutation(len(skel_xs))
    selected_centers = []

    for idx in candidate_indices:
        cx, cy = skel_xs[idx], skel_ys[idx]
        pt = np.array([cx, cy])

        if len(selected_centers) == 0:
            selected_centers.append(pt)
        else:
            dists = np.linalg.norm(np.array(selected_centers) - pt, axis=1)
            dynamic_min_dist = base_dist * np.random.uniform(0.25, 1.75)
            if np.all(dists >= dynamic_min_dist):
                selected_centers.append(pt)

        if len(selected_centers) >= len(gap_specs):
            break

    # 7. Generate gap mask
    gap_mask = np.zeros_like(binary, dtype=np.uint8)
    for pt, (rx, ry, angle) in zip(selected_centers, gap_specs):
        cx, cy = pt[0], pt[1]
        cv2.ellipse(gap_mask, (cx, cy), (rx, ry), angle, 0, 360, 1, -1)

    # 8. Cut gaps cleanly from the uniform line image
    broken_binary = uniform_thick_binary * (1 - gap_mask)

    # 9. Format output image (0 or 255)
    if is_white_bg:
        result = np.where(broken_binary > 0, 0, 255).astype(np.uint8)
    else:
        result = np.where(broken_binary > 0, 255, 0).astype(np.uint8)

    return result, ratios, orig_thickness, chosen_thickness


def _worker_process_single_image(args: tuple) -> tuple[bool, str]:
    input_path, output_path, max_thick_ratio, min_gap_dist, breakage_ratio = args
    filename = os.path.basename(input_path)

    cv2.setNumThreads(1)

    gt_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        return False, f"Failed to load: {filename}"

    try:
        degraded_img, _, _, _ = degrade_grain_boundary_mixed(
            gt_mask,
            max_thick_ratio=max_thick_ratio,
            min_gap_dist=min_gap_dist,
            breakage_ratio=breakage_ratio
        )
        cv2.imwrite(output_path, degraded_img)
        return True, filename
    except Exception as e:
        return False, f"Error processing {filename}: {e}"


def process_dataset_parallel(
    input_dir: str,
    output_dir: str,
    max_thick_ratio: float = 1.6,
    min_gap_dist: int = 90,
    breakage_ratio: float = 1.0,
    valid_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
    num_workers: int = None
):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist -> {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    file_list = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(valid_extensions)
    ]

    if not file_list:
        print(f"No supported image files found in: {input_dir}")
        return

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)

    print(f"=== Parallel Processing Started ===")
    print(f"Total Images:     {len(file_list)}")
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Breakage Ratio:   {breakage_ratio}")
    print(f"Parallel Workers: {num_workers} CPU Cores\n")

    tasks = [
        (
            os.path.join(input_dir, filename),
            os.path.join(output_dir, filename),
            max_thick_ratio,
            min_gap_dist,
            breakage_ratio
        )
        for filename in file_list
    ]

    success_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker_process_single_image, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing images"):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                print(f"\n[Warning] {msg}")

    print(f"\n=== Processing Complete ({success_count}/{len(file_list)} images saved) ===")


if __name__ == "__main__":
    
    # ############### 316L Grains ###############
    INPUT_DIR = "dataset/316L_Grains/RGMask"
    OUTPUT_DIR = "dataset/316L_Grains/boundary_broken"
    breakage_ratio = 1.0
    min_gap_dist = 300

    # # ############### TBM ###############
    # INPUT_DIR = "dataset/TBM/expert_label"
    # OUTPUT_DIR = "dataset/TBM/boundary_broken"
    # breakage_ratio = 0.8
    # min_gap_dist = 90


    process_dataset_parallel(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        max_thick_ratio=1.0,
        min_gap_dist=min_gap_dist,
        breakage_ratio=breakage_ratio,  # Broken intensity (0.5: subtle, 1.0: default, 1.5-2.0: intense)
        num_workers=None     # None -> Automatically assign available workers
    )