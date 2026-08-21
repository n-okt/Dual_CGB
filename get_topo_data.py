import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from utils.topoloss_pytorch import precompute_gt_topology


def process_single_image(gt_path, save_base_path, topo_size):
    try:
        filename = os.path.basename(gt_path)
        name, _ = os.path.splitext(filename)
        save_path = os.path.join(save_base_path, f"{name}.pkl")

        if os.path.exists(save_path):
            return True

        gt = np.asarray(Image.open(gt_path).convert("L"))
        gt = (gt > 127).astype(np.float32)
        gt_tensor = torch.from_numpy(gt).unsqueeze(0)

        gt_topo_data = precompute_gt_topology(gt_tensor, topo_size=topo_size, n_jobs=1)

        # save
        with open(save_path, "wb") as f:
            pickle.dump(gt_topo_data, f)

        return True
    except Exception as e:
        print(f"Error processing {gt_path}: {e}")
        return False


def run_precompute(base_path, gt_dir, save_dir, topo_size=100, ext="png", max_workers=None):
    search_pattern = os.path.join(base_path, gt_dir, f"*.{ext}")
    gt_paths = glob.glob(search_pattern)

    print(f"Found {len(gt_paths)} images with extension '.{ext}'")

    save_base_path = os.path.join(base_path, save_dir)
    os.makedirs(save_base_path, exist_ok=True)

    if max_workers is None:
        max_workers = os.cpu_count()
    print(f"Executing in parallel using {max_workers} CPU cores...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_image, path, save_base_path, topo_size)
            for path in gt_paths
        ]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing ground truth images"):
            pass


if __name__ == "__main__":

    ########## 316L_Grains ##########
    base_path = "dataset/316L_Grains"
    gt_dir = "RGMask"
    save_dir = "topo"
    ext = "jpg"
    topo_size = 200

    # ########## TBM ##########
    # base_path = "dataset/TBM"
    # gt_dir = "expert_label"
    # save_dir = "topo"
    # ext = "png"
    # topo_size = 128
    
    run_precompute(base_path, gt_dir, save_dir, topo_size=topo_size, ext=ext, max_workers=None)