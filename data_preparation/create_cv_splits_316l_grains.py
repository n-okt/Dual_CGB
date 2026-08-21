import os
import csv
import random
import re
from pathlib import Path

def create_fold_splits(root_path, output_path, num_folds=5, seed=42):
    dataset_dir = Path(root_path)
    
    # Cropped image directories
    photo_dir = dataset_dir / "RG"
    mask_dir = dataset_dir / "RGMask"
    hed_dir = dataset_dir / "HED_PRE"
    grad_dir = dataset_dir / "GRAD_PRE"
    thresh_dir = dataset_dir / "THRESH_PRE"
    
    # Merged original image (pre-crop) directories
    rg_merged_dir = dataset_dir / "RG_merged"
    rg_mask_merged_dir = dataset_dir / "RGMask_merged"
    
    # Check if directories exist
    for d in [photo_dir, mask_dir, hed_dir, grad_dir, thresh_dir, rg_merged_dir, rg_mask_merged_dir]:
        if not d.exists():
            print(f"Warning: {d} not found. Please check the path.")

    # 1. Get all cropped images in the RG folder
    photo_filenames = [f.name for f in photo_dir.glob("*.jpg")]
    
    if not photo_filenames:
        print("Error: No images found in the specified directory.")
        return

    # 2. Extract parent image (original image) group IDs (e.g., 1 to 40) to prevent data leakage
    pattern = re.compile(r"RG(\d+)_\d+_\d+\.jpg")
    group_ids = set()
    
    for filename in photo_filenames:
        match = pattern.match(filename)
        if match:
            group_ids.add(match.group(1))
            
    group_ids = sorted(list(group_ids), key=int)
    
    # 3. Shuffle group IDs and split into K-Folds
    random.seed(seed)
    random.shuffle(group_ids)
    
    folds = {i: [] for i in range(1, num_folds + 1)}
    for i, gid in enumerate(group_ids):
        fold_idx = (i % num_folds) + 1
        folds[fold_idx].append(gid)

    os.makedirs(output_path, exist_ok=True)

    prefix_mapping = {
        "HED_PRE": "HEDPre",
        "GRAD_PRE": "GradPre",
        "THRESH_PRE": "ThreshPre"
    }

    # 4. Create a CSV file for each fold
    for fold in range(1, num_folds + 1):
        test_groups = set(folds[fold])
        initial_train_groups = list(set(group_ids) - test_groups)
        initial_train_groups.sort(key=int)
        
        rng = random.Random(seed + fold)
        rng.shuffle(initial_train_groups)
        
        num_val = max(1, int(len(initial_train_groups) * 0.1))
        val_groups = set(initial_train_groups[:num_val])
        train_groups = set(initial_train_groups[num_val:])
        
        # Data leakage check
        assert test_groups.isdisjoint(val_groups), f"Data Leakage: Test and Val overlap in Fold {fold}"
        assert test_groups.isdisjoint(train_groups), f"Data Leakage: Test and Train overlap in Fold {fold}"
        assert val_groups.isdisjoint(train_groups), f"Data Leakage: Val and Train overlap in Fold {fold}"

        # Create a list of cropped images assigned to Train
        train_files = [f for f in photo_filenames if pattern.match(f) and pattern.match(f).group(1) in train_groups]
        rng.shuffle(train_files)
        
        num_train = len(train_files)
        num_rg = int(num_train * 0.5)                   # 50%
        num_hed = int(num_train * (1 / 6))              # 16.7%
        num_grad = int(num_train * (1 / 6))             # 16.7%
        
        input_mapping = {}
        for i, f in enumerate(train_files):
            if i < num_rg:
                input_mapping[f] = "RG"
            elif i < num_rg + num_hed:
                input_mapping[f] = "HED_PRE"
            elif i < num_rg + num_hed + num_grad:
                input_mapping[f] = "GRAD_PRE"
            else:
                input_mapping[f] = "THRESH_PRE"
        
        output_csv = os.path.join(output_path, f"fold_{fold}_paths.csv")
        
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["phase", "photo_path", "mask_path", "base_group"])
            
            # --- TRAIN: Output cropped images ---
            for filename in train_files:
                match = pattern.match(filename)
                gid = match.group(1)
                
                input_type = input_mapping[filename]
                
                # Resolve photo path
                if input_type == "RG":
                    photo_path = photo_dir / filename
                else:
                    input_dir = dataset_dir / input_type
                    correct_prefix = prefix_mapping[input_type]
                    photo_filename = filename.replace("RG", correct_prefix)
                    photo_path = input_dir / photo_filename
                
                # Resolve mask path
                mask_filename = filename.replace("RG", "RGMask")
                mask_path = mask_dir / mask_filename
                
                writer.writerow([
                    "train",
                    photo_path.as_posix(),
                    mask_path.as_posix(),
                    f"RG{gid}"
                ])
                
            # --- VAL: Output original images (merged images) ---
            for gid in sorted(list(val_groups), key=int):
                photo_path = rg_merged_dir / f"RG{gid}.jpg"
                mask_path = rg_mask_merged_dir / f"RGMask{gid}.jpg"
                writer.writerow([
                    "val",
                    photo_path.as_posix(),
                    mask_path.as_posix(),
                    f"RG{gid}"
                ])

            # --- TEST: Output original images (merged images) ---
            for gid in sorted(list(test_groups), key=int):
                photo_path = rg_merged_dir / f"RG{gid}.jpg"
                mask_path = rg_mask_merged_dir / f"RGMask{gid}.jpg"
                writer.writerow([
                    "test",
                    photo_path.as_posix(),
                    mask_path.as_posix(),
                    f"RG{gid}"
                ])
                
        train_types = list(input_mapping.values())
        print(f"Exported CSV for Fold {fold}: {output_csv}")
        print(f"  - Train groups: {len(train_groups)} | Samples(Cropped): {len(train_files)} (RG: {train_types.count('RG')}, HED_PRE: {train_types.count('HED_PRE')}, GRAD_PRE: {train_types.count('GRAD_PRE')}, THRESH_PRE: {train_types.count('THRESH_PRE')})")
        print(f"  - Val groups: {len(val_groups)} | Samples(Merged): {len(val_groups)}")
        print(f"  - Test groups: {len(test_groups)} | Samples(Merged): {len(test_groups)}\n")

if __name__ == "__main__":
    # Define dataset paths
    root_path = "dataset/316L_Grains"
    output_path = "dataset/316L_Grains/cv_seg"
    
    # Generate CSV files with 5-fold cross-validation split
    create_fold_splits(root_path, output_path, num_folds=5)