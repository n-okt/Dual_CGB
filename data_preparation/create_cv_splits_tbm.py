import os
import csv
import random
import re
from pathlib import Path

def create_fold_splits(root_path, output_path, num_folds=5, seed=42):
    dataset_dir = Path(root_path)
    image_dir = dataset_dir / "input_image"
    label_dir = dataset_dir / "expert_label"
    
    # Check if directories exist
    if not image_dir.exists():
        print(f"Error: {image_dir} not found. Please check the path.")
        return
    if not label_dir.exists():
        print(f"Error: {label_dir} not found. Please check the path.")
        return

    # 1. Get all image files in the input_image folder (PNG format)
    image_filenames = [f.name for f in image_dir.glob("*.png")]
    
    if not image_filenames:
        print("Error: No images found in the specified directory.")
        return

    # 2. Extract parent image group IDs to prevent data leakage
    # Naming convention assumes the first number before the hyphen is the parent ID
    # e.g., "10-0-768-0-0.png" -> group ID is "10"
    pattern = re.compile(r"^(\d+)-.*\.png$")
    group_ids = set()
    
    for filename in image_filenames:
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
        
        assert test_groups.isdisjoint(val_groups), f"Data Leakage: Test and Val overlap in Fold {fold}"
        assert test_groups.isdisjoint(train_groups), f"Data Leakage: Test and Train overlap in Fold {fold}"
        assert val_groups.isdisjoint(train_groups), f"Data Leakage: Val and Train overlap in Fold {fold}"
        
        output_csv = os.path.join(output_path, f"fold_{fold}_paths.csv")
        
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Define column names suited for the TBM dataset structure
            writer.writerow(["phase", "input_path", "label_path", "base_group"])
            
            for filename in image_filenames:
                match = pattern.match(filename)
                if not match:
                    continue
                
                gid = match.group(1)
                
                # Both input_image and expert_label share the exact same filename
                image_path = image_dir / filename
                label_path = label_dir / filename
                
                # Determine Train / Val / Test phase based on the parent image's group ID
                if gid in test_groups:
                    phase = "test"
                elif gid in val_groups:
                    phase = "val"
                elif gid in train_groups:
                    phase = "train"
                else:
                    continue
                
                writer.writerow([
                    phase,
                    image_path.as_posix(),
                    label_path.as_posix(),
                    f"Group_{gid}" # Base group name for tracking
                ])
                
        print(f"Exported CSV for Fold {fold}: {output_csv} (Train groups: {len(train_groups)}, Val groups: {len(val_groups)}, Test groups: {len(test_groups)})")

if __name__ == "__main__":
    # Define dataset paths
    root_path = "dataset/TBM"
    output_path = "dataset/TBM/cv_seg"
    
    # Generate CSV files with 5-fold cross-validation split
    create_fold_splits(root_path, output_path, num_folds=5)