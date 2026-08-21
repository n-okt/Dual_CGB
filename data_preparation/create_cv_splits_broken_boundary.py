import os
import pandas as pd


def update_cv_csv_paths(
    csv_dir: str,
    output_dir: str,
    new_image_dir: str,
    target_column: str = "photo_path",
    mask_column: str = "mask_path"
):
    """
    Updates the image path column in cross-validation CSV files by replacing the
    directory portion with a new target path and using the filename from the mask path.

    Args:
        csv_dir (str): Folder containing input CSV files (e.g., 'dataset/cv').
        output_dir (str): Folder where updated CSV files will be saved.
        new_image_dir (str): New directory path to replace existing image paths.
        target_column (str): Column name containing the input photo paths to be updated.
        mask_column (str): Column name containing the ground truth mask paths to extract filenames from.
    """
    if not os.path.exists(csv_dir):
        print(f"Error: Input CSV directory does not exist -> {csv_dir}")
        return

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Search for all CSV files in the input directory
    csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in directory: {csv_dir}")
        return

    print(f"=== Starting Path Update for {len(csv_files)} CSV file(s) ===")
    print(f"Input CSV Folder:   {csv_dir}")
    print(f"Output CSV Folder:  {output_dir}")
    print(f"Target Column:      '{target_column}'")
    print(f"Mask Column:        '{mask_column}'")
    print(f"New Directory Path: '{new_image_dir}'\n")

    success_count = 0

    for filename in sorted(csv_files):
        input_csv_path = os.path.join(csv_dir, filename)
        output_csv_path = os.path.join(output_dir, filename)

        try:
            # Read CSV file
            df = pd.read_csv(input_csv_path)

            # Check if required columns exist
            if target_column not in df.columns:
                print(f"Error processing '{filename}': Column '{target_column}' not found.")
                continue
            if mask_column not in df.columns:
                print(f"Error processing '{filename}': Column '{mask_column}' not found.")
                continue

            # Replace target_column path using the filename from mask_column
            # (Converts Windows backslashes to forward slashes for portability)
            df[target_column] = df[mask_column].apply(
                lambda mask_path_str: os.path.join(
                    new_image_dir, os.path.basename(str(mask_path_str))
                ).replace('\\', '/')
            )

            # Save modified DataFrame to output CSV
            df.to_csv(output_csv_path, index=False)
            print(f"Success: Updated '{filename}' -> Saved to '{output_csv_path}'")
            success_count += 1

        except Exception as e:
            print(f"Error processing '{filename}': {e}")

    print(f"\n=== Processing Complete ({success_count}/{len(csv_files)} CSV files saved) ===")


if __name__ == "__main__":
    # Configure your paths and column names here

    ############### 316L Grains ###############
    CSV_DIR = "dataset/316L_Grains/cv_seg"
    OUTPUT_DIR = "dataset/316L_Grains/cv_res"
    NEW_IMAGE_DIR = "dataset/316L_Grains/boundary_broken"
    TARGET_COLUMN = "photo_path"
    MASK_COLUMN = "mask_path"

    # ############### TBM ###############
    # CSV_DIR = "dataset/TBM/cv_seg"
    # OUTPUT_DIR = "dataset/TBM/cv_res"
    # NEW_IMAGE_DIR = "dataset/TBM/boundary_broken"
    # TARGET_COLUMN = "input_path"
    # MASK_COLUMN = "mask_path"

    # Run path update procedure
    update_cv_csv_paths(
        csv_dir=CSV_DIR,
        output_dir=OUTPUT_DIR,
        new_image_dir=NEW_IMAGE_DIR,
        target_column=TARGET_COLUMN,
        mask_column=MASK_COLUMN
    )