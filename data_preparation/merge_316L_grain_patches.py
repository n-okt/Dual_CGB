import re
from collections import defaultdict
from pathlib import Path
from PIL import Image


def merge_images_in_directory(target_dir: Path, output_dir: Path) -> None:
    """
    Parse patch image files in target_dir, merge them into full images,
    and save the results to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Regex pattern matching filename format: <base_name>_<row>_<col>.<ext>
    # Example: RG1_1_3.jpg -> base_name="RG1", row=1, col=3
    pattern = re.compile(
        r"^(.+?)_(\d+)_(\d+)\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE
    )

    # Group patch files by base name
    groups = defaultdict(list)

    for file_path in target_dir.iterdir():
        if file_path.is_file():
            match = pattern.match(file_path.name)
            if match:
                base_name, row, col, ext = match.groups()
                groups[base_name].append((int(row), int(col), file_path, ext))

    if not groups:
        print(f"[{target_dir}] No matching image files found.")
        return

    # Process each image group (e.g., RG1, RG2, RGMask1, etc.)
    for base_name, files in sorted(groups.items()):
        # Determine grid dimensions from maximum row and column indices
        max_row = max(f[0] for f in files)
        max_col = max(f[1] for f in files)

        # Map grid coordinates (row, col) to file paths
        patch_map = {(f[0], f[1]): f[2] for f in files}

        # Obtain image dimensions and extension from the first patch
        sample_img = Image.open(files[0][2])
        patch_w, patch_h = sample_img.size
        sample_ext = files[0][3]

        # Create canvas for the merged image
        canvas_w = patch_w * max_col
        canvas_h = patch_h * max_row
        canvas = Image.new(sample_img.mode, (canvas_w, canvas_h))

        # Paste each patch onto the canvas (1-indexed grid)
        for r in range(1, max_row + 1):
            for c in range(1, max_col + 1):
                if (r, c) in patch_map:
                    with Image.open(patch_map[(r, c)]) as patch:
                        x = (c - 1) * patch_w
                        y = (r - 1) * patch_h
                        canvas.paste(patch, (x, y))

        # Save the reconstructed full-size image
        out_file = output_dir / f"{base_name}.{sample_ext}"
        canvas.save(out_file)
        print(
            f"  [Saved] {out_file.name} (Grid: {max_row}x{max_col}, Resolution: {canvas_w}x{canvas_h})"
        )


def main():
    # Set base directory for dataset
    base_dataset_dir = Path("dataset/316L_Grains")

    # Target subdirectories to process
    subfolders = ["RG", "RGMask"]

    # Directory to store reconstructed images
    output_dirs = ["RG_merged", "RGMask_merged"]

    for folder, out_dir in zip(subfolders, output_dirs):
        input_dir = base_dataset_dir / folder
        output_dir = base_dataset_dir / out_dir

        if input_dir.exists():
            print(f"--- Processing: {input_dir} ---")
            merge_images_in_directory(input_dir, output_dir)
        else:
            print(f"[Skipped] Directory does not exist: {input_dir}")


if __name__ == "__main__":
    main()