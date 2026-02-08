import os
import shutil


def sort_shenzhen_images(source_dir="/Downloads/archive/images/images"):
    """Sort Shenzhen Chest X-ray images into 'Normal' and 'Tuberculosis'.

    Files ending with `_0.png` (case-insensitive) are treated as Normal and
    files ending with `_1.png` are treated as Tuberculosis. Non-PNG files are
    ignored. The function will create the target folders if needed and will
    report counts and any failed moves.
    """

    # 1. Define destination paths
    base_dir = "data"
    images_dir = os.path.join(base_dir, "images")
    normal_dir = os.path.join(images_dir, "Normal")
    tb_dir = os.path.join(images_dir, "Tuberculosis")

    # 2. Create the directory structure if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)
    for folder in [normal_dir, tb_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"Created folder: {folder}")

    # 3. Counters for statistics
    count_normal = 0
    count_tb = 0
    count_skipped = 0
    count_failed = 0

    print(f"Scanning directory: {os.path.abspath(source_dir)}...")

    # 4. Iterate through files and move them
    if not os.path.isdir(source_dir):
        print(f"Error: source directory does not exist: {source_dir}")
        return

    try:
        files = os.listdir(source_dir)
    except Exception as e:
        print(f"Error reading source directory: {e}")
        return

    for filename in files:
        # standard Shenzhen format: CHNCXR_####_0.png (Normal) or _1.png (TB)
        if not filename.lower().endswith(".png"):
            continue

        src_path = os.path.join(source_dir, filename)
        if not os.path.isfile(src_path):
            continue

        # Determine destination based on filename suffix (case-insensitive)
        lower = filename.lower()
        try:
            if lower.endswith("_0.png"):
                dst_path = os.path.join(normal_dir, filename)
                shutil.move(src_path, dst_path)
                count_normal += 1

            elif lower.endswith("_1.png"):
                dst_path = os.path.join(tb_dir, filename)
                shutil.move(src_path, dst_path)
                count_tb += 1

            else:
                count_skipped += 1
        except Exception as e:
            print(f"Failed to move {src_path} -> {dst_path if 'dst_path' in locals() else 'unknown'}: {e}")
            count_failed += 1

    # 5. Summary
    print("-" * 30)
    print("Processing Complete.")
    print(f"Moved to 'Normal':       {count_normal}")
    print(f"Moved to 'Tuberculosis': {count_tb}")
    print(f"Skipped/Other files:     {count_skipped}")
    print(f"Images are now organized in: {images_dir}")
    print("-" * 30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize Shenzhen CXR .png images into Normal/Tuberculosis folders"
    )
    parser.add_argument(
        "-s", "--source",
        default="Downloads/archive/images/images",
        help="Path to the folder containing raw .png images (default: Downloads/archive/images/images)",
    )

    args = parser.parse_args()
    source_directory = args.source

    if not os.path.isdir(source_directory):
        print(f"Error: source directory does not exist: {source_directory}")
        print("Please create the directory or pass a valid path via --source.")
    else:
        pngs = [f for f in os.listdir(source_directory) if f.lower().endswith(".png")]
        if len(pngs) == 0:
            print(f"No .png files found in {source_directory}")
            print("Place .png images there or pass a different path with --source.")
        else:
            sort_shenzhen_images(source_directory)