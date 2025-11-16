"""
Download selected Kaggle datasets (using kagglehub) and prepare directory structure
expected by the MedCLIP-SAMv2 repo README.

- Downloads:
  - aryashah2k/breast-ultrasound-images-dataset -> data/breast_tumors/raw
  - anasmohammedtahir/covidqu -> data/lung_chest_xray/raw
  - polomarco/chest-ct-segmentation -> data/lung_ct/raw
- The brain tumor dataset is expected to be already downloaded by the user
  and this script will try to find and extract it rather than re-download.

Notes:
- Requires `kagglehub` to be installed and configured, or replace calls
  with the official `kaggle` CLI.
- The script will create the folder layout from the README and copy any
  discovered image/mask files into `train_images`/`train_masks` by default.
  It does not attempt to create a train/val/test split automatically.

Run example:
    python scripts/download_and_prepare_datasets.py --root data

"""

import os
import shutil
import zipfile
import argparse
import glob
from pathlib import Path

try:
    import kagglehub
except Exception:
    kagglehub = None

IMAGE_EXTS = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
MASK_KEYWORDS = ['mask', 'seg', 'segmentation', 'label']

DATASETS = {
    'aryashah2k/breast-ultrasound-images-dataset': 'breast_tumors',
    'anasmohammedtahir/covidqu': 'lung_chest_xray',
    'polomarco/chest-ct-segmentation': 'lung_ct',
    # Brain tumor dataset is handled separately (user said already downloaded)
}

BRAIN_SHORTNAME = 'brain_tumor'


def ensure_dirs(base):
    """Create dataset directory skeleton from README."""
    parts = ['train_images', 'train_masks', 'val_images', 'val_masks', 'test_images', 'test_masks', 'raw']
    for p in parts:
        d = base / p
        d.mkdir(parents=True, exist_ok=True)


def extract_archive(src_path, dest_dir):
    src_path = Path(src_path)
    dest_dir = Path(dest_dir)
    if src_path.is_file():
        # try common archive formats
        try:
            shutil.unpack_archive(str(src_path), str(dest_dir))
            return True
        except Exception:
            pass
        # try zipfile fallback
        if zipfile.is_zipfile(src_path):
            with zipfile.ZipFile(src_path, 'r') as z:
                z.extractall(dest_dir)
                return True
        return False
    elif src_path.is_dir():
        # already extracted
        # copy contents into dest_dir (or move)
        for item in src_path.iterdir():
            target = dest_dir / item.name
            if not target.exists():
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
        return True
    return False


def find_files_recursive(folder, patterns):
    files = []
    for p in patterns:
        files.extend(Path(folder).rglob(p))
    return [f for f in files if f.is_file()]


def find_masks_recursive(folder):
    files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        for f in Path(folder).rglob(ext):
            name = f.name.lower()
            if any(k in name for k in MASK_KEYWORDS):
                files.append(f)
    return files


def copy_files(files, dest_dir, max_files=None):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in files:
        if max_files and count >= max_files:
            break
        try:
            shutil.copy2(f, dest_dir / f.name)
            count += 1
        except Exception as e:
            print(f"Failed to copy {f}: {e}")
    return count


def handle_dataset(dataset_id, shortname, root):
    print(f"Handling dataset {dataset_id} -> {shortname}")
    dest_base = Path(root) / shortname
    ensure_dirs(dest_base)
    raw_dir = dest_base / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = None
    if kagglehub is None:
        print("Warning: kagglehub not installed. Skipping download. Install via `pip install kagglehub` or use kaggle CLI.")
    else:
        print(f"Downloading {dataset_id} via kagglehub...")
        try:
            downloaded_path = kagglehub.dataset_download(dataset_id)
            print(f"kagglehub returned: {downloaded_path}")
        except Exception as e:
            print(f"kagglehub download failed for {dataset_id}: {e}")
            downloaded_path = None

    # If returned path looks like an archive or folder, extract it
    if downloaded_path:
        success = extract_archive(downloaded_path, raw_dir)
        if not success:
            print(f"Could not extract {downloaded_path} into {raw_dir}. You may need to extract manually.")
    else:
        print(f"No downloaded artifact for {dataset_id}. Please place dataset files into {raw_dir} manually.")

    # scan raw_dir for images and masks
    images = find_files_recursive(raw_dir, IMAGE_EXTS)
    masks = find_masks_recursive(raw_dir)

    print(f"Found {len(images)} images and {len(masks)} masks in {raw_dir}")

    # copy images -> train_images (user can re-split later)
    train_images_dir = dest_base / 'train_images'
    train_masks_dir = dest_base / 'train_masks'

    copied_images = copy_files(images, train_images_dir)
    copied_masks = copy_files(masks, train_masks_dir)

    print(f"Copied {copied_images} images to {train_images_dir}")
    print(f"Copied {copied_masks} masks to {train_masks_dir}")
    print("--- Done ---\n")


def handle_brain_already_downloaded(root):
    # Try to find a locally downloaded brain tumor archive or folder
    cand_names = ['figshare-brain-tumor-dataset', 'brain-tumor-dataset', 'brain_tumor', 'brain_tumors']
    found = None
    cwd = Path.cwd()
    for name in cand_names:
        # check cwd and data root
        for loc in [cwd, Path(root)]:
            for p in loc.rglob(f"*{name}*"):
                if p.is_file() or p.is_dir():
                    found = p
                    break
            if found:
                break
        if found:
            break

    dest_base = Path(root) / BRAIN_SHORTNAME
    ensure_dirs(dest_base)
    raw_dir = dest_base / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    if found:
        print(f"Found brain dataset at {found}. Extracting/copying to {raw_dir}...")
        success = extract_archive(found, raw_dir)
        if not success:
            print(f"Could not extract {found}. Please extract manually into {raw_dir}.")
    else:
        print("Could not automatically find the brain tumor dataset on disk. Please place it at e.g. data/brain_tumor/raw/")

    images = find_files_recursive(raw_dir, IMAGE_EXTS)
    masks = find_masks_recursive(raw_dir)

    print(f"Brain: found {len(images)} images and {len(masks)} masks in {raw_dir}")
    copied_images = copy_files(images, dest_base / 'train_images')
    copied_masks = copy_files(masks, dest_base / 'train_masks')
    print(f"Copied {copied_images} images and {copied_masks} masks for brain dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data', help='Root data folder (relative to repo root)')
    parser.add_argument('--skip-download', action='store_true', help='Skip kaggle download (only prepare dirs and try to extract local archives)')
    parser.add_argument('--no-copy', action='store_true', help='Do not copy files into train_images/train_masks (leave in raw)')
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for ds, short in DATASETS.items():
        if args.skip_download:
            print(f"Skipping download for {ds}. Ensure dataset is available under {root/short}/raw/")
            # still run handle to copy if raw exists
            handle_dataset(ds, short, root)
        else:
            handle_dataset(ds, short, root)

    # handle brain dataset (user already downloaded)
    handle_brain_already_downloaded(root)

    print("All done. Please inspect data/ and split into train/val/test as required by your experiments.")


if __name__ == '__main__':
    main()
