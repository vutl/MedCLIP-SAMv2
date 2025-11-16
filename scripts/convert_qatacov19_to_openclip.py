#!/usr/bin/env python3
"""Convert QaTa-COV19 CSVs to open-clip training TSVs.
Usage:
  python scripts/convert_qatacov19_to_openclip.py \
    --csv D:/Documents/LMIS/Train_text_for_Covid19.csv \
    --images D:/Documents/LMIS/QaTa-COV19/QaTa-COV19-v2/Train Set/Images \
    --out D:/Documents/LMIS/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/data/qatacov19_train.tsv

The script tries to map entries in the CSV (mask filenames in first column) to image files
under the provided images directory by removing a leading 'mask_' and searching for
matching filenames. It writes a TSV with: <absolute_image_path>\t<prompt> per line.
"""
import argparse
import csv
import os
import sys
from pathlib import Path


def find_image_for_mask(mask_name, images_root):
    # Try variants: original mask, remove leading 'mask_', also remove leading 'mask_sub-'
    candidates = [mask_name]
    if mask_name.startswith('mask_'):
        candidates.append(mask_name[len('mask_'):])
    if mask_name.startswith('mask_sub-'):
        candidates.append(mask_name[len('mask_'):])
    # also try without extension for contains-match
    mask_no_ext = os.path.splitext(mask_name)[0]
    # walk images_root once (cache)
    for root, dirs, files in os.walk(images_root):
        for f in files:
            fname = f
            f_no_ext = os.path.splitext(fname)[0]
            for c in candidates:
                # exact match
                if fname == c:
                    return os.path.join(root, fname)
                # match after removing extension
                if f_no_ext == os.path.splitext(c)[0]:
                    return os.path.join(root, fname)
                # contains match (looser)
                if os.path.splitext(c)[0] in f_no_ext:
                    return os.path.join(root, fname)
    return None


def convert(csv_path, images_root, out_path, delimiter=';'):
    csv_path = Path(csv_path)
    images_root = Path(images_root)
    out_path = Path(out_path)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return 1
    if not images_root.exists():
        print(f"ERROR: Images folder not found: {images_root}")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    mapped = 0
    unmatched = []

    with open(csv_path, newline='', encoding='utf-8') as f_in, open(out_path, 'w', encoding='utf-8') as f_out:
        reader = csv.reader(f_in, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            # skip header heuristic
            first = row[0].strip()
            if first.lower().startswith('image') and (len(row) > 1 and row[1].lower().startswith('description')):
                continue
            total += 1
            mask_name = row[0].strip()
            caption = ''
            if len(row) > 1:
                caption = row[1].strip()
            if not mask_name:
                continue
            img_path = find_image_for_mask(mask_name, str(images_root))
            if img_path:
                mapped += 1
                # write absolute path \t caption
                f_out.write(f"{os.path.abspath(img_path)}\t{caption}\n")
            else:
                unmatched.append(mask_name)

    print(f"Conversion finished. Total rows: {total}, Mapped: {mapped}, Unmatched: {len(unmatched)}")
    if unmatched:
        print("Sample unmatched (up to 20):")
        for u in unmatched[:20]:
            print(" -", u)
        print("If many unmatched, ensure mask filenames map to image filenames (maybe different prefixes).")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    sys.exit(convert(args.csv, args.images, args.out))


if __name__ == '__main__':
    main()
