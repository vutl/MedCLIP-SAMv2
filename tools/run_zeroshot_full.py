"""
Run the full zero-shot pipeline (saliency -> postprocess -> SAM)
over all datasets found under `data/*/train_images`.

This script is intended to be executed with the `medclipsamv2` Python interpreter
so GPU and installed dependencies (SAM, BiomedCLIP) are available.

It mirrors the steps in `tools/run_one_smoke.py` but iterates all images
and skips images already processed.
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

BASE = Path(r"D:/Documents/LMIS/MedCLIP-SAMv2")
DATA_DIR = BASE / 'data'
OUT_ROOT = BASE / 'tmp_full'

# You can update or extend dataset_prompts for better text prompts per dataset.
DATASET_PROMPTS = {
    'breast_tumors': 'Breast ultrasound image showing a tumor region.',
    'brain_tumors': 'MRI scan with a brain tumor region.',
    'lung_chest_xray': 'Bilateral pulmonary infection, infected areas in lung.',
    'lung_ct': 'A medical CT scan displaying a clear and detailed image of the lung lobes.'
}

PYTHON_EXE = r'D:\anaconda3\envs\medclipsamv2\python.exe'

def run_cmd(cmd, cwd=BASE):
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}\n  {e}")
        return False

def process_dataset(ds_name, train_dir):
    print(f"Processing dataset (batch): {ds_name} -> {train_dir}")

    # Create output dirs for this dataset
    ds_out = OUT_ROOT / ds_name
    in_dir = ds_out / 'input'
    sal_out = ds_out / 'output'
    post_out = ds_out / 'postproc'
    sam_out = ds_out / 'sam_output'
    for d in [in_dir, sal_out, post_out, sam_out]:
        d.mkdir(parents=True, exist_ok=True)

    # copy inputs (or assume train_dir already contains the images)
    # If train_dir is already the dataset folder, we avoid duplicating large copies.
    if train_dir.name != 'batch_input':
        # we'll reference the dataset train_dir directly when calling saliency; but keep a copy for record
        src_images = [p for p in sorted(train_dir.glob('*')) if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg'] and p.stat().st_size>0]
        for p in src_images:
            dest = in_dir / p.name
            if not dest.exists():
                shutil.copy2(p, dest)
    else:
        # temp batch_input already contains only selected files; copy into input folder
        src_images = [p for p in sorted(train_dir.glob('*')) if p.is_file()]
        for p in src_images:
            dest = in_dir / p.name
            if not dest.exists():
                shutil.copy2(p, dest)

    # Build a prompt.json for the dataset using dataset-level prompt
    prompt = DATASET_PROMPTS.get(ds_name, 'A medical image showing the structure to segment.')
    prompt_json = ds_out / 'prompt.json'
    # create mapping for all images in input folder
    mapping = {p.name: prompt for p in sorted(in_dir.glob('*'))}
    with open(prompt_json, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False)

    # 1) Saliency on the whole folder
    print(f"  [1/3] Saliency for dataset {ds_name} (images={len(mapping)})")
    cmd_sal = [PYTHON_EXE, 'saliency_maps/generate_saliency_maps.py',
               '--input-path', str(in_dir),
               '--output-path', str(sal_out),
               '--model-name', 'BiomedCLIP',
               '--finetuned',
               '--device', 'cuda',
               '--reproduce',
               '--json-path', str(prompt_json)
              ]
    ok = run_cmd(cmd_sal)
    if not ok:
        print(f"    Saliency failed for dataset {ds_name}, skipping")
        return

    # 2) Postprocessing (kmeans) on the whole saliency folder
    print(f"  [2/3] Postprocessing for dataset {ds_name}")
    cmd_pp = [PYTHON_EXE, 'postprocessing/postprocess_saliency_maps.py',
              '--sal-path', str(sal_out),
              '--output-path', str(post_out),
              '--postprocess', 'kmeans'
             ]
    ok = run_cmd(cmd_pp)
    if not ok:
        print(f"    Postprocessing failed for dataset {ds_name}, skipping")
        return

    # 3) SAM on the whole folder
    print(f"  [3/3] SAM for dataset {ds_name}")
    cmd_sam = [PYTHON_EXE, 'segment-anything/prompt_sam.py',
               '--input', str(in_dir),
               '--mask-input', str(post_out),
               '--output', str(sam_out),
               '--model-type', 'vit_h',
               '--checkpoint', 'segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth',
               '--prompts', 'boxes',
               '--device', 'cuda'
              ]
    ok = run_cmd(cmd_sam)
    if not ok:
        print(f"    SAM failed for dataset {ds_name}, skipping")
        return

    print(f"  Done dataset: {ds_name} (outputs in {ds_out})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-per-dataset', type=int, default=0,
                        help='If >0, process at most this many images per dataset (for testing).')
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logf = OUT_ROOT / 'run.log'
    # Try to open the canonical log; if it's locked or permission denied
    # (race with another process), fall back to a timestamped file.
    try:
        lf = open(logf, 'a', encoding='utf-8')
        lf.write(f"\n=== Run started: {datetime.now().isoformat()} args={vars(args)} ===\n")
        lf.flush()
    except PermissionError:
        alt_name = f"run_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.log"
        logf = OUT_ROOT / alt_name
        lf = open(logf, 'a', encoding='utf-8')
        lf.write(f"\n=== Run started (fallback): {datetime.now().isoformat()} args={vars(args)} ===\n")
        lf.flush()

    # iterate data/* folders
    for ds in sorted(DATA_DIR.iterdir()):
        if not ds.is_dir():
            continue
        train_images = ds / 'train_images'
        if train_images.exists():
            # pass limit into processing by slicing the Path list
            if args.max_per_dataset and args.max_per_dataset > 0:
                imgs = [p for p in sorted(train_images.glob('*')) if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg'] and p.stat().st_size>0][:args.max_per_dataset]
                if not imgs:
                    print(f"Skipping {ds.name}: no valid images")
                    continue
                # create a temporary folder with only those images copied to process
                temp_dir = OUT_ROOT / ds.name / 'batch_input'
                temp_dir.mkdir(parents=True, exist_ok=True)
                for p in imgs:
                    dest = temp_dir / p.name
                    if not dest.exists():
                        shutil.copy2(p, dest)
                process_dataset(ds.name, temp_dir)
            else:
                process_dataset(ds.name, train_images)
        else:
            print(f"Skipping {ds.name}: no train_images folder")

    try:
        lf.write(f"\n=== Run finished: {datetime.now().isoformat()} ===\n")
        lf.flush()
        lf.close()
    except Exception:
        # If the file handle was lost for any reason, write to a new fallback log
        try:
            fallback = OUT_ROOT / f"run_finish_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
            with open(fallback, 'a', encoding='utf-8') as f2:
                f2.write(f"\n=== Run finished (fallback): {datetime.now().isoformat()} ===\n")
        except Exception:
            pass


if __name__ == '__main__':
    main()
