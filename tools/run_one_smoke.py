"""
Simple smoke test runner for one dataset
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_one_smoke.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    base = Path("D:/Documents/LMIS/MedCLIP-SAMv2")
    
    configs = {
        'lung_chest_xray': {
            'folder': base / 'data/lung_chest_xray/train_images',
            'prompt': 'Bilateral pulmonary infection, infected areas in lung.'
        },
        'lung_ct': {
            'folder': base / 'data/lung_ct/train_images',
            'prompt': 'A medical CT scan displaying a clear and detailed image of the lung lobes.'
        }
    }
    
    if dataset_name not in configs:
        print(f"Unknown dataset: {dataset_name}")
        sys.exit(1)
    
    config = configs[dataset_name]
    train_path = config['folder']
    
    if not train_path.exists():
        print(f"SKIP {dataset_name}: train_images not found")
        return
    
    # Get first valid image
    images = [f for f in train_path.glob('*') if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg'] and f.stat().st_size > 0]
    if not images:
        print(f"NO_IMAGE_FOUND in {dataset_name}")
        return
    
    src_image = images[0]
    print(f"Selected image: {src_image.name}")
    
    # Setup directories
    outdir = base / 'tmp_smoke' / dataset_name
    for subdir in ['input', 'output', 'postproc', 'sam_output']:
        (outdir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Copy image
    dest = outdir / 'input' / src_image.name
    shutil.copy2(src_image, dest)
    
    # Create prompt JSON
    prompt_json = outdir / 'prompt.json'
    with open(prompt_json, 'w', encoding='utf-8') as f:
        json.dump({src_image.name: config['prompt']}, f, ensure_ascii=False)
    
    python_exe = r'D:\anaconda3\envs\medclipsamv2\python.exe'
    
    # Run saliency
    print("[1/3] Generating saliency map...")
    subprocess.run([
        python_exe, 'saliency_maps/generate_saliency_maps.py',
        '--input-path', str(outdir / 'input'),
        '--output-path', str(outdir / 'output'),
        '--model-name', 'BiomedCLIP',
        '--finetuned',
        '--device', 'cuda',
        '--reproduce',
        '--json-path', str(prompt_json)
    ], cwd=str(base), check=True)
    
    # Run postprocessing
    print("[2/3] Postprocessing...")
    subprocess.run([
        python_exe, 'postprocessing/postprocess_saliency_maps.py',
        '--sal-path', str(outdir / 'output'),
        '--output-path', str(outdir / 'postproc'),
        '--postprocess', 'kmeans'
    ], cwd=str(base), check=True)
    
    # Run SAM
    print("[3/3] Running SAM...")
    subprocess.run([
        python_exe, 'segment-anything/prompt_sam.py',
        '--input', str(outdir / 'input'),
        '--mask-input', str(outdir / 'postproc'),
        '--output', str(outdir / 'sam_output'),
        '--model-type', 'vit_h',
        '--checkpoint', 'segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth',
        '--prompts', 'boxes',
        '--device', 'cuda'
    ], cwd=str(base), check=True)
    
    print(f"[OK] {dataset_name} completed successfully")
    
    # List outputs
    for f in (outdir).rglob('*'):
        if f.is_file():
            print(f"  {f.relative_to(base)}  ({f.stat().st_size} bytes)")

if __name__ == '__main__':
    main()
