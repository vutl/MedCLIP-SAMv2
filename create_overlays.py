"""
Create overlay visualizations for smoke test results
"""
from PIL import Image
import numpy as np
from pathlib import Path

def create_overlay(input_path, mask_path, output_path):
    """Create red overlay of mask on original image"""
    # Load images
    img = Image.open(input_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    # Resize mask to match image if needed
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)
    
    # Create red overlay
    red = Image.new('RGBA', img.size, (255, 0, 0, 120))
    img_rgba = img.convert('RGBA')
    overlay = Image.composite(red, img_rgba, mask)
    
    # Convert to RGB if output is JPEG
    if str(output_path).lower().endswith(('.jpg', '.jpeg')):
        output_path = str(output_path).rsplit('.', 1)[0] + '.png'
        output_path = Path(output_path)
    
    # Save
    overlay.save(output_path)
    
    # Print stats
    mask_arr = np.array(mask)
    print(f"  Mask stats: min={mask_arr.min()}, max={mask_arr.max()}, nonzero_pixels={np.count_nonzero(mask_arr)}")

base = Path('tmp_smoke')

datasets = ['breast_tumors', 'lung_chest_xray', 'lung_ct']

for ds in datasets:
    ds_path = base / ds
    if not ds_path.exists():
        continue
    
    print(f"\n{ds}:")
    
    # Find input, saliency, postproc, SAM files
    input_files = list((ds_path / 'input').glob('*'))
    if not input_files:
        print("  No input found")
        continue
    
    input_file = input_files[0]
    filename = input_file.name
    
    saliency_file = ds_path / 'output' / filename
    postproc_file = ds_path / 'postproc' / filename
    sam_file = ds_path / 'sam_output' / filename
    
    # Create overlays
    if saliency_file.exists():
        overlay_sal = ds_path / f'overlay_saliency_{filename}'
        create_overlay(input_file, saliency_file, overlay_sal)
        print(f"  Created {overlay_sal.name}")
    
    if postproc_file.exists():
        overlay_post = ds_path / f'overlay_postproc_{filename}'
        create_overlay(input_file, postproc_file, overlay_post)
        print(f"  Created {overlay_post.name}")
    
    if sam_file.exists():
        overlay_sam = ds_path / f'overlay_SAM_{filename}'
        create_overlay(input_file, sam_file, overlay_sam)
        print(f"  Created {overlay_sam.name}")

print("\n=== SMOKE TEST SUMMARY ===")
print(f"Completed: {len(datasets)} datasets")
print("Overlays created in tmp_smoke/<dataset>/overlay_*.png")
