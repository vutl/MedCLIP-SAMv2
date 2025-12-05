#!/usr/bin/env python3
"""
Script to perform inference with U-Mamba checkpoint and evaluate results
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Add evaluation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluation'))
try:
    from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
    USE_SURFACE_DICE = True
except ImportError:
    print("Warning: SurfaceDice module not available, will use simple metrics")
    USE_SURFACE_DICE = False


def compute_metrics(pred, gt):
    """
    Compute evaluation metrics for binary masks
    """
    # Ensure binary
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    
    # Flatten arrays
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Compute basic metrics
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    
    # Dice coefficient
    if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    elif np.sum(gt_flat) == 0:
        dice = 0.0
        iou = 0.0
        precision = 0.0
        recall = 0.0
    else:
        dice = 2 * intersection / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-8)
        iou = intersection / (union + 1e-8)
        precision = intersection / (np.sum(pred_flat) + 1e-8)
        recall = intersection / (np.sum(gt_flat) + 1e-8)
    
    # Accuracy
    accuracy = np.sum(pred_flat == gt_flat) / len(pred_flat)
    
    metrics = {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }
    
    # Add surface dice if available
    if USE_SURFACE_DICE and np.sum(gt_flat) > 0:
        try:
            pred_3d = pred[..., None]
            gt_3d = gt[..., None]
            spacing = [1, 1, 1]
            surface_distances = compute_surface_distances(gt_3d, pred_3d, spacing)
            nsd = compute_surface_dice_at_tolerance(surface_distances, 2)
            metrics['nsd'] = nsd
            
            # Also use library's dice
            dice_lib = compute_dice_coefficient(gt_3d, pred_3d)
            metrics['dice_lib'] = dice_lib
        except:
            pass
    
    return metrics


def prepare_nnunet_input_from_sam_masks(sam_masks_dir, output_dir):
    """
    Convert SAM masks to nnUNet input format.
    nnUNet expects NIfTI images with naming: <casename>_0000.nii.gz
    """
    print(f"\n=== Preparing nnUNet input from SAM masks ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Import nibabel for NIfTI conversion
    try:
        import nibabel as nib
    except ImportError:
        print("Installing nibabel...")
        subprocess.run(["pip", "install", "nibabel"], check=True)
        import nibabel as nib
    
    sam_masks = sorted(list(Path(sam_masks_dir).glob("*.png")))
    print(f"Found {len(sam_masks)} SAM masks")
    
    for mask_path in tqdm(sam_masks, desc="Converting SAM masks to NIfTI"):
        # Get case name (without extension)
        case_name = mask_path.stem
        # nnUNet expects _0000 suffix for single-channel images
        output_name = f"{case_name}_0000.nii.gz"
        output_path = os.path.join(output_dir, output_name)
        
        # Load PNG image
        img = np.array(Image.open(str(mask_path)))
        
        # Normalize to 0-1 range if needed
        if img.max() > 1:
            img = img.astype(np.float32) / 255.0
        
        # Add extra dimension for 2D slices (height, width, 1)
        img_3d = img[:, :, np.newaxis]
        
        # Create NIfTI image with identity affine
        nii_img = nib.Nifti1Image(img_3d, affine=np.eye(4))
        
        # Save as NIfTI
        nib.save(nii_img, output_path)
    
    print(f"Prepared {len(sam_masks)} input images in {output_dir}")
    return len(sam_masks)


def run_nnunet_prediction(input_dir, output_dir, checkpoint_dir, dataset_id=801, 
                          configuration='2d', fold=0, trainer='nnUNetTrainerUMambaBotSmallBatch'):
    """
    Run nnUNet prediction using the trained checkpoint
    """
    print(f"\n=== Running nnUNet Prediction ===")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct prediction command
    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", str(dataset_id),
        "-c", configuration,
        "-tr", trainer,
        "-f", str(fold),
        "-chk", "checkpoint_best.pth"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run prediction
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Prediction completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during prediction: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def convert_predictions_to_png(prediction_dir, output_dir):
    """
    Convert nnUNet predictions (which might be .nii.gz or other formats) to PNG
    and remove the _0000 suffix to match ground truth naming
    """
    print(f"\n=== Converting predictions to PNG format ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # nnUNet saves predictions as .png files with same name as input
    pred_files = sorted(list(Path(prediction_dir).glob("*.png")))
    
    if len(pred_files) == 0:
        print(f"No .png predictions found, checking for .nii.gz files...")
        import nibabel as nib
        pred_files = sorted(list(Path(prediction_dir).glob("*.nii.gz")))
        
        for pred_path in tqdm(pred_files, desc="Converting NIfTI to PNG"):
            # Load NIfTI
            img = nib.load(str(pred_path))
            data = img.get_fdata()
            
            # Remove singleton dimensions (H, W, 1) -> (H, W)
            data = np.squeeze(data)
            
            # Convert to uint8 (assuming binary mask with values 0 and 1)
            # Multiply by 255 to get 0 or 255
            data = (data * 255).astype(np.uint8)
            
            # Get case name and remove _0000 suffix
            case_name = pred_path.stem.replace('.nii', '').replace('_0000', '')
            output_path = os.path.join(output_dir, f"{case_name}.png")
            
            # Save as PNG
            Image.fromarray(data).save(output_path)
    else:
        print(f"Found {len(pred_files)} PNG predictions")
        for pred_path in tqdm(pred_files, desc="Copying predictions"):
            # Remove _0000 suffix from filename
            case_name = pred_path.stem.replace('_0000', '')
            output_path = os.path.join(output_dir, f"{case_name}.png")
            
            # Copy or convert the prediction
            shutil.copy(str(pred_path), output_path)
    
    final_preds = list(Path(output_dir).glob("*.png"))
    print(f"Converted {len(final_preds)} predictions to {output_dir}")
    return len(final_preds)


def evaluate_predictions(pred_dir, gt_dir):
    """
    Evaluate predictions against ground truth masks
    """
    print(f"\n=== Evaluating Predictions ===")
    print(f"Predictions dir: {pred_dir}")
    print(f"Ground truth dir: {gt_dir}")
    
    # Get all prediction and ground truth files
    pred_files = sorted(list(Path(pred_dir).glob("*.png")))
    gt_files = sorted(list(Path(gt_dir).glob("*.png")))
    
    print(f"Found {len(pred_files)} predictions and {len(gt_files)} ground truth masks")
    
    # Create mapping from filename to path
    pred_dict = {p.name: str(p) for p in pred_files}
    gt_dict = {g.name: str(g) for g in gt_files}
    
    # Find common files
    common_names = set(pred_dict.keys()) & set(gt_dict.keys())
    print(f"Found {len(common_names)} matching pairs")
    
    if len(common_names) == 0:
        print("ERROR: No matching prediction-GT pairs found!")
        print("Sample prediction names:", list(pred_dict.keys())[:5])
        print("Sample GT names:", list(gt_dict.keys())[:5])
        return None
    
    # Compute metrics for each pair
    all_metrics = []
    for name in tqdm(sorted(common_names), desc="Computing metrics"):
        pred_path = pred_dict[name]
        gt_path = gt_dict[name]
        
        # Load images
        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))
        
        # Binarize (threshold at 127)
        pred_binary = (pred > 127).astype(np.uint8)
        gt_binary = (gt > 127).astype(np.uint8)
        
        # Compute metrics
        metrics = compute_metrics(pred_binary, gt_binary)
        metrics['filename'] = name
        all_metrics.append(metrics)
    
    # Aggregate results
    print("\n" + "="*60)
    print("EVALUATION RESULTS - U-Mamba on Brain Tumors Dataset")
    print("="*60)
    
    # Compute mean metrics
    metric_names = ['dice', 'iou', 'precision', 'recall', 'accuracy']
    for metric in metric_names:
        values = [m[metric] for m in all_metrics if metric in m]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("="*60)
    
    # Save detailed results
    results_file = os.path.join(pred_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("U-Mamba Evaluation Results - Brain Tumors Dataset\n")
        f.write("="*60 + "\n\n")
        
        # Summary statistics
        f.write("Summary Statistics:\n")
        f.write("-" * 60 + "\n")
        for metric in metric_names:
            values = [m[metric] for m in all_metrics if metric in m]
            if values:
                f.write(f"{metric.upper():12s}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
        
        # Individual results
        f.write("\n\nDetailed Results per Image:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Filename':<20} {'Dice':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'Accuracy':>10}\n")
        f.write("-" * 60 + "\n")
        
        for m in all_metrics:
            f.write(f"{m['filename']:<20} {m['dice']:>8.4f} {m['iou']:>8.4f} "
                   f"{m['precision']:>10.4f} {m['recall']:>8.4f} {m['accuracy']:>10.4f}\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return all_metrics


def main():
    # Define paths
    base_dir = "/home/long/projects/MedCLIP-SAMv2-finetune"
    
    # Input: SAM masks
    sam_masks_dir = os.path.join(base_dir, "sam_outputs/brain_tumors/test_masks")
    
    # Ground truth masks
    gt_masks_dir = os.path.join(base_dir, "data/brain_tumors/test_masks")
    
    # Checkpoint directory
    checkpoint_dir = os.path.join(base_dir, 
                                 "nnUNet_results/Dataset801_BrainTumors/"
                                 "nnUNetTrainerUMambaBotSmallBatch__nnUNetPlans__2d/fold_0")
    
    # Temporary directory for nnUNet input
    nnunet_input_dir = os.path.join(base_dir, "umamba_inference_input/brain_tumors")
    
    # Output directory for predictions
    nnunet_output_dir = os.path.join(base_dir, "umamba_predictions_raw/brain_tumors")
    
    # Final predictions directory (converted to PNG)
    final_predictions_dir = os.path.join(base_dir, "umamba_predictions/brain_tumors")
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Step 1: Prepare input from SAM masks
    num_inputs = prepare_nnunet_input_from_sam_masks(sam_masks_dir, nnunet_input_dir)
    
    if num_inputs == 0:
        print("ERROR: No input images prepared!")
        return
    
    # Step 2: Run nnUNet prediction
    success = run_nnunet_prediction(
        input_dir=nnunet_input_dir,
        output_dir=nnunet_output_dir,
        checkpoint_dir=checkpoint_dir,
        dataset_id=801,
        configuration='2d',
        fold=0,
        trainer='nnUNetTrainerUMambaBotSmallBatch'
    )
    
    if not success:
        print("ERROR: Prediction failed!")
        return
    
    # Step 3: Convert predictions to PNG format
    num_preds = convert_predictions_to_png(nnunet_output_dir, final_predictions_dir)
    
    if num_preds == 0:
        print("ERROR: No predictions converted!")
        return
    
    # Step 4: Evaluate predictions
    metrics = evaluate_predictions(final_predictions_dir, gt_masks_dir)
    
    if metrics is None:
        print("ERROR: Evaluation failed!")
        return
    
    print("\n✓ Inference and evaluation completed successfully!")


if __name__ == "__main__":
    main()
