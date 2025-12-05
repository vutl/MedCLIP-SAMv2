"""
Batch training script for multiple datasets with automatic validation and evaluation.
Trains both brain_tumors and breast_tumors sequentially, runs evaluation after each,
and generates a comprehensive summary report with step-by-step visualizations.
"""

import os
import sys
import subprocess
import torch
from datetime import datetime
import glob
import shutil

def get_timestamp():
    """Get timestamp for folder naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_command(cmd, description, cwd=None):
    """Run a shell command and return success status"""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(__file__))
    
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not pth_files:
        return None
    return max(pth_files, key=os.path.getctime)

def copy_results(source_pattern, dest_dir, dataset_name):
    """Copy result files matching pattern to destination"""
    files = glob.glob(source_pattern)
    copied_files = []
    for f in files:
        if dataset_name in f:
            try:
                dest_file = os.path.join(dest_dir, os.path.basename(f))
                shutil.copy(f, dest_file)
                copied_files.append(dest_file)
                print(f"✓ Copied: {dest_file}")
            except Exception as e:
                print(f"! Error copying {f}: {e}")
    return copied_files

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = get_timestamp()
    
    # Create organized directory structure
    base_results_dir = os.path.join(os.path.dirname(script_dir), f"results_{timestamp}")
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Summary file
    summary_file = os.path.join(base_results_dir, "SUMMARY_REPORT.txt")
    
    datasets = [
        {"name": "brain_tumors", "epochs": 100},
        {"name": "breast_tumors", "epochs": 100}
    ]
    
    all_results = []
    
    print(f"\n{'#'*70}")
    print(f"# BATCH TRAINING AND EVALUATION - {timestamp}")
    print(f"# Base Results Directory: {base_results_dir}")
    print(f"{'#'*70}\n")
    
    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        epochs = dataset_config["epochs"]
        
        # Create dataset-specific directories
        dataset_ckpt_dir = os.path.join(base_results_dir, f"{dataset_name}_checkpoints")
        dataset_eval_dir = os.path.join(base_results_dir, f"{dataset_name}_eval", "visualizations")
        dataset_results_dir = os.path.join(base_results_dir, f"{dataset_name}_eval")
        
        os.makedirs(dataset_ckpt_dir, exist_ok=True)
        os.makedirs(dataset_eval_dir, exist_ok=True)
        os.makedirs(dataset_results_dir, exist_ok=True)
        
        print(f"\n{'*'*70}")
        print(f"* Processing: {dataset_name.upper()}")
        print(f"* Checkpoints: {dataset_ckpt_dir}")
        print(f"* Eval Results: {dataset_results_dir}")
        print(f"{'*'*70}\n")
        
        # Step 1: Train
        ckpt_dir_rel = os.path.join("..", f"results_{timestamp}", f"{dataset_name}_checkpoints")
        train_cmd = (f"python .\\train_freq_fusion.py "
                     f"--dataset {dataset_name} "
                     f"--epochs {epochs} "
                     f"--batch-size 4 "
                     f"--lr 1e-4 "
                     f"--save-dir {ckpt_dir_rel}")
        
        train_success = run_command(train_cmd, f"TRAINING: {dataset_name}", script_dir)
        
        if train_success:
            print(f"\n✓ Training completed for {dataset_name}")
            
            # Step 2: Find best checkpoint
            best_ckpt = find_latest_checkpoint(dataset_ckpt_dir)
            
            if best_ckpt:
                print(f"✓ Found checkpoint: {best_ckpt}")
                
                # Step 3: Evaluate
                ckpt_rel = os.path.relpath(best_ckpt, script_dir)
                eval_cmd = (f"python .\\evaluate_freqmedclip.py "
                           f"--dataset {dataset_name} "
                           f"--checkpoint {ckpt_rel} "
                           f"--batch-size 4")
                
                eval_success = run_command(eval_cmd, f"EVALUATING: {dataset_name}", script_dir)
                
                if eval_success:
                    print(f"✓ Evaluation completed for {dataset_name}")
                    
                    # Copy results and visualizations
                    # Copy result txt files
                    results_txt_files = copy_results(
                        os.path.join(script_dir, f"results_{dataset_name}_*.txt"),
                        dataset_results_dir,
                        dataset_name
                    )
                    
                    # Copy visualizations
                    viz_src = os.path.join(script_dir, f"visualizations", f"{dataset_name}_eval")
                    if os.path.exists(viz_src):
                        try:
                            for item in os.listdir(viz_src):
                                src_item = os.path.join(viz_src, item)
                                dst_item = os.path.join(dataset_eval_dir, item)
                                if os.path.isfile(src_item):
                                    shutil.copy(src_item, dst_item)
                            print(f"✓ Copied visualizations to: {dataset_eval_dir}")
                        except Exception as e:
                            print(f"! Error copying visualizations: {e}")
                    
                    # Read and store results
                    if results_txt_files:
                        for results_file in results_txt_files:
                            try:
                                with open(results_file, 'r') as f:
                                    results_content = f.read()
                                all_results.append({
                                    "dataset": dataset_name,
                                    "checkpoint": os.path.basename(best_ckpt),
                                    "results_file": results_file,
                                    "content": results_content
                                })
                                print(f"✓ Results stored: {results_file}")
                            except Exception as e:
                                print(f"! Error reading results: {e}")
                else:
                    print(f"✗ Evaluation failed for {dataset_name}")
            else:
                print(f"✗ No checkpoint found for {dataset_name}")
        else:
            print(f"✗ Training failed for {dataset_name}")
    
    # Step 4: Generate summary report
    print(f"\n{'='*70}")
    print("Generating Summary Report...")
    print(f"{'='*70}\n")
    
    with open(summary_file, 'w') as f:
        f.write(f"BATCH TRAINING AND EVALUATION SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results Base Directory: {base_results_dir}\n")
        f.write(f"\nDirectory Structure:\n")
        f.write(f"  - [dataset]_checkpoints/     : Saved model checkpoints\n")
        f.write(f"  - [dataset]_eval/            : Evaluation metrics and visualizations\n")
        f.write(f"    - visualizations/          : Step-by-step intermediate output visualizations\n")
        f.write(f"\n{'='*70}\n\n")
        
        for result in all_results:
            f.write(f"\n{'─'*70}\n")
            f.write(f"DATASET: {result['dataset'].upper()}\n")
            f.write(f"Checkpoint: {result['checkpoint']}\n")
            f.write(f"Results File: {result['results_file']}\n")
            f.write(f"{'─'*70}\n\n")
            f.write(result['content'])
            f.write(f"\n\n")
        
        # Comparison table
        if all_results:
            f.write(f"\n{'='*70}\n")
            f.write("METRICS COMPARISON TABLE\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"{'Dataset':<20} {'Dice':<15} {'IoU':<15} {'Precision':<15} {'Recall':<15}\n")
            f.write(f"{'-'*80}\n")
            
            for result in all_results:
                # Parse metrics from content
                lines = result['content'].split('\n')
                metrics = {"dice": "N/A", "iou": "N/A", "precision": "N/A", "recall": "N/A"}
                
                for line in lines:
                    if 'Dice Score:' in line:
                        try:
                            dice_val = line.split()[2].split('±')[0]
                            metrics['dice'] = f"{dice_val}"
                        except: pass
                    elif 'IoU:' in line:
                        try:
                            iou_val = line.split()[1].split('±')[0]
                            metrics['iou'] = f"{iou_val}"
                        except: pass
                    elif 'Precision:' in line:
                        try:
                            prec_val = line.split()[1].split('±')[0]
                            metrics['precision'] = f"{prec_val}"
                        except: pass
                    elif 'Recall:' in line:
                        try:
                            rec_val = line.split()[1].split('±')[0]
                            metrics['recall'] = f"{rec_val}"
                        except: pass
                
                f.write(f"{result['dataset']:<20} {metrics['dice']:<15} {metrics['iou']:<15} "
                       f"{metrics['precision']:<15} {metrics['recall']:<15}\n")
        
        f.write(f"\n{'='*70}\n")
        f.write("NOTES:\n")
        f.write(f"- All visualizations include intermediate outputs:\n")
        f.write(f"  * Input image, ground truth mask\n")
        f.write(f"  * ViT branch prediction, Frequency branch prediction\n")
        f.write(f"  * Frequency features, FPN multi-scale features\n")
        f.write(f"  * Final prediction with metrics overlay\n")
        f.write(f"- Results include val and test metrics where applicable\n")
        f.write(f"{'='*70}\n")
        f.write("END OF REPORT\n")
        f.write(f"{'='*70}\n")
    
    print(f"\n✓ Summary report saved to:\n  {summary_file}\n")
    print(f"{'#'*70}")
    print(f"# BATCH PROCESSING COMPLETE")
    print(f"# Results organized in: {base_results_dir}")
    print(f"{'#'*70}\n")
    print(f"Structure:")
    for ds in datasets:
        name = ds["name"]
        print(f"  {name}:")
        print(f"    - Checkpoints: {name}_checkpoints/")
        print(f"    - Eval Results: {name}_eval/")
        print(f"    - Visualizations: {name}_eval/visualizations/")

if __name__ == "__main__":
    main()
