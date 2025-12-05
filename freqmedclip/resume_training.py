"""
Resume training from existing checkpoints for both datasets.
Continues training for an additional 100 epochs.
"""

import os
import sys
import subprocess
from datetime import datetime
import glob

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not pth_files:
        return None
    return max(pth_files, key=os.path.getctime)

def get_epoch_from_checkpoint(ckpt_path):
    """Extract epoch number from checkpoint filename"""
    import re
    match = re.search(r'epoch(\d+)', os.path.basename(ckpt_path))
    if match:
        return int(match.group(1))
    return 0

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Previous training results in checkpoints folder
    prev_results_dir = "../checkpoints/results_20251205_011222"
    
    # New results directory in checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_base = os.path.join(os.path.dirname(script_dir), "checkpoints")
    new_results_dir = os.path.join(checkpoints_base, f"results_resume_{timestamp}")
    os.makedirs(new_results_dir, exist_ok=True)
    
    datasets = [
        {
            "name": "brain_tumors",
            "prev_ckpt_dir": os.path.join(prev_results_dir, "brain_tumors_checkpoints"),
            "new_ckpt_dir": os.path.join(new_results_dir, "brain_tumors_checkpoints_resume")
        },
        {
            "name": "breast_tumors",
            "prev_ckpt_dir": os.path.join(prev_results_dir, "breast_tumors_checkpoints"),
            "new_ckpt_dir": os.path.join(new_results_dir, "breast_tumors_checkpoints_resume")
        }
    ]
    
    print(f"\n{'#'*70}")
    print(f"# RESUME TRAINING - {timestamp}")
    print(f"# New Results: {new_results_dir}")
    print(f"{'#'*70}\n")
    
    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        prev_ckpt_dir = dataset_config["prev_ckpt_dir"]
        new_ckpt_dir = dataset_config["new_ckpt_dir"]
        
        os.makedirs(new_ckpt_dir, exist_ok=True)
        
        print(f"\n{'*'*70}")
        print(f"* Dataset: {dataset_name.upper()}")
        print(f"{'*'*70}")
        
        # Find latest checkpoint
        latest_ckpt = find_latest_checkpoint(prev_ckpt_dir)
        
        if latest_ckpt is None:
            print(f"✗ No checkpoint found in {prev_ckpt_dir}")
            print(f"  Starting fresh training instead...")
            ckpt_arg = ""
            start_epoch = 0
        else:
            start_epoch = get_epoch_from_checkpoint(latest_ckpt)
            ckpt_rel = os.path.relpath(latest_ckpt, script_dir)
            ckpt_arg = f"--resume {ckpt_rel}"
            print(f"✓ Found checkpoint: {os.path.basename(latest_ckpt)}")
            print(f"  Starting from epoch {start_epoch + 1}")
        
        # Calculate total epochs
        additional_epochs = 100
        total_epochs = start_epoch + additional_epochs
        
        print(f"  Training for {additional_epochs} more epochs (total: {total_epochs})")
        
        # Build command
        new_ckpt_rel = os.path.relpath(new_ckpt_dir, script_dir)
        cmd = (f"python .\\train_freq_fusion.py "
               f"--dataset {dataset_name} "
               f"--epochs {total_epochs} "
               f"--batch-size 4 "
               f"--lr 1e-4 "
               f"--save-dir {new_ckpt_rel} "
               f"{ckpt_arg}")
        
        print(f"\n{'='*70}")
        print(f"Command: {cmd}")
        print(f"{'='*70}\n")
        
        # Run training
        result = subprocess.run(cmd, shell=True, cwd=script_dir)
        
        if result.returncode == 0:
            print(f"\n✓ Training completed for {dataset_name}")
            
            # Run evaluation
            final_ckpt = find_latest_checkpoint(new_ckpt_dir)
            if final_ckpt:
                print(f"\nRunning evaluation...")
                final_ckpt_rel = os.path.relpath(final_ckpt, script_dir)
                eval_cmd = (f"python .\\evaluate_freqmedclip.py "
                           f"--dataset {dataset_name} "
                           f"--checkpoint {final_ckpt_rel} "
                           f"--batch-size 4")
                
                eval_result = subprocess.run(eval_cmd, shell=True, cwd=script_dir)
                
                if eval_result.returncode == 0:
                    print(f"✓ Evaluation completed")
                else:
                    print(f"✗ Evaluation failed")
        else:
            print(f"\n✗ Training failed for {dataset_name}")
    
    print(f"\n{'#'*70}")
    print(f"# RESUME TRAINING COMPLETE")
    print(f"# Results: {new_results_dir}")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    main()
