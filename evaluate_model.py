import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from scripts.freq_components import SmartFusionBlock, DWTForward
from train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate Dice, IoU, Precision, Recall"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    iou = (intersection + 1e-8) / (pred_binary.sum() + target_binary.sum() - intersection + 1e-8)
    
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    
    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load BiomedCLIP
    print("\nLoading BiomedCLIP...")
    model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    # Initialize components
    print("Loading trained model...")
    dwt = DWTForward().to(device)
    fusion = SmartFusionBlock(hf_channels=777, lf_channels=1, out_channels=32).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    fusion.load_state_dict(checkpoint)
    print(f"✓ Loaded checkpoint: {args.checkpoint}")
    
    # Create model
    class Args:
        pass
    model_args = Args()
    model = FrequencyMedCLIPSAMv2(biomedclip, fusion, dwt, model_args).to(device)
    model.eval()
    
    # Load test dataset
    print(f"\nLoading test dataset: {args.dataset}...")
    test_dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward
            preds = model(pixel_values, input_ids).squeeze(1)
            
            # Calculate metrics for each sample in batch
            for i in range(preds.shape[0]):
                metrics = calculate_metrics(preds[i], masks[i])
                all_metrics.append(metrics)
    
    # Aggregate results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics])
    }
    
    std_metrics = {
        'dice': np.std([m['dice'] for m in all_metrics]),
        'iou': np.std([m['iou'] for m in all_metrics]),
        'precision': np.std([m['precision'] for m in all_metrics]),
        'recall': np.std([m['recall'] for m in all_metrics])
    }
    
    print(f"\nDataset: {args.dataset}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"\nMetrics (Mean ± Std):")
    print(f"  Dice Score:  {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    print(f"  IoU:         {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    print(f"  Precision:   {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"  Recall:      {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    
    # Save results
    results_file = f"results_{args.dataset}_epoch{args.checkpoint.split('epoch')[-1].split('.')[0]}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write(f"Dice Score:  {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}\n")
        f.write(f"IoU:         {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}\n")
        f.write(f"Precision:   {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}\n")
        f.write(f"Recall:      {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    print("="*60)

if __name__ == '__main__':
    main()
