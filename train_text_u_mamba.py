"""
Training Script for Text-U-Mamba (Weakly Supervised Segmentation)
Replaces nnU-Net with a linear-complexity Mamba-based architecture.
"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
import cv2
import numpy as np

# Import Text-U-Mamba Model
from text_u_mamba import TextUMamba
from saliency_maps.text_prompts import *

# --- 1. Dataset Class ---
class TextUMambaDataset(Dataset):
    def __init__(self, root_dir, dataset_name, processor, tokenizer, split='train', max_length=77):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        
        self.img_dir = os.path.join(root_dir, dataset_name, f"{split}_images")
        # Assuming we use the same mask directory (ground truth or pseudo-labels)
        self.mask_dir = os.path.join(root_dir, dataset_name, f"{split}_masks")
        
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Load prompts
        self.prompts = []
        if 'breast' in dataset_name:
            self.prompts = breast_tumor_P2_prompts + benign_breast_tumor_P3_prompts + malignant_breast_tumor_P3_prompts
        elif 'lung' in dataset_name:
            self.prompts = lung_CT_P2_prompts + lung_xray_P2_prompts + covid_lung_P3_prompts + viral_pneumonia_lung_P3_prompts + lung_opacity_P3_prompts
        elif 'brain' in dataset_name:
            self.prompts = brain_tumor_P2_prompts + glioma_brain_tumor_P3_prompts + meningioma_brain_tumor_P3_prompts + pituitary_brain_tumor_P3_prompts
        else:
            self.prompts = ["A medical image showing an abnormality."]
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load Image
        image = Image.open(img_path).convert('RGB')
        
        # Load Mask
        try:
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            mask = (mask > 127).astype(np.float32)
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            mask = np.zeros(image.size[::-1], dtype=np.float32)

        # Process Image (BiomedCLIP processor resizes to 224x224)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0) # (3, 224, 224)
        
        # Resize mask to 224x224
        mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).long()
        
        # Text Prompt
        text_prompt = random.choice(self.prompts)
        text_inputs = self.tokenizer(text_prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = text_inputs['input_ids'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'mask': mask_tensor,
            'img_name': img_name
        }

# --- 2. Loss Function ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

# --- 3. Training Loop ---
def train_single_dataset(args, dataset_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training Text-U-Mamba on: {dataset_name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load Processor/Tokenizer (BiomedCLIP)
    model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Initialize Model
    print("Initializing Text-U-Mamba Model...")
    model = TextUMamba(num_classes=1).to(device)
    
    # Dataset
    print(f"Loading Dataset: {dataset_name}...")
    train_dataset = TextUMambaDataset(args.data_root, dataset_name, processor, tokenizer, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Training samples: {len(train_dataset)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Loss
    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Save Dir
    save_dir = os.path.join(args.save_dir, f'text_u_mamba_{dataset_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Loop
    print("\nStarting Training...\n")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device).float()
            
            # Forward
            # Note: TextUMamba forward takes (x, input_ids)
            preds = model(pixel_values, input_ids) 
            # Output is [B, 1, H, W] or list if deep supervision
            if isinstance(preds, list):
                preds = preds[0]
            preds = preds.squeeze(1)
            
            # Loss
            loss_d = dice_loss(preds, masks)
            loss_b = bce_loss(preds, masks)
            loss = loss_d + loss_b
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if args.dry_run:
                print("Dry run completed successfully.")
                return

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(save_dir, f"text_u_mamba_{dataset_name}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")

    print(f"\n✅ Training completed for {dataset_name}!\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['breast_tumors'],
                       help='List of datasets to train on')
    parser.add_argument('--data-root', type=str, default='data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per dataset')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--dry-run', action='store_true', help='Run single batch for debugging')
    args = parser.parse_args()
    
    for dataset_name in args.datasets:
        train_single_dataset(args, dataset_name)

if __name__ == '__main__':
    main()
