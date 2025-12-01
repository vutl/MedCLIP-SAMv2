"""
TGCAM-based Training Script for MedCLIP-SAMv2
Trains the TGCAM fusion modules (GatedITEM + SharpenedTGCAM) on medical imaging datasets.
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
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import cv2
import numpy as np

# Import TGCAM components
import sys
sys.path.insert(0, 'd:/Documents/LMIS/MedCLIP-SAMv2')
from saliency_maps.scripts.tgcam_components import TGCAMPipeline
from saliency_maps.text_prompts import *


# --- 1. Dataset Class ---
class TGCAMDataset(Dataset):
    def __init__(self, root_dir, dataset_name, processor, tokenizer, split='train', max_length=77):
        """
        Args:
            root_dir (str): Path to 'data' directory
            dataset_name (str): Name of dataset (e.g., 'breast_tumors')
            processor: BiomedCLIP processor
            tokenizer: BiomedCLIP tokenizer
            split (str): 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        
        # Construct paths
        self.img_dir = os.path.join(root_dir, dataset_name, f'{split}_images')
        self.mask_dir = os.path.join(root_dir, dataset_name, f'{split}_masks')
        
        # Get image files
        self.image_files = sorted([f for f in os.listdir(self.img_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Select prompts based on dataset
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
        original_size = image.size[::-1]  # (H, W)
        
        # Load Mask
        try:
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            mask = (mask > 127).astype(np.float32)
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            mask = np.zeros(original_size, dtype=np.float32)
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Resize mask to 224x224
        mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).long()
        
        # Randomly select text prompt
        text_prompt = random.choice(self.prompts)
        text_inputs = self.tokenizer(text_prompt, padding='max_length', truncation=True, 
                                     max_length=self.max_length, return_tensors="pt")
        input_ids = text_inputs['input_ids'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'mask': mask_tensor,
            'img_name': img_name
        }


# --- 2. Model Wrapper ---
class TGCAMMedCLIPSAMv2(nn.Module):
    def __init__(self, biomedclip_model, tgcam_pipeline, args):
        super().__init__()
        self.biomedclip = biomedclip_model
        self.tgcam = tgcam_pipeline
        self.args = args
        
        # Freeze BiomedCLIP
        for param in self.biomedclip.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values, input_ids, layer_idx=7):
        """
        Args:
            pixel_values: [B, 3, 224, 224]
            input_ids: [B, 77]
            layer_idx: Layer to extract features from (default: 7)
        Returns:
            saliency_final: [B, 1, 224, 224]
        """
        # A. Extract Vision Features
        vision_outputs = self.biomedclip.vision_model(pixel_values, output_hidden_states=True)
        visual_features_raw = vision_outputs.hidden_states[layer_idx + 1]  # [B, 197, 768]
        
        # B. Extract Text Features
        text_outputs = self.biomedclip.text_model(input_ids, output_hidden_states=True)
        text_features_raw = text_outputs[0] if isinstance(text_outputs, tuple) else text_outputs.last_hidden_state
        # text_features_raw: [B, 77, 768]
        
        # C. Remove CLS token from visual features
        B, V_seq, V_dim = visual_features_raw.shape
        spatial_root = int(V_seq ** 0.5)
        if spatial_root * spatial_root == V_seq:
            visual_features = visual_features_raw
        else:
            visual_features = visual_features_raw[:, 1:, :]  # Remove CLS token
        
        # D. TGCAM Forward Pass
        saliency_map, fused_features = self.tgcam(visual_features, text_features_raw)
        # saliency_map: [B, 1, 14, 14]
        
        # E. Upsample to 224x224
        saliency_final = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        
        return saliency_final


# --- 3. Loss Function ---
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


# --- 4. Main Training Loop ---
def train_single_dataset(args, dataset_name):
    """Train TGCAM on a single dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training on Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load BiomedCLIP
    print("Loading BiomedCLIP...")
    model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    # Initialize TGCAM
    print("Initializing TGCAM Pipeline...")
    tgcam = TGCAMPipeline(
        visual_dim=768,
        text_dim=768,
        mid_channels=args.mid_channels,
        num_item_iterations=args.num_item_iterations
    ).to(device)
    
    # Create model wrapper
    model = TGCAMMedCLIPSAMv2(biomedclip, tgcam, args).to(device)
    
    # Load dataset
    print(f"Loading Dataset: {dataset_name}...")
    train_dataset = TGCAMDataset(args.data_root, dataset_name, processor, tokenizer, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Training samples: {len(train_dataset)}")
    
    # Optimizer (only TGCAM parameters)
    optimizer = torch.optim.AdamW(tgcam.parameters(), lr=args.lr)
    
    # Loss functions
    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Create checkpoint directory
    save_dir = os.path.join(args.save_dir, f'tgcam_{dataset_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
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
            preds = model(pixel_values, input_ids).squeeze(1)
            
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
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(save_dir, f"tgcam_{dataset_name}_epoch{epoch+1}.pth")
            torch.save(tgcam.state_dict(), checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    print(f"\n✅ Training completed for {dataset_name}!\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['breast_tumors', 'brain_tumors', 'lung_CT', 'lung_Xray'],
                       help='List of datasets to train on')
    parser.add_argument('--data-root', type=str, default='data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs per dataset')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mid-channels', type=int, default=512, help='TGCAM mid channels')
    parser.add_argument('--num-item-iterations', type=int, default=2, help='Number of ITEM iterations')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--dry-run', action='store_true', help='Run single batch for debugging')
    args = parser.parse_args()
    
    # Train on each dataset sequentially
    for dataset_name in args.datasets:
        train_single_dataset(args, dataset_name)
    
    print("\n" + "="*60)
    print("🎉 ALL DATASETS TRAINING COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    main()
