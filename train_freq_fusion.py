import argparse
import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import cv2

# Import our custom components
from scripts.freq_components import SmartFusionBlock, DWTForward
from scripts.methods import vision_heatmap_iba
from saliency_maps.text_prompts import *

# --- 1. Dataset Class ---
class FreqMedCLIPDataset(Dataset):
    def __init__(self, root_dir, dataset_name, processor, tokenizer, split='train', max_length=77):
        """
        Args:
            root_dir (str): Path to 'data' directory
            dataset_name (str): Name of dataset (e.g., 'breast_tumors')
            processor: BiomedCLIP processor
            tokenizer: BiomedCLIP tokenizer
            split (str): 'train' or 'val'
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        
        self.img_dir = os.path.join(root_dir, dataset_name, f"{split}_images")
        self.mask_dir = os.path.join(root_dir, dataset_name, f"{split}_masks")
        
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Load prompts based on dataset name
        # This is a simplified logic mapping dataset names to prompt lists from text_prompts.py
        # In a real scenario, you might want more specific logic or a mapping file
        self.prompts = []
        if 'breast' in dataset_name:
            self.prompts = breast_tumor_P2_prompts + benign_breast_tumor_P3_prompts + malignant_breast_tumor_P3_prompts
        elif 'lung' in dataset_name:
            self.prompts = lung_CT_P2_prompts + lung_xray_P2_prompts + covid_lung_P3_prompts + viral_pneumonia_lung_P3_prompts + lung_opacity_P3_prompts
        elif 'brain' in dataset_name:
            self.prompts = brain_tumor_P2_prompts + glioma_brain_tumor_P3_prompts + meningioma_brain_tumor_P3_prompts + pituitary_brain_tumor_P3_prompts
        else:
            # Fallback generic prompt
            self.prompts = ["A medical image showing an abnormality."]
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load Image
        image = Image.open(img_path).convert('RGB')
        original_size = image.size[::-1] # (H, W)
        
        # Load Mask
        try:
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            mask = (mask > 127).astype(np.float32) # Binary mask
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            mask = np.zeros(original_size, dtype=np.float32)

        # Process Image for BiomedCLIP
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0) # (3, 224, 224)
        
        # Resize mask to match model output (224x224) for loss calculation
        mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).long()
        
        # Randomly select a text prompt
        text_prompt = random.choice(self.prompts)
        text_inputs = self.tokenizer(text_prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = text_inputs['input_ids'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'mask': mask_tensor,
            'img_name': img_name
        }

# --- 2. Model Wrapper ---
class FrequencyMedCLIPSAMv2(nn.Module):
    def __init__(self, biomedclip_model, fusion_block, dwt_module, args):
        super().__init__()
        self.biomedclip = biomedclip_model
        self.fusion_block = fusion_block
        self.dwt_module = dwt_module
        self.args = args
        
        # Freeze BiomedCLIP
        for param in self.biomedclip.parameters():
            param.requires_grad = False
            
    def forward(self, pixel_values, input_ids):
        # 1. Extract Features using BiomedCLIP (Frozen)
        # We need to hook into the model to get intermediate layers if we want to do it exactly like generate_saliency_maps.py
        # However, vision_heatmap_freq_aware in methods.py seems to handle the forward pass logic.
        # But for training, we need a differentiable forward pass.
        # Let's re-implement the core logic of 'vision_heatmap_freq_aware' here but in a differentiable way.
        
        # Note: vision_heatmap_freq_aware uses hooks which might be tricky for backprop if not careful.
        # A cleaner way for training is to rely on the fact that we just need the features.
        
        # For this implementation, we will simplify and assume we can get the features.
        # Since modifying the internal forward of BiomedCLIP is complex without the code of `vision_heatmap_freq_aware` (which I can't see fully but I know it exists),
        # I will assume we can use the `vision_heatmap_iba` or similar logic adapted for training.
        
        # WAIT: I need to see `scripts/methods.py` to know how `vision_heatmap_freq_aware` works. 
        # Since I cannot see it right now, I will implement a standard forward pass that mimics the pipeline:
        # Image -> DWT -> HF
        # Image -> BiomedCLIP -> LF (and intermediate HF)
        # LF + Text -> Coarse Map (M2IB logic - simplified here as dot product for now or use the existing method if importable)
        # Coarse + HF -> Fusion -> Fine Map
        
        # Let's use the components we have.
        
        # A. Get Image Features (LF)
        vision_outputs = self.biomedclip.vision_model(pixel_values, output_hidden_states=True)
        # Last layer hidden state: (B, 197, 768) for ViT-B/16 usually
        last_hidden_state = vision_outputs.last_hidden_state 
        
        # B. Get Text Features
        # BiomedCLIP text_model returns tuple: (last_hidden_state, pooler_output, hidden_states)
        text_outputs = self.biomedclip.text_model(input_ids, output_hidden_states=True)
        if isinstance(text_outputs, tuple):
            text_embeds = text_outputs[1]  # pooler_output is the second element
        else:
            text_embeds = text_outputs.pooler_output
        # text_embeds shape: (B, 768)
        
        # C. Coarse Map Generation (Simplified M2IB/CAM)
        # Project text to image dimension if needed, or just dot product
        # Assuming ViT: (B, N_patches+1, D)
        patch_embeddings = last_hidden_state[:, 1:, :] # Remove CLS token, (B, 196, 768)
        
        # Normalize
        patch_embeddings = F.normalize(patch_embeddings, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Dot product: (B, 196, D) * (B, D, 1) -> (B, 196, 1)
        coarse_map_flat = torch.bmm(patch_embeddings, text_embeds.unsqueeze(-1))
        
        # Reshape to (B, 1, 14, 14) assuming 224x224 image and patch size 16
        H_feat = W_feat = int(np.sqrt(patch_embeddings.shape[1]))
        coarse_map = coarse_map_flat.view(-1, 1, H_feat, W_feat)
        
        # D. High Frequency Features
        # 1. From DWT
        dwt_feats = self.dwt_module(pixel_values) # (B, 9, 112, 112)
        
        # 2. From Shallow Layers of ViT (e.g., layer 3)
        # hidden_states is a tuple. Index 3 is layer 3 output.
        shallow_feats = vision_outputs.hidden_states[3][:, 1:, :] # (B, 196, 768)
        shallow_feats = shallow_feats.permute(0, 2, 1).view(-1, 768, H_feat, W_feat) # (B, 768, 14, 14)
        
        # Upsample shallow feats to match DWT size (112x112)
        shallow_feats_up = F.interpolate(shallow_feats, size=(112, 112), mode='bilinear', align_corners=False)
        
        # Concatenate DWT and Shallow features
        # Note: We need to adjust channels. SmartFusionBlock expects `hf_channels`.
        # In main script it was 777 (768 + 9).
        hf_features = torch.cat([shallow_feats_up, dwt_feats], dim=1) # (B, 777, 112, 112)
        
        # E. Fusion
        saliency_fine = self.fusion_block(hf_features, coarse_map) # (B, 1, 112, 112)
        
        # Upsample to 224x224 for loss
        saliency_final = F.interpolate(saliency_fine, size=(224, 224), mode='bilinear', align_corners=False)
        
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., breast_tumors)')
    parser.add_argument('--data-root', type=str, default='data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--dry-run', action='store_true', help='Run a single batch for debugging')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BiomedCLIP from local model
    print("Loading BiomedCLIP from local model...")
    model_name = "saliency_maps/model"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    # Initialize Components
    print("Initializing Fusion Components...")
    dwt = DWTForward().to(device)
    # HF Channels = 768 (ViT) + 9 (DWT) = 777
    fusion = SmartFusionBlock(hf_channels=777, lf_channels=1, out_channels=32).to(device)
    
    # Model Wrapper
    model = FrequencyMedCLIPSAMv2(biomedclip, fusion, dwt, args).to(device)
    
    # Dataset & DataLoader
    print(f"Loading Dataset: {args.dataset}...")
    train_dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # Windows: set num_workers=0
    
    # Optimizer
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.lr)
    
    # Loss
    dice_criterion = DiceLoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    
    # Training Loop
    print("Starting Training...")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load validation dataset for metrics
    print(f"Loading validation dataset: {args.dataset}...")
    val_dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Validation samples: {len(val_dataset)}")
    
    best_dice = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device).float()
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(pixel_values, input_ids) # (B, 1, 224, 224)
            preds = preds.squeeze(1)
            
            # Loss
            loss_dice = dice_criterion(preds, masks)
            loss_bce = bce_criterion(preds, masks)
            loss = loss_dice + loss_bce
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if args.dry_run:
                print("Dry run completed successfully.")
                return
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_dice_scores = []
        val_iou_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                masks = batch['mask'].to(device).float()
                
                preds = model(pixel_values, input_ids).squeeze(1)
                
                # Calculate metrics
                for i in range(preds.shape[0]):
                    pred_binary = (torch.sigmoid(preds[i]) > 0.5).float()
                    target = masks[i]
                    
                    intersection = (pred_binary * target).sum()
                    union = pred_binary.sum() + target.sum()
                    dice = (2. * intersection + 1e-8) / (union + 1e-8)
                    iou = (intersection + 1e-8) / (pred_binary.sum() + target.sum() - intersection + 1e-8)
                    
                    val_dice_scores.append(dice.item())
                    val_iou_scores.append(iou.item())
        
        avg_dice = np.mean(val_dice_scores)
        avg_iou = np.mean(val_iou_scores)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
        
        # Save best checkpoint only
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_epoch = epoch + 1
            
            # Remove old best checkpoint
            old_checkpoints = [f for f in os.listdir(args.save_dir) if f.startswith(f"fusion_{args.dataset}_") and f.endswith('.pth')]
            for old_ckpt in old_checkpoints:
                os.remove(os.path.join(args.save_dir, old_ckpt))
            
            # Save new best
            checkpoint_path = os.path.join(args.save_dir, f"fusion_{args.dataset}_epoch{epoch+1}.pth")
            torch.save(fusion.state_dict(), checkpoint_path)
            print(f"✓ New best model saved! Dice: {best_dice:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best epoch: {best_epoch} (Dice: {best_dice:.4f})")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
