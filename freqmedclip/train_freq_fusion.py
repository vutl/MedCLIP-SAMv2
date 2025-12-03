import argparse
import os
import sys
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

# Try importing albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not found. Data augmentation will be disabled.")

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom components
from freqmedclip.scripts.freq_components import SmartDecoderBlock, DWTForward, FrequencyEncoder, BottleneckFusion, FPNAdapter
from scripts.methods import vision_heatmap_iba
from saliency_maps.text_prompts import *
from loss.hnl import HardNegativeLoss

# --- 0. Data Augmentation ---
def get_transforms(split='train'):
    if not ALBUMENTATIONS_AVAILABLE:
        return None
        
    if split == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2()
        ])

# --- 1. Dataset Class ---
class FreqMedCLIPDataset(Dataset):
    def __init__(self, root_dir, dataset_name, processor, tokenizer, split='train', max_length=77):
        """
        Args:
            root_dir (str): Path to 'data' directory
            dataset_name (str): Name of dataset (e.g., 'breast_tumors')
            processor: BiomedCLIP processor (used if albumentations not available or for text)
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
        
        self.transforms = get_transforms(split)
        
        # Load prompts based on dataset name
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
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Mask not found")
            mask = (mask > 127).astype(np.float32) # Binary mask
        except Exception as e:
            # print(f"Error loading mask {mask_path}: {e}")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            pixel_values = augmented['image']
            mask_tensor = augmented['mask'].long()
        else:
            # Fallback to processor if albumentations missing
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
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
    def __init__(self, biomedclip_model, dwt_module, freq_encoder, fpn_adapter, bottleneck_fusion, args):
        super().__init__()
        self.biomedclip = biomedclip_model
        self.dwt_module = dwt_module
        self.freq_encoder = freq_encoder
        self.fpn_adapter = fpn_adapter
        self.bottleneck_fusion = bottleneck_fusion
        
        # --- Freeze BiomedCLIP ---
        for param in self.biomedclip.parameters():
            param.requires_grad = False
        
        # Unfreeze specific layers for multi-scale adaptation
        layers_to_unfreeze = [3, 6, 9, 11] # 0-indexed
        for i in layers_to_unfreeze:
            for param in self.biomedclip.vision_model.encoder.layers[i].parameters():
                param.requires_grad = True
        self.biomedclip.vision_model.post_layernorm.requires_grad_(True)
        
        # --- Progressive Decoder with Skip Connections ---
        # FPN Output Scales:
        # s1: 14x14 (768) - Bottleneck
        # s2: 28x28 (384) - Skip 1
        # s3: 56x56 (192) - Skip 2
        # s4: 112x112 (96) - Skip 3 (Optional, usually not used in U-Net bottleneck-up flow directly unless deep supervision)
        
        # Frequency Encoder Output Scales:
        # f3: 14x14 (256) - Bottleneck Fusion
        # f2: 28x28 (128) - Skip Fusion 1
        # f1: 56x56 (64)  - Skip Fusion 2
        
        # Stage 1: 14 -> 28
        # Input: Fused Bottleneck (768)
        # Skip: FPN s2 (384)
        # Freq Skip: f2 (128)
        # Total In: 768 + 384 + 128 = 1280 -> Out: 384
        self.dec1 = SmartDecoderBlock(in_channels=768, out_channels=384, skip_channels=384, freq_channels=128, use_lffi=True)
        
        # Stage 2: 28 -> 56
        # Input: 384
        # Skip: FPN s3 (192)
        # Freq Skip: f1 (64)
        # Total In: 384 + 192 + 64 = 640 -> Out: 192
        self.dec2 = SmartDecoderBlock(in_channels=384, out_channels=192, skip_channels=192, freq_channels=64, use_lffi=True)
        
        # Stage 3: 56 -> 112
        # Input: 192
        # Skip: FPN s4 (96)
        # Freq Skip: None (or raw DWT?)
        # Let's use FPN s4.
        # Total In: 192 + 96 = 288 -> Out: 96
        self.dec3 = SmartDecoderBlock(in_channels=192, out_channels=96, skip_channels=96, freq_channels=0, use_lffi=True)
        
        # --- Final Segmentation Head ---
        # 112 -> 224
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.seg_head = nn.Conv2d(96, 1, kernel_size=1)
        
    def forward(self, pixel_values, input_ids):
        # 1. Image Encoder (ViT)
        vision_outputs = self.biomedclip.vision_model(pixel_values, output_hidden_states=True)
        hidden_states = vision_outputs.hidden_states
        
        # Extract features (B, 197, 768)
        # We need intermediate layers for FPNAdapter: [3, 6, 9, 12]
        # hidden_states is a tuple of (embeddings, layer_1, ..., layer_12)
        # So index 0 is embeddings, index 1 is layer 1 output.
        # We want output of layer 3, 6, 9, 12.
        # Indices in hidden_states: 3, 6, 9, 12 (if 1-based from embeddings?)
        # Let's check: len(hidden_states) = 13 (embeddings + 12 layers)
        # hidden_states[0] = embeddings
        # hidden_states[1] = layer 1 output
        # ...
        # hidden_states[12] = layer 12 output (final)
        
        # We want layers 3, 6, 9, 12
        layers_idx = [12, 9, 6, 3] # Order: s1(14), s2(28), s3(56), s4(112)
        
        fpn_inputs = []
        for idx in layers_idx:
            # (B, 197, 768) -> remove cls token -> (B, 196, 768)
            feat = hidden_states[idx][:, 1:, :] 
            B, N, C = feat.shape
            H = W = int(N**0.5) # 14
            # Reshape to (B, C, 14, 14)
            feat_reshaped = feat.permute(0, 2, 1).view(B, C, H, W)
            fpn_inputs.append(feat_reshaped)
        
        # 2. FPN Adapter (Generate Multi-Scale Features)
        # Input: list of [feat12, feat9, feat6, feat3]
        fpn_feats = self.fpn_adapter(fpn_inputs)
        x_bottleneck = fpn_feats[0] # 14x14
        x_skip1 = fpn_feats[1]      # 28x28
        x_skip2 = fpn_feats[2]      # 56x56
        x_skip3 = fpn_feats[3]      # 112x112
        
        # 3. Text Encoder
        text_outputs = self.biomedclip.text_model(input_ids, output_hidden_states=True)
        if isinstance(text_outputs, tuple):
            text_embeds = text_outputs[0]  # (B, SeqLen, 768)
        else:
            text_embeds = text_outputs.last_hidden_state

        # 4. Parallel Frequency Encoder
        # DWT on original image (B, 3, 224, 224) -> (B, 9, 112, 112)
        # FIX: Denormalize pixel_values before DWT to avoid noise amplification
        # pixel_values are normalized with mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=pixel_values.device).view(1, 3, 1, 1)
        
        denorm_imgs = pixel_values * std + mean
        # Clip to [0, 1] just in case
        denorm_imgs = torch.clamp(denorm_imgs, 0, 1)
        
        dwt_feats = self.dwt_module(denorm_imgs)
        
        # Encode Frequency Features: [f3(14), f2(28), f1(56)]
        freq_feats = self.freq_encoder(dwt_feats, text_embeds=text_embeds)
        x_freq_bottleneck = freq_feats[0] # 14x14
        x_freq_skip1 = freq_feats[1]      # 28x28
        x_freq_skip2 = freq_feats[2]      # 56x56
        
        # 5. Bottleneck Fusion
        # Fuse ViT Bottleneck with Frequency Features
        x_fused = self.bottleneck_fusion(x_bottleneck, x_freq_bottleneck)
            
        # 6. Progressive Decoding with Skips and LFFI
        
        # Stage 1: 14 -> 28
        # Use Fused Features as input
        x = self.dec1(x_fused, skip=x_skip1, freq_skip=x_freq_skip1, text_embeds=text_embeds)
        
        # Stage 2: 28 -> 56
        x = self.dec2(x, skip=x_skip2, freq_skip=x_freq_skip2, text_embeds=text_embeds)
        
        # Stage 3: 56 -> 112
        x = self.dec3(x, skip=x_skip3, freq_skip=None, text_embeds=text_embeds) 
        
        # 7. Final Prediction
        x = self.final_up(x) # 112 -> 224
        logits = self.seg_head(x)
        
        # Prepare features for HNL (Contrastive Loss)
        # Pool bottleneck features: (B, C, 14, 14) -> (B, C)
        img_feats_pooled = F.adaptive_avg_pool2d(x_fused, (1, 1)).squeeze(-1).squeeze(-1)
        # Pool text features: (B, SeqLen, 768) -> (B, 768)
        text_feats_pooled = torch.max(text_embeds, dim=1)[0]
        
        return logits, img_feats_pooled, text_feats_pooled

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
    parser.add_argument('--data-root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32) # Increased batch size
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4) # Base LR for decoder
    parser.add_argument('--backbone-lr', type=float, default=1e-5) # Lower LR for backbone
    parser.add_argument('--save-dir', type=str, default='../checkpoints')
    parser.add_argument('--dry-run', action='store_true', help='Run a single batch for debugging')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BiomedCLIP from local model
    print("Loading BiomedCLIP from local model...")
    model_name = "../saliency_maps/model"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    # Initialize Components
    print("Initializing Fusion Components...")
    # DWT Module
    dwt = DWTForward().to(device)
    
    # FPN Adapter
    fpn_adapter = FPNAdapter(in_channels=768, out_channels=[768, 384, 192, 96]).to(device)
    
    # Frequency Encoder
    # Input: 9 channels (DWT), Output: Multi-scale
    freq_encoder = FrequencyEncoder(in_channels=9, base_channels=64, text_dim=768).to(device)
    
    # Bottleneck Fusion
    bottleneck_fusion = BottleneckFusion(dim=768, freq_dim=256).to(device)
    
    # Model Wrapper
    model = FrequencyMedCLIPSAMv2(biomedclip, dwt, freq_encoder, fpn_adapter, bottleneck_fusion, args).to(device)
    
    # Dataset & DataLoader
    print(f"Loading Dataset: {args.dataset}...")
    train_dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # Windows: set num_workers=0
    
    # Optimizer with Differential Learning Rates
    # Group 1: Backbone (Unfrozen layers) -> Low LR
    # Group 2: Decoder & Fusion (Scratch) -> High LR
    backbone_params = filter(lambda p: p.requires_grad, model.biomedclip.parameters())
    decoder_params = list(model.dec1.parameters()) + \
                     list(model.dec2.parameters()) + \
                     list(model.dec3.parameters()) + \
                     list(model.freq_encoder.parameters()) + \
                     list(model.fpn_adapter.parameters()) + \
                     list(model.bottleneck_fusion.parameters()) + \
                     list(model.final_up.parameters()) + \
                     list(model.seg_head.parameters())
                     
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': decoder_params, 'lr': args.lr}
    ])
    
    # Loss
    dice_criterion = DiceLoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    hnl_criterion = HardNegativeLoss()
    
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
        optimizer.zero_grad() # Initialize gradients
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device).float()
            
            # optimizer.zero_grad() # Handled by gradient accumulation
            
            # Forward
            preds, img_feats, text_feats = model(pixel_values, input_ids) # (B, 1, 224, 224)
            preds = preds.squeeze(1)
            
            # Loss
            loss_dice = dice_criterion(preds, masks)
            loss_bce = bce_criterion(preds, masks)
            loss_hnl = hnl_criterion(img_feats, text_feats, batch_size=pixel_values.shape[0])
            
            # Total Loss (Weighted)
            loss = loss_dice + loss_bce + 0.1 * loss_hnl
            
            # Normalize loss for gradient accumulation
            loss = loss / args.grad_accum_steps
            
            # Backward
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'hnl': loss_hnl.item()})
            
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
                
                preds, _, _ = model(pixel_values, input_ids)
                preds = preds.squeeze(1)
                
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
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[BEST] New best model saved! Dice: {best_dice:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best epoch: {best_epoch} (Dice: {best_dice:.4f})")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
