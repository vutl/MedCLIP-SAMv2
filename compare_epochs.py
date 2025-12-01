import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
import numpy as np
import cv2
import random
import sys
sys.path.insert(0, 'd:/Documents/LMIS/MedCLIP-SAMv2')

from scripts.freq_components import SmartFusionBlock, DWTForward
from saliency_maps.text_prompts import breast_tumor_P2_prompts

device = torch.device('cuda')

# Load BiomedCLIP
print("Loading BiomedCLIP...")
model_name = "chuhac/BiomedCLIP-vit-bert-hf"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

# Initialize components
dwt = DWTForward().to(device)

# Load sample
img_path = "data/breast_tumors/train_images/benign (1).png"
mask_path = "data/breast_tumors/train_masks/benign (1).png"

image = Image.open(img_path).convert('RGB')
mask = Image.open(mask_path).convert('L')
mask = np.array(mask)
mask = (mask > 127).astype(np.float32)
mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).to(device)

inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs['pixel_values'].to(device)

text_prompt = random.choice(breast_tumor_P2_prompts)
text_inputs = tokenizer(text_prompt, padding='max_length', truncation=True, max_length=77, return_tensors="pt")
input_ids = text_inputs['input_ids'].to(device)

print(f"Sample: {img_path}")
print(f"Text: {text_prompt[:80]}...")

# Forward pass function
def forward_pass(fusion_module):
    with torch.no_grad():
        # Vision
        vision_outputs = biomedclip.vision_model(pixel_values, output_hidden_states=True)
        last_hidden_state = vision_outputs.last_hidden_state
        
        # Text
        text_outputs = biomedclip.text_model(input_ids, output_hidden_states=True)
        text_embeds = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs.pooler_output
        
        # Coarse map
        patch_embeddings = last_hidden_state[:, 1:, :]
        patch_embeddings = F.normalize(patch_embeddings, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        coarse_map_flat = torch.bmm(patch_embeddings, text_embeds.unsqueeze(-1))
        H_feat = W_feat = int(np.sqrt(patch_embeddings.shape[1]))
        coarse_map = coarse_map_flat.view(-1, 1, H_feat, W_feat)
        
        # HF features
        dwt_feats = dwt(pixel_values)
        shallow_feats = vision_outputs.hidden_states[3][:, 1:, :]
        shallow_feats = shallow_feats.permute(0, 2, 1).view(-1, 768, H_feat, W_feat)
        shallow_feats_up = F.interpolate(shallow_feats, size=(112, 112), mode='bilinear', align_corners=False)
        hf_features = torch.cat([shallow_feats_up, dwt_feats], dim=1)
        
        # Fusion
        saliency_fine = fusion_module(hf_features, coarse_map)
        saliency_final = F.interpolate(saliency_fine, size=(224, 224), mode='bilinear', align_corners=False)
        
        return saliency_final.squeeze(1)

# Test with epoch 1 and epoch 10
print("\n" + "="*60)
print("Comparing Epoch 1 vs Epoch 10")
print("="*60)

for epoch in [1, 10]:
    fusion = SmartFusionBlock(hf_channels=777, lf_channels=1, out_channels=32).to(device)
    checkpoint = torch.load(f"checkpoints/fusion_breast_tumors_epoch{epoch}.pth", map_location=device)
    fusion.load_state_dict(checkpoint)
    fusion.eval()
    
    pred = forward_pass(fusion)
    
    # Calculate loss
    dice_loss = 1 - (2 * (torch.sigmoid(pred) * mask_tensor).sum() + 1) / (torch.sigmoid(pred).sum() + mask_tensor.sum() + 1)
    bce_loss = F.binary_cross_entropy_with_logits(pred, mask_tensor)
    total_loss = dice_loss + bce_loss
    
    print(f"\nEpoch {epoch}:")
    print(f"  Dice Loss: {dice_loss.item():.4f}")
    print(f"  BCE Loss:  {bce_loss.item():.4f}")
    print(f"  Total:     {total_loss.item():.4f}")
    print(f"  Pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
    print(f"  Pred mean:  {pred.mean().item():.3f}")

print("\n" + "="*60)
improvement = (total_loss.item() - dice_loss.item()) / total_loss.item() * 100
print(f"✅ Model đã học! (Nếu loss epoch 10 < epoch 1)")
print("="*60)
