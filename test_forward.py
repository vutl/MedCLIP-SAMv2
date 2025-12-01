import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
import numpy as np
import cv2
import random

# Import components
import sys
sys.path.insert(0, 'd:/Documents/LMIS/MedCLIP-SAMv2')
from scripts.freq_components import SmartFusionBlock, DWTForward
from saliency_maps.text_prompts import breast_tumor_P2_prompts

print("Testing FreqMedCLIP Forward Pass...")

device = torch.device('cuda')
print(f"Device: {device}")

# Load BiomedCLIP
print("\nLoading BiomedCLIP...")
model_name = "chuhac/BiomedCLIP-vit-bert-hf"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

# Initialize components
print("Initializing components...")
dwt = DWTForward().to(device)
fusion = SmartFusionBlock(hf_channels=777, lf_channels=1, out_channels=32).to(device)

# Load a sample image
print("\nLoading sample...")
img_path = "data/breast_tumors/train_images/benign (1).png"
image = Image.open(img_path).convert('RGB')
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs['pixel_values'].to(device)

# Load text
text_prompt = random.choice(breast_tumor_P2_prompts)
text_inputs = tokenizer(text_prompt, padding='max_length', truncation=True, max_length=77, return_tensors="pt")
input_ids = text_inputs['input_ids'].to(device)

print(f"Image shape: {pixel_values.shape}")
print(f"Text shape: {input_ids.shape}")

# Forward pass
print("\nRunning forward pass...")
try:
    # Vision features
    vision_outputs = biomedclip.vision_model(pixel_values, output_hidden_states=True)
    last_hidden_state = vision_outputs.last_hidden_state
    print(f"Vision output shape: {last_hidden_state.shape}")
    
    # Text features
    text_outputs = biomedclip.text_model(input_ids, output_hidden_states=True)
    if isinstance(text_outputs, tuple):
        text_embeds = text_outputs[1]
    else:
        text_embeds = text_outputs.pooler_output
    print(f"Text embeds shape: {text_embeds.shape}")
    
    # Coarse map
    patch_embeddings = last_hidden_state[:, 1:, :]
    patch_embeddings = F.normalize(patch_embeddings, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    coarse_map_flat = torch.bmm(patch_embeddings, text_embeds.unsqueeze(-1))
    H_feat = W_feat = int(np.sqrt(patch_embeddings.shape[1]))
    coarse_map = coarse_map_flat.view(-1, 1, H_feat, W_feat)
    print(f"Coarse map shape: {coarse_map.shape}")
    
    # HF features
    dwt_feats = dwt(pixel_values)
    shallow_feats = vision_outputs.hidden_states[3][:, 1:, :]
    shallow_feats = shallow_feats.permute(0, 2, 1).view(-1, 768, H_feat, W_feat)
    shallow_feats_up = F.interpolate(shallow_feats, size=(112, 112), mode='bilinear', align_corners=False)
    hf_features = torch.cat([shallow_feats_up, dwt_feats], dim=1)
    print(f"HF features shape: {hf_features.shape}")
    
    # Fusion
    saliency_fine = fusion(hf_features, coarse_map)
    saliency_final = F.interpolate(saliency_fine, size=(224, 224), mode='bilinear', align_corners=False)
    print(f"Final saliency shape: {saliency_final.shape}")
    
    print("\n✅ Forward pass successful!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
