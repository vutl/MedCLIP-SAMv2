"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA
"""

from scripts.iba import IBAInterpreter, Estimator
import numpy as np
# import clip
# import copy
import torch 
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast

# Feature Map is the output of a certain layer given X
def extract_feature_map(model, layer_idx, x):
    with torch.no_grad():
        states = model(x, output_hidden_states=True) 
        feature = states['hidden_states'][layer_idx+1] # +1 because the first output is embedding 
        return feature
    
def extract_text_feature_map(model, layer_idx, x):
    with torch.no_grad():
        states = model(x, output_hidden_states=True) 
        feature = states[0] # +1 because the first output is embedding 
        return feature

# Extract BERT Layer
def extract_bert_layer(model, layer_idx):
    desired_layer = ''
    for _, submodule in model.named_children():
        for n, s in submodule.named_children():
            if n == 'layers' or n == 'resblocks':
                for n2, s2 in s.named_children():
                    if n2 == str(layer_idx):
                        desired_layer = s2
                        return desired_layer

# Get an estimator for the compression term
def get_compression_estimator(var, layer, features):
    estimator = Estimator(layer)
    estimator.M = torch.zeros_like(features)
    estimator.S = var*np.ones(features.shape)
    estimator.N = 1
    estimator.layer = layer
    return estimator

def text_heatmap_iba(text_t, image_t, model, layer_idx, beta, var, lr=1, train_steps=10, progbar=True):
    # features = extract_feature_map(model.text_model, layer_idx, text_t)
    features = extract_text_feature_map(model.text_model,layer_idx, text_t)
    layer = extract_bert_layer(model.text_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)
    reader = IBAInterpreter(model, compression_estimator, beta=beta, lr=lr, steps=train_steps, progbar=progbar)
    return reader.text_heatmap(text_t, image_t)

def vision_heatmap_iba(text_t, image_t, model, layer_idx, beta, var, lr=1, train_steps=10,ensemble=False, progbar=True):
    features = extract_feature_map(model.vision_model, layer_idx, image_t)
    layer = extract_bert_layer(model.vision_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)
    reader = IBAInterpreter(model, compression_estimator, beta=beta, lr=lr, steps=train_steps, progbar=progbar,ensemble=ensemble)
    return reader.vision_heatmap(text_t, image_t)

from scripts.freq_components import DWTForward, SmartFusionBlock
import torch.nn.functional as F


def vision_heatmap_freq_aware(text_t, image_t, model, layer_idx, beta, var, fusion_block, cross_attn, attn_proj, shallow_fusion, dwt_module=None, lr=1, train_steps=10, ensemble=False, progbar=True):
    """
    Generates a frequency-aware saliency map using Smart Single-Stream and Coarse-to-Fine Fusion.
    
    Architecture (per PIPELINE_-FreqMedCLIP-Smart-Single-Stream.md):
    1. SINGLE forward pass through BiomedCLIP to extract features at multiple layers
    2. F_LF (Deep/Semantic): Features from layer_idx (default 7) for M2IB
    3. F_HF (Shallow/Detail): Features from layer 3 + Wavelet injection
    4. Coarse-to-Fine Fusion: M2IB(F_LF) + HF → S_fine
    
    Args:
        text_t: Tokenized text prompt
        image_t: Image tensor (B, C, H, W)
        model: BiomedCLIP model with ClipWrapper
        layer_idx: Layer index for deep features (default 7 for M2IB)
        beta: IBA beta parameter (Unused in Dot Product version)
        var: IBA variance parameter (Unused in Dot Product version)
        fusion_block: SmartFusionBlock module
        cross_attn: Trained MultiheadAttention module
        attn_proj: Trained Linear projection module
        shallow_fusion: Trained ShallowFusionBlock module
        dwt_module: Pre-initialized DWTForward module (for efficiency)
        lr: IBA learning rate (Unused)
        train_steps: IBA training steps (Unused)
        ensemble: Whether to use ensemble (Unused)
        progbar: Show progress bar (Unused)
        
    Returns:
        s_fine_np: Fine-grained saliency map as numpy array (H, W) normalized to [0, 1]
    """
    
    # ============================================================================
    # STEP 1: SINGLE FORWARD PASS - Extract features at ALL needed layers
    # ============================================================================
    # This is the CORE of Smart Single-Stream: ONE encoder run for ALL features
    with torch.no_grad():
        vision_outputs = model.vision_model(image_t, output_hidden_states=True)
        hidden_states = vision_outputs['hidden_states']
        
        # Get Text Embeddings for Cross-Attention
        # text_t is already tokenized input_ids
        text_outputs = model.text_model(text_t, output_hidden_states=True)
        if isinstance(text_outputs, tuple):
            text_embeds = text_outputs[0]  # last_hidden_state (B, SeqLen, 768)
        else:
            text_embeds = text_outputs.last_hidden_state
    
    # Extract features at different depths:
    # - Early layer (index 4 = layer 3 after embeddings) for High-Frequency branch
    # - Deep layer (layer_idx + 1) for Low-Frequency/Semantic branch
    f_early = hidden_states[4]  # (B, seq_len, dim) - Layer 3 features for HF
    f_deep = hidden_states[layer_idx + 1]  # (B, seq_len, dim) - Layer 7+ features for LF/M2IB
    
    # Validate dimensions
    assert f_early.dim() == 3, f"Expected f_early to be 3D (B, seq_len, dim), got shape {f_early.shape}"
    assert f_deep.dim() == 3, f"Expected f_deep to be 3D (B, seq_len, dim), got shape {f_deep.shape}"
    
    # ============================================================================
    # STEP 2: CONSTRUCT HIGH-FREQUENCY FEATURES (F_HF)
    # ============================================================================
    # 2a. Apply Discrete Wavelet Transform to extract high-frequency components from image
    if dwt_module is None:
        dwt_module = DWTForward().to(image_t.device)
    i_hf = dwt_module(image_t)  # (B, C*3, H/2, W/2) - e.g., (1, 9, 112, 112) for 224x224 RGB
    
    # 2b. Convert early ViT features from sequence format to spatial format
    # Re-introduced f_early extraction for ShallowFusionBlock
    b, seq_len, dim = f_early.shape
    grid_len = seq_len - 1  # Subtract CLS token
    grid_size = int(grid_len ** 0.5)
    
    # Reshape: (B, seq_len-1, dim) → (B, dim, grid_size, grid_size)
    f_early_spatial = f_early[:, 1:, :].permute(0, 2, 1).reshape(b, dim, grid_size, grid_size)
    
    # 2d. Fuse DWT and Shallow features using ShallowFusionBlock
    # Per Pipeline.md: "Injection (Tiêm tần số): ... cộng vào feature này"
    f_hf = shallow_fusion(i_hf, f_early_spatial)  # (B, 32, 112, 112)
    
    # ============================================================================
    # STEP 3: GENERATE COARSE MAP using CROSS-ATTENTION (Single-Stream)
    # ============================================================================
    # Replaced IBA/DotProduct with Cross-Attention to match training pipeline
    
    # 1. Prepare Patch Embeddings from f_deep
    # f_deep is (B, 197, 768). Remove CLS token.
    patch_embeddings = f_deep[:, 1:, :] # (B, 196, 768)
    
    # 2. Cross-Attention
    # Query=Visual, Key=Text, Value=Text
    # Use full text sequence for Key/Value to allow spatial localization
    text_seq = text_embeds # (B, SeqLen, 768)
    
    # attn_output: (B, 196, 768)
    attn_output, _ = cross_attn(query=patch_embeddings, key=text_seq, value=text_seq)
    
    # 3. Project to 1 channel: (B, 196, 1)
    coarse_map_flat = attn_proj(attn_output)
    
    # 4. Reshape to spatial map
    H_feat = W_feat = int(np.sqrt(patch_embeddings.shape[1]))
    s_coarse = coarse_map_flat.view(b, 1, H_feat, W_feat) # (B, 1, 14, 14)
    
    # ============================================================================
    # STEP 4: COARSE-TO-FINE FUSION
    # ============================================================================
    # Per Pipeline.md Section 3.2: Frequency Refinement
    # "Gating: Dùng S_coarse để 'khoanh vùng' trên F_HF"
    # "Sharpening: Kích hoạt các pixel có giá trị gradient cao (biên)"
    fusion_block = fusion_block.to(image_t.device)
    fusion_block.eval()  # Ensure eval mode for GroupNorm
    
    with torch.no_grad():
        s_fine = fusion_block(f_hf, s_coarse)  # (1, 1, H_hf, W_hf)
    
    # ============================================================================
    # STEP 5: RESIZE TO ORIGINAL IMAGE RESOLUTION & NORMALIZE
    # ============================================================================
    s_fine_up = F.interpolate(
        s_fine, 
        size=image_t.shape[-2:],  # Original image size
        mode='bilinear', 
        align_corners=False
    )
    
    s_fine_np = s_fine_up.squeeze().cpu().numpy()
    
    # Normalize using 98th percentile (Robust to outliers)
    # Per feedback: "Replace Min-Max normalization ... with Percentile normalization"
    v_min = s_fine_np.min()
    v_p98 = np.percentile(s_fine_np, 98)
    
    if v_p98 - v_min > 1e-8:
        s_fine_np = (s_fine_np - v_min) / (v_p98 - v_min)
    else:
        s_fine_np = np.zeros_like(s_fine_np)
        
    # Clip to [0, 1]
    s_fine_np = np.clip(s_fine_np, 0, 1)
    
    return s_fine_np



