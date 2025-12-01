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


def vision_heatmap_freq_aware(text_t, image_t, model, layer_idx, beta, var, fusion_block, dwt_module=None, lr=1, train_steps=10, ensemble=False, progbar=True):
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
        beta: IBA beta parameter
        var: IBA variance parameter
        fusion_block: SmartFusionBlock module
        dwt_module: Pre-initialized DWTForward module (for efficiency)
        lr: IBA learning rate
        train_steps: IBA training steps
        ensemble: Whether to use ensemble
        progbar: Show progress bar
        
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
    # Remove CLS token (index 0) and reshape to spatial grid
    b, seq_len, dim = f_early.shape
    grid_len = seq_len - 1  # Subtract CLS token
    grid_size = int(grid_len ** 0.5)
    
    # Validate grid is square
    if grid_size * grid_size != grid_len:
        raise ValueError(f"Non-square grid: seq_len={seq_len}, grid_len={grid_len}, grid_size={grid_size}")
    
    # Reshape: (B, seq_len-1, dim) → (B, dim, grid_size, grid_size)
    f_early_spatial = f_early[:, 1:, :].permute(0, 2, 1).reshape(b, dim, grid_size, grid_size)
    
    # 2c. Upsample early features to match wavelet resolution
    # This enables pixel-level fusion
    target_size = i_hf.shape[-2:]  # Match wavelet output size
    f_early_up = F.interpolate(f_early_spatial, size=target_size, mode='bilinear', align_corners=False)
    
    # 2d. Combine wavelet (texture/edges) with early ViT features (mid-level semantics)
    # Per Pipeline.md: "Injection (Tiêm tần số): ... cộng vào feature này"
    f_hf = torch.cat([i_hf, f_early_up], dim=1)  # (B, 9+dim, H/2, W/2) - e.g., (1, 777, 112, 112)
    
    # ============================================================================
    # STEP 3: GENERATE COARSE MAP using M2IB on DEEP FEATURES
    # ============================================================================
    # Use pre-extracted f_deep features to avoid redundant forward pass
    # Note: IBA framework still needs model access for optimization, but uses our features as starting point
    layer = extract_bert_layer(model.vision_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, f_deep)
    
    
    reader = IBAInterpreter(
        model, 
        compression_estimator, 
        beta=beta, 
        lr=lr, 
        steps=train_steps, 
        progbar=progbar, 
        ensemble=ensemble
    )
    
    # ARCHITECTURAL NOTE: IBA's vision_heatmap will perform additional forward passes
    # during its optimization loop. This is inherent to the IBA algorithm which iteratively
    # optimizes a bottleneck. Our pre-extracted features (f_deep) are used to initialize
    # the compression estimator, but IBA still needs to run the model during optimization.
    # 
    # TRUE Single-Stream would require modifying IBA's internals or using a non-iterative
    # saliency method. For now, we accept this as a limitation.
    # 
    # Future improvement: Replace M2IB with a feed-forward saliency method that can use
    # pre-extracted features directly (e.g., simple attention pooling or learned projection).
    s_coarse_np = reader.vision_heatmap(text_t, image_t)
    
    # Convert to tensor for fusion
    s_coarse = torch.tensor(
        s_coarse_np, 
        device=image_t.device, 
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
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
    
    # Normalize to [0, 1] range
    s_min, s_max = s_fine_np.min(), s_fine_np.max()
    if s_max - s_min > 1e-8:
        s_fine_np = (s_fine_np - s_min) / (s_max - s_min)
    else:
        s_fine_np = np.zeros_like(s_fine_np)  # Handle edge case of uniform map
    
    return s_fine_np



