"""
Text-U-Mamba: Text-Guided U-Mamba for Weakly Supervised Medical Image Segmentation
Integrates BiomedCLIP text encoder with U-Mamba for linear-complexity segmentation.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Add U-Mamba to Python path
UMAMBA_PATH = os.path.join(os.path.dirname(__file__), 'U-Mamba', 'umamba')
sys.path.insert(0, UMAMBA_PATH)

# Import U-Mamba components
try:
    from nnunetv2.nets.UMambaEnc_2d import UMambaEnc, ResidualMambaEncoder, UNetResDecoder, MambaLayer
    print("✓ Successfully imported U-Mamba components")
except ImportError as e:
    print(f"✗ Failed to import U-Mamba: {e}")
    print(f"  Make sure U-Mamba is cloned at: {UMAMBA_PATH}")
    raise


class TextGatedMambaBlock(nn.Module):
    """
    Text-Guided Mamba Fusion Block (Bottleneck Injection).
    Implements: Gate = σ(V · T_proj), V_mod = V ⊙ Gate, V_out = Mamba(V_mod)
    """
    def __init__(self, visual_dim, text_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.visual_dim = visual_dim
        
        # 1. Text projection to visual dimension
        self.text_proj = nn.Linear(text_dim, visual_dim)
        
        # 2. Mamba layer for sequence modeling (reuse U-Mamba's MambaLayer)
        # Note: MambaLayer expects [B, C, H, W] and handles flattening internally
        self.mamba = MambaLayer(dim=visual_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
    def forward(self, visual_features, text_embedding):
        """
        Args:
            visual_features: [B, C, H, W] from encoder bottleneck
            text_embedding: [B, D_text] from PubMedBERT [CLS] token
        Returns:
            out: [B, C, H, W] fused features
        """
        B, C, H, W = visual_features.shape
        
        # Step 1: Project text to visual dimension [B, C]
        t_proj = self.text_proj(text_embedding)
        
        # Step 2: Generate gate (element-wise modulation)
        # Reshape t_proj to [B, C, 1, 1] for broadcasting
        t_proj_spatial = t_proj.unsqueeze(-1).unsqueeze(-1)
        
        # Gate = σ(V · T)
        # Use dot product along channel dimension, then sigmoid
        gate = torch.sigmoid(visual_features * t_proj_spatial)
        
        # Step 3: Modulate visual features
        v_mod = visual_features * gate
        
        # Step 4: Sequence modeling with Mamba
        v_out = self.mamba(v_mod)
        
        # Residual connection
        out = v_out + visual_features
        
        return out


class TextUMamba(nn.Module):
    """
    Text-U-Mamba: Inject text guidance into U-Mamba bottleneck.
    Architecture:
        Input Image → U-Mamba Encoder → Text-Gated Fusion → U-Mamba Decoder → Mask
                      (ResNet + Mamba)    (Bottleneck)     (Mamba + Upsample)
    """
    def __init__(
        self, 
        num_classes=1,
        input_channels=3,
        input_size=(224, 224),
        text_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        deep_supervision=False
    ):
        super().__init__()
        
        # --- 1. Text Encoder (Frozen BiomedCLIP) ---
        print(f"Loading Text Encoder: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        text_dim = self.text_encoder.config.hidden_size  # 768 for PubMedBERT
        
        # --- 2. Visual Encoder/Decoder (U-Mamba) ---
        print("Initializing U-Mamba backbone...")
        
        # U-Mamba configuration (simplified for 2D)
        n_stages = 6
        features_per_stage = [32, 64, 128, 256, 320, 320]  # Standard U-Mamba-Bot config
        
        self.umamba = UMambaEnc(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv2d,
            kernel_sizes=[[3, 3]] * n_stages,
            strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            n_conv_per_stage=[2, 2, 2, 2, 2, 2],
            num_classes=num_classes,
            n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=deep_supervision
        )
        
        # --- 3. Text-Gated Fusion (Our Addition) ---
        bottleneck_dim = features_per_stage[-1]  # 320
        self.fusion_block = TextGatedMambaBlock(
            visual_dim=bottleneck_dim,
            text_dim=text_dim
        )
        
        print(f"✓ Text-U-Mamba initialized (Bottleneck dim: {bottleneck_dim})")
        
    def forward(self, x, input_ids, attention_mask=None):
        """
        Args:
            x: [B, 3, H, W] input image
            input_ids: [B, L] tokenized text
            attention_mask: [B, L] (optional)
        Returns:
            out: [B, num_classes, H, W] segmentation mask
        """
        # A. Text Encoding (Frozen)
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            # Use [CLS] token embedding
            text_embed = text_outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        # B. Visual Encoding (U-Mamba Encoder)
        # encoder returns list of skip connections: [stem, stage1, ..., stage6 (bottleneck)]
        skips = self.umamba.encoder(x)
        
        # C. Text-Guided Fusion at Bottleneck
        bottleneck_features = skips[-1]  # [B, 320, H/32, W/32]
        fused_bottleneck = self.fusion_block(bottleneck_features, text_embed)
        
        # Replace bottleneck in skips
        skips[-1] = fused_bottleneck
        
        # D. Decoding (U-Mamba Decoder)
        out = self.umamba.decoder(skips)
        
        return out


# --- Test/Debug Entry Point ---
if __name__ == "__main__":
    print("="*60)
    print("Testing Text-U-Mamba")
    print("="*60)
    
    # Create model
    model = TextUMamba(num_classes=1)
    
    # Dummy inputs
    img = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 128))
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(img, input_ids)
    
    print(f"✓ Output shape: {out.shape}")
    print(f"✓ Expected: [2, 1, 224, 224]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    print("\n✅ Text-U-Mamba is ready!")
