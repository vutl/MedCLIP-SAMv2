import torch
import torch.nn as nn
import torch.nn.functional as F

class SmartFusionBlock(nn.Module):
    """
    FFBI-inspired Coarse-to-Fine Fusion Module.
    Implements the 'Gating' and 'Sharpening' logic from the pipeline.
    """
    def __init__(self, hf_channels, lf_channels, out_channels):
        super(SmartFusionBlock, self).__init__()
        
        # Adapters to match dimensions
        self.hf_adapter = nn.Conv2d(hf_channels, out_channels, kernel_size=1)
        self.lf_adapter = nn.Conv2d(lf_channels, out_channels, kernel_size=1)
        
        # Refinement convolution
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion layer
        self.final_conv = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, hf_features, coarse_map):
        """
        Args:
            hf_features (torch.Tensor): High-Frequency features from shallow layers + Wavelet.
            coarse_map (torch.Tensor): Coarse Saliency Map from Deep Layers (M2IB output).
        """
        # 1. Align Dimensions (Upsample Coarse Map to match HF resolution)
        target_size = hf_features.shape[-2:]
        coarse_map_up = F.interpolate(coarse_map, size=target_size, mode='bilinear', align_corners=False)
        
        # Project features
        F_hf = self.hf_adapter(hf_features)
        S_coarse = self.lf_adapter(coarse_map_up) # If coarse_map is already 1 channel, this might be skipped
        
        # 2. Gating Mechanism (Section 3.2 in Pipeline.pdf)
        # "Use S_coarse to 'mask' F_HF"
        # We use sigmoid to treat coarse map as a probability attention mask
        attention_mask = torch.sigmoid(S_coarse)
        F_hf_masked = F_hf * attention_mask
        
        # 3. Sharpening / Residual Fusion
        # Combine the masked details with the original coarse structure
        # "Activate pixels with high gradient" -> The HF features inherently contain gradients/edges
        fused = F_hf_masked + S_coarse
        
        # 4. Final Refinement
        refined_feat = self.refine_conv(fused)
        
        # Output: Fine Saliency Map
        S_fine = self.final_conv(refined_feat)
        
        return S_fine

# Example usage for the Agent
if __name__ == "__main__":
    # Example: HF has 64 channels, LF (Coarse Map) has 1 channel
    fusion = SmartFusionBlock(hf_channels=64, lf_channels=1, out_channels=32)
    
    dummy_hf = torch.randn(1, 64, 112, 112) # From shallow layer
    dummy_coarse = torch.randn(1, 1, 14, 14) # From deep layer (M2IB)
    
    s_fine = fusion(dummy_hf, dummy_coarse)
    print(f"Fine Map Shape: {s_fine.shape}") # Should be [1, 1, 112, 112]