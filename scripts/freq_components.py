import torch
import torch.nn as nn
import torch.nn.functional as F

class DWTForward(nn.Module):
    """
    Discrete Wavelet Transform (DWT) implementation for PyTorch.
    Uses Haar Wavelet filters by default as specified in the FreqMedCLIP pipeline.
    Splits an image into 4 frequency components: LL, LH, HL, HH.
    """
    def __init__(self):
        super(DWTForward, self).__init__()
        # Haar Wavelet Filters (LL, LH, HL, HH)
        # Chuẩn hóa theo định nghĩa separable 2D DWT:
        # Low-pass: [1/√2, 1/√2], High-pass: [-1/√2, 1/√2]
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])      # LL: Low-Low (Approximation)
        lh = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])   # LH: Low-High (Horizontal edges)
        hl = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])   # HL: High-Low (Vertical edges)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])   # HH: High-High (Diagonal edges)

        # Stack filters into a 4x1x2x2 weight tensor
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('kernels', kernels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image batch of shape (B, C, H, W)
        Returns:
            torch.Tensor: High-Frequency components stacked (B, C*3, H/2, W/2) 
                          or all components depending on configuration.
        """
        b, c, h, w = x.shape
        # Apply DWT channel-wise
        # Reshape input to (B*C, 1, H, W) for group convolution
        x_reshaped = x.view(b * c, 1, h, w)
        
        # Convolve with stride 2 to downsample
        # kernels is (4, 1, 2, 2). We want to apply this to each of the B*C channels independently.
        # So we use groups=B*C? No, groups=1 because input is (B*C, 1, ...).
        # The weight should be (Out, In/Groups, kH, kW).
        # Here In=1, Groups=1. Out=4.
        # So output is (B*C, 4, H/2, W/2).
        
        out = F.conv2d(x_reshaped, self.kernels, stride=2, padding=0)
        
        # Reshape back: (B, C, 4, H/2, W/2)
        out = out.view(b, c, 4, h // 2, w // 2)
        
        # Split components
        # LL = out[:, :, 0, :, :] # Low Frequency (Approximation)
        LH = out[:, :, 1, :, :] # Horizontal Detail
        HL = out[:, :, 2, :, :] # Vertical Detail
        HH = out[:, :, 3, :, :] # Diagonal Detail
        
        # Per Pipeline.pdf: Keep only High-Frequency components (LH, HL, HH)
        # Concatenate them along the channel dimension
        # Output shape: (B, C*3, H/2, W/2)
        freq_features = torch.cat([LH, HL, HH], dim=1)
        
        return freq_features

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
            nn.GroupNorm(num_groups=8, num_channels=out_channels),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True)
        )
        
        # Final fusion layer to 1 channel (Saliency Map)
        self.final_conv = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, hf_features, coarse_map):
        """
        Args:
            hf_features (torch.Tensor): High-Frequency features (B, hf_channels, H_hf, W_hf).
            coarse_map (torch.Tensor): Coarse Saliency Map from Deep Layers (B, lf_channels, H_lf, W_lf).
        """
        # 1. Align Dimensions (Upsample Coarse Map to match HF resolution)
        target_size = hf_features.shape[-2:]
        coarse_map_up = F.interpolate(coarse_map, size=target_size, mode='bilinear', align_corners=False)
        
        # Project features
        F_hf = self.hf_adapter(hf_features)
        S_coarse = self.lf_adapter(coarse_map_up) 
        
        # 2. Gating Mechanism (Section 3.2 in Pipeline.pdf)
        # "Use S_coarse to 'mask' F_HF"
        # We use sigmoid to treat coarse map as a probability attention mask
        # Note: S_coarse here is projected features. We might want to use the raw coarse map for gating?
        # The reference uses S_coarse (projected). Let's stick to that.
        attention_mask = torch.sigmoid(S_coarse)
        F_hf_masked = F_hf * attention_mask
        
        # 3. Sharpening / Residual Fusion
        # Combine the masked details with the original coarse structure
        fused = F_hf_masked + S_coarse
        
        # 4. Final Refinement
        refined_feat = self.refine_conv(fused)
        
        # Output: Fine Saliency Map
        S_fine = self.final_conv(refined_feat)
        
        return S_fine
