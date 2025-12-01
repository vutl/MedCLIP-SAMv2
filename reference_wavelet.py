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
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

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
        out = F.conv2d(x_reshaped, self.kernels, stride=2, padding=0)
        
        # Reshape back: (B, C, 4, H/2, W/2)
        out = out.view(b, c, 4, h // 2, w // 2)
        
        # Split components
        LL = out[:, :, 0, :, :]
        LH = out[:, :, 1, :, :]
        HL = out[:, :, 2, :, :]
        HH = out[:, :, 3, :, :]
        
        # Per Pipeline.pdf: Keep only High-Frequency components (LH, HL, HH)
        # Concatenate them along the channel dimension
        # Output shape: (B, C*3, H/2, W/2)
        freq_features = torch.cat([LH, HL, HH], dim=1)
        
        return freq_features

# Example usage for the Agent
if __name__ == "__main__":
    dwt = DWTForward()
    dummy_img = torch.randn(1, 3, 224, 224) # B, C, H, W
    hf_feats = dwt(dummy_img)
    print(f"Input shape: {dummy_img.shape}")
    print(f"HF Features shape: {hf_feats.shape}") # Should be [1, 9, 112, 112]