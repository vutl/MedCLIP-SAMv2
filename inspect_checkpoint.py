import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load checkpoint
checkpoint_path = "checkpoints/fusion_breast_tumors_epoch10.pth"
fusion_state = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint loaded successfully!")
print(f"Number of parameters: {len(fusion_state)}")
print("\nParameter shapes:")
for name, param in fusion_state.items():
    print(f"  {name}: {param.shape}")
    
# Check if weights have changed (not all zeros/ones)
print("\nParameter statistics:")
for name, param in fusion_state.items():
    print(f"  {name}:")
    print(f"    Mean: {param.mean().item():.6f}")
    print(f"    Std:  {param.std().item():.6f}")
    print(f"    Min:  {param.min().item():.6f}")
    print(f"    Max:  {param.max().item():.6f}")
