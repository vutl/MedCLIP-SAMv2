import sys
sys.path.insert(0, 'd:/Documents/LMIS/MedCLIP-SAMv2')

import torch
from train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2, DiceLoss
from scripts.freq_components import SmartFusionBlock, DWTForward
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import torch.nn as nn

print("=" * 50)
print("DEBUG: Testing FreqMedCLIP Training Pipeline")
print("=" * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n1. Device: {device}")

# Load models
print("\n2. Loading BiomedCLIP...")
model_name = "chuhac/BiomedCLIP-vit-bert-hf"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
print("   ✓ BiomedCLIP loaded")

# Initialize components
print("\n3. Initializing components...")
dwt = DWTForward().to(device)
fusion = SmartFusionBlock(hf_channels=777, lf_channels=1, out_channels=32).to(device)
print("   ✓ DWT and Fusion initialized")

# Create dataset
print("\n4. Loading dataset...")
class Args:
    pass
args = Args()
dataset = FreqMedCLIPDataset('data', 'breast_tumors', processor, tokenizer, split='train')
print(f"   ✓ Dataset loaded: {len(dataset)} samples")

# Create model
print("\n5. Creating model wrapper...")
model = FrequencyMedCLIPSAMv2(biomedclip, fusion, dwt, args).to(device)
print("   ✓ Model created")

# Test forward pass
print("\n6. Testing forward pass...")
try:
    sample = dataset[0]
    pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device).float()
    
    print(f"   - Input shape: {pixel_values.shape}")
    print(f"   - Text shape: {input_ids.shape}")
    print(f"   - Mask shape: {mask.shape}")
    
    with torch.no_grad():
        output = model(pixel_values, input_ids)
    
    print(f"   - Output shape: {output.shape}")
    print("   ✓ Forward pass successful!")
    
    # Test loss
    print("\n7. Testing loss calculation...")
    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    loss_d = dice_loss(output.squeeze(1), mask)
    loss_b = bce_loss(output.squeeze(1), mask)
    total_loss = loss_d + loss_b
    
    print(f"   - Dice Loss: {loss_d.item():.4f}")
    print(f"   - BCE Loss: {loss_b.item():.4f}")
    print(f"   - Total Loss: {total_loss.item():.4f}")
    print("   ✓ Loss calculation successful!")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
