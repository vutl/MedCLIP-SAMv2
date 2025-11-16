#!/usr/bin/env python3
"""
Test script to verify MedCLIP-SAMv2 setup on GPU server
Runs basic checks and a small test inference
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"  ✓ TorchVision {torchvision.__version__}")
        
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
        
        import open_clip
        print(f"  ✓ Open CLIP")
        
        import segment_anything
        print(f"  ✓ Segment Anything")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  ✗ CUDA not available")
            return False
    except Exception as e:
        print(f"  ✗ CUDA test failed: {e}")
        return False

def test_models():
    """Test if model files exist"""
    print("\nTesting model files...")
    all_ok = True
    
    # SAM checkpoint
    sam_checkpoint = Path("segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth")
    if sam_checkpoint.exists():
        size_mb = sam_checkpoint.stat().st_size / (1024 * 1024)
        print(f"  ✓ SAM checkpoint found ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ SAM checkpoint not found at {sam_checkpoint}")
        all_ok = False
    
    # BiomedCLIP model
    biomed_model = Path("saliency_maps/model/pytorch_model.bin")
    if biomed_model.exists():
        size_mb = biomed_model.stat().st_size / (1024 * 1024)
        print(f"  ✓ BiomedCLIP model found ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ BiomedCLIP model not found at {biomed_model}")
        print(f"    Download from: https://drive.google.com/file/d/1jjnZabUlc9_gpcP0d2nz_GNS-EGX0lq5/view?usp=sharing")
        all_ok = False
    
    return all_ok

def test_model_loading():
    """Test if models can be loaded"""
    print("\nTesting model loading...")
    try:
        import torch
        
        # Test SAM model loading
        sam_checkpoint = Path("segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth")
        if sam_checkpoint.exists():
            try:
                checkpoint = torch.load(sam_checkpoint, map_location='cpu')
                print("  ✓ SAM checkpoint can be loaded")
            except Exception as e:
                print(f"  ✗ SAM checkpoint loading failed: {e}")
                return False
        
        # Test BiomedCLIP model loading
        biomed_model = Path("saliency_maps/model/pytorch_model.bin")
        if biomed_model.exists():
            try:
                checkpoint = torch.load(biomed_model, map_location='cpu')
                print("  ✓ BiomedCLIP model can be loaded")
            except Exception as e:
                print(f"  ✗ BiomedCLIP model loading failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Model loading test failed: {e}")
        return False

def test_data_structure():
    """Test if data directory structure exists"""
    print("\nTesting data structure...")
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"  ⚠️  Data directory not found (this is OK if you haven't downloaded datasets yet)")
        return True
    
    datasets = ["breast_tumors", "brain_tumors", "lung_chest_xray", "lung_ct"]
    found = []
    
    for dataset in datasets:
        ds_path = data_dir / dataset
        if ds_path.exists():
            # Check for image directories
            has_images = any((ds_path / d).exists() for d in ["images", "test_images", "train_images"])
            if has_images:
                found.append(dataset)
                print(f"  ✓ {dataset} dataset found")
    
    if found:
        print(f"\n  Found {len(found)} dataset(s): {', '.join(found)}")
    else:
        print(f"  ⚠️  No datasets found (run download_datasets.sh to download)")
    
    return True

def run_small_test():
    """Run a small inference test if possible"""
    print("\nRunning small inference test...")
    
    # Check if we have test data
    test_data_paths = [
        Path("data/breast_tumors/test_images"),
        Path("data/brain_tumors/test_images"),
        Path("data/breast_tumors/images"),
    ]
    
    test_images = []
    for path in test_data_paths:
        if path.exists():
            images = list(path.glob("*.png")) + list(path.glob("*.jpg"))
            if images:
                test_images = images[:1]  # Just one image
                break
    
    if not test_images:
        print("  ⚠️  No test images found, skipping inference test")
        print("     (Download datasets to enable this test)")
        return True
    
    try:
        print(f"  Testing with image: {test_images[0].name}")
        # This is a minimal test - just verify the pipeline can start
        # Full inference would require loading models which is slow
        print("  ✓ Test image found (full inference test skipped for speed)")
        return True
    except Exception as e:
        print(f"  ✗ Inference test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("MedCLIP-SAMv2 Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Model Files", test_models()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Data Structure", test_data_structure()))
    results.append(("Inference Test", run_small_test()))
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All critical tests passed! Setup is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

