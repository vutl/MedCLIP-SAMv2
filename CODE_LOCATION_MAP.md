# FreqMedCLIP Code Location Map

## 📍 Core Components

### 1. **Frequency & Fusion Modules**
```
📂 scripts/freq_components.py
   ├── DWTForward                    # Wavelet transform (Haar filters)
   │   ├── __init__()               # Initialize 4 Haar kernels (LL, LH, HL, HH)
   │   └── forward()                # Split image → HF features (LH, HL, HH)
   │
   └── SmartFusionBlock              # Coarse-to-Fine fusion
       ├── __init__()               # Conv adapters + GroupNorm
       └── forward()                # Gating + Residual fusion
```

**Key Files:**
- `scripts/freq_components.py` — DWT + SmartFusionBlock
- `reference_wavelet.py` — Reference implementation
- `reference_fusion.py` — Reference implementation

---

### 2. **Training Pipeline**
```
📂 train_freq_fusion.py
   ├── FreqMedCLIPDataset           # Load images/masks/prompts
   │   ├── __init__()              # Setup data paths
   │   ├── __len__()               # Dataset size
   │   └── __getitem__()           # Load 1 sample
   │
   ├── FrequencyMedCLIPSAMv2        # Model wrapper
   │   ├── __init__()              # Wrap BiomedCLIP + DWT + Fusion
   │   └── forward()               # Full forward pass
   │
   ├── DiceLoss                     # Loss function
   │   └── forward()               # Dice coefficient
   │
   └── main()                       # Training loop
       ├── Load data               # DataLoader
       ├── Initialize model        # BiomedCLIP + components
       ├── Training loop           # Backprop + optimize
       └── Save checkpoints        # Every 10 epochs
```

**Helper Scripts:**
- `run_train_phase2.ps1` — PowerShell wrapper
- `train_and_eval.bat` — Train + evaluate

---

### 3. **Postprocessing (THÊM MỚI)**
```
📂 scripts/postprocess.py
   ├── postprocess_saliency_kmeans()     # KMeans clustering (default)
   │   ├── Resize to 256x256            # Speed up
   │   ├── KMeans(n_clusters=2)         # Foreground/Background
   │   ├── Identify foreground          # Higher centroid
   │   ├── Resize back                  # Original size
   │   └── Filter top-K components      # Keep largest
   │
   └── postprocess_saliency_threshold()  # Simple thresholding
       ├── Apply threshold              # Fixed value
       └── Filter top-K components      # Keep largest
```

**Batch Processing Scripts:**
- `save_freqmedclip_predictions.py` — Generate raw saliency maps
- `postprocess_freqmedclip_outputs.py` — Batch postprocess
- `run_freqmedclip_pipeline.ps1` — Complete pipeline

**Visualization:**
- `visualize_prediction.py` — Before/after comparison (3x3 grid)

---

### 4. **Evaluation**
```
📂 evaluation/
   ├── eval.py                      # Compute Dice/IoU
   └── SurfaceDice.py              # Surface Dice (NSD)
```

**Comparison Scripts:**
- `compare_epochs.py` — Compare different checkpoints
- `create_overlays.py` — Create overlay visualizations

---

### 5. **Text Prompts**
```
📂 saliency_maps/text_prompts.py
   ├── breast_tumor_P2_prompts     # 20 prompts for breast tumors
   ├── brain_tumor_prompts         # Brain tumor prompts
   ├── lung_CT_prompts            # Lung CT prompts
   └── lung_Xray_prompts          # Lung X-ray prompts

📂 saliency_maps/text_prompts/
   ├── breast_tumors_testing.json  # JSON mapping: {filename: prompt}
   ├── brain_tumors_testing.json
   ├── lung_CT_testing.json
   └── lung_Xray_testing.json
```

**Create Prompts:**
- `scripts/create_prompts.py` — Generate JSON prompt files

---

### 6. **BiomedCLIP Wrapper**
```
📂 scripts/biomedclip_wrapper.py   # For BiomedCLIP (timm-free)
📂 scripts/clip_wrapper.py         # For OpenAI CLIP (legacy)
📂 scripts/methods.py              # M2IB/IBA methods
   ├── vision_heatmap_iba()        # Vision saliency map
   ├── text_heatmap_iba()          # Text saliency map
   └── vision_heatmap_freq_aware() # Frequency-aware (NEW)
```

---

### 7. **SAM Integration**
```
📂 segment-anything/
   ├── prompt_sam.py               # SAM refinement script
   └── sam_checkpoints/
       └── sam_vit_h_4b8939.pth    # SAM-ViT-H checkpoint
```

---

## 🔄 Pipeline Flow

### **Phase 2 Training (Current)**
```
Input Image (224x224)
    ↓
[BiomedCLIP Vision Encoder]
    ├── Deep Layers (7-11) → LF Features (semantic) → Coarse Map (M2IB)
    └── Shallow Layers (3-4) → HF base
    ↓
[DWT Forward] (scripts/freq_components.py)
    ↓ Input: pixel_values (B,3,224,224)
    ↓ Output: HF_wavelet (B,9,112,112)
    ↓
[Concatenate] shallow_HF + DWT_HF → hf_features (B,777,112,112)
    ↓
[SmartFusionBlock] (scripts/freq_components.py)
    ↓ Input: hf_features + coarse_map
    ↓ Output: fine_map (B,1,112,112)
    ↓
[Loss] BCE + Dice with GT masks
    ↓
[Optimizer] AdamW (lr=1e-4)
    ↓
Checkpoints saved every 10 epochs
```

### **Inference Pipeline (NEW with Postprocessing)**
```
Input Image
    ↓
[Trained FreqMedCLIP] (train_freq_fusion.py)
    ↓ Output: fine_map (B,1,224,224)
    ↓
[Upsample + Sigmoid]
    ↓ Output: raw_saliency (224,224) [0-1]
    ↓
[POSTPROCESSING] (scripts/postprocess.py)  ← THÊM MỚI
    ├── KMeans clustering (default)
    ├── Connected components filtering
    └── Keep top-K largest
    ↓ Output: cleaned_mask (224,224) {0,255}
    ↓
[SAM Refinement] (segment-anything/prompt_sam.py)
    ├── Extract bounding box from cleaned_mask
    ├── SAM inference with box prompt
    └── Output: final_mask (original size)
    ↓
Final Segmentation
```

---

## 📂 File Organization

```
MedCLIP-SAMv2/
├── scripts/
│   ├── freq_components.py              ✅ DWT + SmartFusionBlock
│   ├── postprocess.py                  ✅ KMeans + Threshold (NEW)
│   ├── biomedclip_wrapper.py           ✅ BiomedCLIP encoder wrapper
│   ├── methods.py                      ✅ M2IB/IBA methods
│   ├── utils.py                        ✅ Helper functions
│   ├── plot.py                         ✅ Visualization utils
│   └── create_prompts.py               ✅ Generate prompt JSONs
│
├── saliency_maps/
│   ├── text_prompts.py                 ✅ Python prompt lists
│   ├── text_prompts/                   ✅ JSON prompt files
│   └── model/                          ✅ BiomedCLIP config
│
├── train_freq_fusion.py                ✅ Phase 2 training script
├── save_freqmedclip_predictions.py     ✅ Batch generate predictions (NEW)
├── postprocess_freqmedclip_outputs.py  ✅ Batch postprocessing (NEW)
├── visualize_prediction.py             ✅ Before/after visualization (UPDATED)
├── run_freqmedclip_pipeline.ps1        ✅ Complete pipeline (NEW)
│
├── evaluation/
│   ├── eval.py                         ✅ Dice/IoU evaluation
│   └── SurfaceDice.py                  ✅ NSD metric
│
├── segment-anything/
│   ├── prompt_sam.py                   ✅ SAM refinement
│   └── sam_checkpoints/                ✅ SAM weights
│
├── checkpoints/                        📁 Saved fusion weights
│   └── breast_tumors/
│       └── fusion_breast_tumors_epoch100.pth
│
├── predictions/                        📁 Inference outputs (NEW)
│   ├── breast_tumors_raw/              → Raw saliency maps
│   └── breast_tumors_cleaned/          → Cleaned masks (USE THIS)
│
├── visualizations/                     📁 Visual comparisons
│   └── breast_tumors/
│       └── visual_*.png                → 3x3 grid comparisons
│
├── data/                               📁 Datasets
│   ├── breast_tumors/
│   │   ├── train_images/
│   │   ├── train_masks/
│   │   ├── val_images/
│   │   ├── val_masks/
│   │   ├── test_images/
│   │   └── test_masks/
│   ├── brain_tumors/
│   ├── lung_CT/
│   └── lung_Xray/
│
└── Documentation/
    ├── PIPELINE_-FreqMedCLIP-Smart-Single-Stream.md  ✅ Architecture overview
    ├── FreqMedCLIP_Implementation_Guide.md           ✅ Detailed guide
    ├── Pipeline.md                                   ✅ Detailed pipeline
    ├── POSTPROCESSING_GUIDE.md                       ✅ Postprocessing guide (NEW)
    └── IMPLEMENTATION_SUMMARY.md                     ✅ TGCAM variant
```

---

## 🎯 Quick Reference

### **Train FreqMedCLIP**
```powershell
.\run_train_phase2.ps1 -Dataset breast_tumors -Epochs 100
```

### **Generate Predictions + Postprocess**
```powershell
.\run_freqmedclip_pipeline.ps1 `
    -Dataset breast_tumors `
    -Checkpoint checkpoints/breast_tumors/fusion_breast_tumors_epoch100.pth
```

### **Evaluate Results**
```bash
python evaluation/eval.py \
    --pred-dir predictions/breast_tumors_cleaned \
    --gt-dir data/breast_tumors/test_masks
```

### **Visualize Sample**
```bash
python visualize_prediction.py
```

---

## 🆕 What's New (Postprocessing Integration)

1. **`scripts/postprocess.py`**
   - `postprocess_saliency_kmeans()` — KMeans clustering
   - `postprocess_saliency_threshold()` — Simple thresholding

2. **`save_freqmedclip_predictions.py`**
   - Batch generate raw saliency maps

3. **`postprocess_freqmedclip_outputs.py`**
   - Batch postprocess predictions
   - Support KMeans and threshold methods

4. **`visualize_prediction.py` (UPDATED)**
   - Show before/after postprocessing
   - 3x3 grid with overlays

5. **`run_freqmedclip_pipeline.ps1`**
   - Complete pipeline: predict → postprocess → visualize

6. **`POSTPROCESSING_GUIDE.md`**
   - Comprehensive postprocessing documentation

---

## 📚 Documentation Files

- **`PIPELINE_-FreqMedCLIP-Smart-Single-Stream.md`** — High-level architecture
- **`FreqMedCLIP_Implementation_Guide.md`** — Detailed implementation
- **`Pipeline.md`** — Step-by-step pipeline (99 lines)
- **`POSTPROCESSING_GUIDE.md`** — Postprocessing methods & usage (NEW)
- **`IMPLEMENTATION_SUMMARY.md`** — TGCAM variant summary

---

**Last Updated:** December 1, 2025  
**Status:** Production-ready with postprocessing  
**Next Phase:** SAM integration + Phase 3 (weakly-supervised nnU-Net)
