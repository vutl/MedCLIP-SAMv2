# FreqMedCLIP Postprocessing Guide

## 📌 Vấn đề

FreqMedCLIP ban đầu **thiếu bước lọc nhiễu** (postprocessing) giống như MedCLIP-SAMv2 gốc, dẫn đến:
- Saliency maps có nhiễu (noisy predictions)
- Boundary không rõ ràng
- False positives cao

## 🔧 Giải pháp

Tích hợp **KMeans clustering** và **Connected Components filtering** như MedCLIP-SAMv2 pipeline gốc.

---

## 📂 Cấu trúc Code Postprocessing

```
scripts/
  └── postprocess.py                    # Core postprocessing functions
                                        # - postprocess_saliency_kmeans()
                                        # - postprocess_saliency_threshold()

save_freqmedclip_predictions.py        # Batch generate raw predictions
postprocess_freqmedclip_outputs.py     # Batch postprocess predictions
visualize_prediction.py                # Visualize before/after postprocessing
run_freqmedclip_pipeline.ps1           # Complete pipeline script
```

---

## 🚀 Cách sử dụng

### **Option 1: Complete Pipeline (Recommended)**

Chạy toàn bộ pipeline tự động:

```powershell
.\run_freqmedclip_pipeline.ps1 `
    -Dataset breast_tumors `
    -Checkpoint checkpoints/breast_tumors/fusion_breast_tumors_epoch100.pth
```

Output:
- `predictions/breast_tumors_raw/` — Raw saliency maps
- `predictions/breast_tumors_cleaned/` — **Cleaned masks (use for evaluation)**
- `visualizations/breast_tumors/` — Before/after comparison images

---

### **Option 2: Step-by-Step**

#### Step 1: Generate raw predictions
```bash
python save_freqmedclip_predictions.py \
    --dataset breast_tumors \
    --checkpoint checkpoints/breast_tumors/fusion_breast_tumors_epoch100.pth \
    --output predictions/breast_tumors_raw \
    --split test
```

#### Step 2: Postprocess with KMeans
```bash
python postprocess_freqmedclip_outputs.py \
    --input predictions/breast_tumors_raw \
    --output predictions/breast_tumors_cleaned \
    --method kmeans \
    --top-k 1
```

**Parameters:**
- `--method`: `kmeans` (default) or `threshold`
- `--top-k`: Keep top K largest connected components (default: 1)
  - For lungs: use `--top-k 2` (left + right lung)
- `--num-clusters`: KMeans clusters (default: 2)
- `--threshold`: Threshold value for threshold method (default: 0.3)

#### Step 3: Visualize results
```bash
python visualize_prediction.py
```

Creates 3x3 grid comparing:
- Row 1: Original, GT, Coarse Map
- Row 2: Fine Raw, Fine Binary, **Fine Cleaned** (KMeans)
- Row 3: GT Overlay, Binary Overlay, **Cleaned Overlay**

---

## 📊 Postprocessing Methods

### 1. **KMeans Clustering** (Default - Recommended)
**Method:** `kmeans`

**Pipeline:**
1. Resize saliency map to 256x256 (faster)
2. KMeans clustering (2 clusters: foreground vs background)
3. Identify foreground cluster (higher centroid value)
4. Resize back to original size
5. Keep only top K largest connected components

**Pros:**
- Robust to noise
- Adaptive thresholding (không cần chọn threshold cố định)
- Same method as MedCLIP-SAMv2 gốc

**Cons:**
- Slower than thresholding (~2-3x)

**Best for:** General use, high-noise predictions

---

### 2. **Simple Thresholding**
**Method:** `threshold`

**Pipeline:**
1. Apply fixed threshold (default: 0.3)
2. Keep only top K largest connected components

**Pros:**
- Very fast
- Deterministic

**Cons:**
- Threshold cần tune cho từng dataset
- Less robust to varying intensity distributions

**Best for:** Quick testing, low-noise predictions

---

## 🎯 Khi nào dùng `--top-k`?

| Dataset | `--top-k` | Lý do |
|---------|-----------|-------|
| `breast_tumors` | 1 | Chỉ 1 tumor per image |
| `brain_tumors` | 1 | 1 tumor region |
| `lung_CT` | **2** | Left + Right lung |
| `lung_Xray` | **2** | Left + Right lung |

---

## 📈 So sánh Performance

**Dựa trên MedCLIP-SAMv2 pipeline gốc:**

| Stage | DSC (Dice) | Note |
|-------|------------|------|
| Stage 1: Raw Saliency (M2IB) | ~46% | Noisy, blobby |
| **Stage 2: Postprocess (KMeans)** | **~58%** | **+12% improvement** |
| Stage 3: SAM Refinement | ~78% | Final output |

**FreqMedCLIP expected:**
- Raw saliency: ~50-55% (better than M2IB due to HF features)
- **After postprocessing: ~60-65%**
- After SAM: **~80-85% (target)**

---

## 🔍 Troubleshooting

### Problem: Cleaned mask vẫn còn nhiều noise
**Solution:**
```bash
# Tăng top-k lên 2-3 nếu cần giữ nhiều components
python postprocess_freqmedclip_outputs.py \
    --input predictions/raw \
    --output predictions/cleaned \
    --method kmeans \
    --top-k 2
```

### Problem: Cleaned mask quá aggressive (mất mát vùng quan trọng)
**Solution:**
```bash
# Dùng threshold method với threshold thấp hơn
python postprocess_freqmedclip_outputs.py \
    --input predictions/raw \
    --output predictions/cleaned \
    --method threshold \
    --threshold 0.2 \
    --top-k 1
```

### Problem: Multiple objects nhưng chỉ giữ 1
**Solution:**
```bash
# Tăng top-k
python postprocess_freqmedclip_outputs.py \
    --input predictions/raw \
    --output predictions/cleaned \
    --method kmeans \
    --top-k 3  # Keep top 3 largest components
```

---

## 📝 Integration với Pipeline Gốc

**MedCLIP-SAMv2 Original:**
```
M2IB → [Postprocess] → SAM
```

**FreqMedCLIP New:**
```
SmartFusionBlock → [Postprocess] → SAM
                     ↑
                   THÊM MỚI
```

**Code location:**
- Postprocessing functions: `scripts/postprocess.py`
- Batch script: `postprocess_freqmedclip_outputs.py`
- Visualization: `visualize_prediction.py` (updated)

---

## ✅ Validation

Sau khi postprocess, evaluate results:

```bash
python evaluation/eval.py \
    --pred-dir predictions/breast_tumors_cleaned \
    --gt-dir data/breast_tumors/test_masks
```

Expected metrics:
- **Dice Score (DSC):** 60-65% (before SAM)
- **IoU:** 55-60%
- **Hausdorff Distance:** Improved boundary accuracy

---

## 🎨 Visualization Output

`visualize_prediction.py` tạo 3x3 grid:

```
| Original Image | Ground Truth | Coarse Map (Text-Guided) |
|----------------|--------------|--------------------------|
| Fine Raw       | Fine Binary  | Fine Cleaned (KMeans) ← |
| GT Overlay     | Binary       | Cleaned Overlay        ← |
```

**Màu sắc:**
- Red: Ground Truth
- Green: Binary prediction (threshold=0.5)
- Blue: **Cleaned prediction (recommended)**

---

## 📚 References

- MedCLIP-SAMv2 original: `postprocessing/postprocess_saliency_maps.py`
- FMISeg paper: Uses similar frequency-aware postprocessing
- Implementation: `scripts/postprocess.py`

---

## 🚨 Important Notes

1. **ALWAYS postprocess before evaluation**
   - Raw saliency maps are NOT final predictions
   - Postprocessing is essential part of pipeline

2. **For SAM refinement:**
   - Use cleaned masks to generate bounding boxes
   - Better boxes → Better SAM output

3. **For comparison with baselines:**
   - Compare cleaned FreqMedCLIP vs. MedCLIP-SAMv2 Stage 2 output
   - Both should use same postprocessing method

---

**Updated:** December 1, 2025  
**Author:** Ngo Thanh Trung  
**Version:** 1.0 (Production-ready)
