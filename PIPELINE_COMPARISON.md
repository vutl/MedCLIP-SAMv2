# MedCLIP-SAMv2 Pipeline Comparison: FreqMedCLIP vs. TGCAM

This document provides a detailed technical breakdown of the two architectural variants implemented for MedCLIP-SAMv2: **FreqMedCLIP** (Frequency-Aware Fusion) and **TGCAM** (Text-Guided Common Attention Module).

---

## 1. FreqMedCLIP (Frequency-Aware Fusion)

**Core Philosophy:** "Smart Single-Stream" with Frequency Decomposition.
This approach leverages the observation that medical segmentation relies heavily on both high-frequency details (boundaries, textures) and low-frequency semantics (organ shapes, location). It uses a frozen BiomedCLIP encoder and fuses these features using a lightweight decoder.

### 1.1 Architecture Overview

```mermaid
graph TD
    Img[Input Image] --> DWT[DWT (Haar Wavelet)]
    Img --> ViT[BiomedCLIP Vision (Frozen)]
    Txt[Text Prompt] --> Bert[BiomedCLIP Text (Frozen)]

    subgraph "Feature Extraction"
        DWT -->|LL, LH, HL, HH| WaveletFeats[Wavelet Coeffs]
        ViT -->|Layer 3| ShallowFeats[Shallow Features (HF)]
        ViT -->|Last Layer| DeepFeats[Deep Features (LF)]
        Bert --> TxtEmbed[Text Embeddings]
    end

    subgraph "Coarse Map Generation"
        DeepFeats -->|Normalize| NormDeep
        TxtEmbed -->|Normalize| NormTxt
        NormDeep -->|Dot Product| CoarseMap[Coarse Saliency Map]
        NormTxt --> CoarseMap
    end

    subgraph "High-Frequency Construction"
        ShallowFeats -->|Upsample| UpShallow
        WaveletFeats -->|Concat| HF_Input
        UpShallow -->|Concat| HF_Input[HF Features Combined]
    end

    subgraph "Smart Fusion"
        HF_Input --> Fusion[SmartFusionBlock]
        CoarseMap --> Fusion
        Fusion -->|Gating & Refinement| FineSaliency[Fine Saliency Map]
    end

    FineSaliency -->|Upsample| FinalOutput[Final Segmentation/Saliency]
```

### 1.2 Detailed Pipeline Steps

1.  **Input Processing**:
    *   **Image**: Resized to 224x224.
    *   **Text**: Tokenized prompt (e.g., "A breast tumor").

2.  **Feature Extraction (Frozen BiomedCLIP)**:
    *   **Vision**:
        *   *Shallow Features*: Extracted from Layer 3 of ViT. Captures edges and textures (High Frequency proxy).
        *   *Deep Features*: Extracted from the last hidden state. Captures semantic content (Low Frequency).
    *   **Text**: Full text embeddings from the text encoder.

3.  **Frequency Decomposition (DWT)**:
    *   Input image undergoes Discrete Wavelet Transform (Haar kernel).
    *   Produces 4 sub-bands: LL (Low), LH (Vertical), HL (Horizontal), HH (Diagonal).
    *   LH, HL, HH are concatenated to form the **Wavelet Features**.

4.  **Coarse Map Generation (M2IB-style)**:
    *   Standard dot product between normalized Deep Visual Features (patches) and Text Embeddings.
    *   Result: A rough heatmap indicating where the text concept is located.

5.  **HF Feature Construction**:
    *   Shallow ViT features are upsampled to match DWT dimensions (112x112).
    *   Concatenated with Wavelet Features to form a rich **High-Frequency Feature Map**.

6.  **Smart Fusion (The Trainable Module)**:
    *   **Inputs**: HF Feature Map + Coarse Map.
    *   **Mechanism**:
        *   *Gating*: The Coarse Map acts as a spatial gate, suppressing HF noise in irrelevant regions.
        *   *Refinement*: Convolutions refine the gated HF features to produce sharp boundaries.
    *   **Output**: A high-resolution (112x112) fine saliency map.

7.  **Training**:
    *   **Loss**: Dice Loss + BCE Loss against Ground Truth masks.
    *   **Trainable Params**: Only the `SmartFusionBlock` (~142KB).

---

## 2. TGCAM (Text-Guided Common Attention Module)

**Core Philosophy:** Symmetric Attention with Iterative Refinement.
This approach addresses "Semantic Drift" and "Noise" in zero-shot/weakly-supervised settings. It treats the problem as a bidirectional alignment task where text adapts to the image context before generating the heatmap.

### 2.1 Architecture Overview

```mermaid
graph TD
    Img[Input Image] --> ViT[BiomedCLIP Vision (Frozen)]
    Txt[Text Prompt] --> Bert[BiomedCLIP Text (Frozen)]

    subgraph "Preprocessing"
        ViT -->|Remove CLS| VisualPatches[Visual Patches (N x D)]
        Bert --> TextFeats[Text Features (L x D)]
    end

    subgraph "GatedITEM (Text Refinement)"
        VisualPatches -->|Key/Value| GatedAttn
        TextFeats -->|Query| GatedAttn
        GatedAttn -->|Gate α| RefinedText[Refined Text Features]
        TextFeats -->|Residual| RefinedText
    end

    subgraph "SharpenedTGCAM (Symmetric Attention)"
        VisualPatches -->|Proj| V_Common
        RefinedText -->|Proj| T_Common
        V_Common --> Affinity[Affinity Matrix]
        T_Common --> Affinity
        
        Affinity -->|Max Pool| Saliency[Saliency Map]
        Affinity -->|Softmax(A/τ)| Context[Context Features]
        
        Context -->|Concat| FusedFeats
        VisualPatches -->|Residual| FusedFeats[Fused Visual-Text Features]
    end

    Saliency -->|Upsample| FinalSaliency[Saliency Map Output]
    FusedFeats -->|Optional| DecoderInput[Input to SAM Decoder]
```

### 2.2 Detailed Pipeline Steps

1.  **Input Processing**:
    *   Same as FreqMedCLIP.

2.  **Feature Extraction**:
    *   **Vision**: Last hidden state of ViT. **Crucial Step**: The CLS token is explicitly removed to leave only spatial patches.
    *   **Text**: Full sequence of text features.

3.  **GatedITEM (Iterative Text Enhancement)**:
    *   **Goal**: Adapt the generic text prompt to the specific image instance without losing meaning.
    *   **Mechanism**:
        *   Cross-Attention: Text queries the Image ("What parts of this image look like 'tumor'?").
        *   **Gating ($\alpha$)**: A learnable parameter controls how much visual information updates the text. Initialized low (0.1) to prevent the text from "drifting" to match incorrect visual features (e.g., matching a benign cyst when the prompt is "malignant").
    *   **Output**: Context-aware Text Embeddings.

4.  **SharpenedTGCAM**:
    *   **Symmetric Projection**: Both Image and Text are projected to a common dimension (512).
    *   **Affinity Calculation**: Matrix multiplication of Image and Text features.
    *   **Saliency Generation**:
        *   *Max Pooling*: For each image patch, take the maximum activation across all text tokens. This finds regions relevant to *any* part of the prompt.
        *   *Normalization*: Instance normalization to scale the map for thresholding.
    *   **Context Retrieval (Sharpened)**:
        *   *Temperature Scaling ($\tau=0.07$)*: Softmax with low temperature makes the attention distribution "sharp" (sparse), focusing only on the most relevant regions and suppressing background noise.
        *   Weighted sum of text features based on this sharp attention.
    *   **Fusion**: Concatenation of Visual Features and Retrieved Text Context + Residual connection.

5.  **Training**:
    *   **Loss**: Can be trained supervised (Dice/BCE) or used Zero-Shot.
    *   **Trainable Params**: `GatedITEM` + `SharpenedTGCAM`.

---

## 3. Key Differences Summary

| Feature | FreqMedCLIP | TGCAM |
| :--- | :--- | :--- |
| **Primary Signal** | **Frequency**: Combines DWT (HF) and Deep ViT (LF). | **Attention**: Symmetric Cross-Attention between Image and Text. |
| **Text Handling** | **Static**: Uses fixed text embeddings from BiomedCLIP. | **Dynamic**: Refines text embeddings iteratively based on the image (`GatedITEM`). |
| **Fusion Logic** | **Gating**: Coarse map gates HF features. | **Sharpening**: Temperature-scaled attention filters noise. |
| **Spatial Resolution** | Explicitly constructs high-res (112x112) features via DWT. | Operates at patch level (14x14 or 16x16), then upsamples. |
| **Strengths** | Excellent for **boundary precision** and texture details. Efficient. | Excellent for **semantic alignment**, noise suppression, and zero-shot robustness. |
| **Best Use Case** | Supervised training where precise segmentation masks are needed. | Zero-shot or Weakly-supervised scenarios where semantic ambiguity is high. |

## 4. Integration Recommendation

For the ultimate **MedCLIP-SAMv2**, a hybrid approach is possible:

1.  Use **TGCAM** to generate the "Coarse Map" instead of the simple dot product. This provides a much higher quality, noise-suppressed semantic prior.
2.  Feed this **TGCAM Saliency Map** into the **SmartFusionBlock** of FreqMedCLIP.
3.  The SmartFusionBlock then uses this high-quality map to gate the **DWT High-Frequency features**.

This combines the **semantic robustness** of TGCAM with the **spatial precision** of FreqMedCLIP.
