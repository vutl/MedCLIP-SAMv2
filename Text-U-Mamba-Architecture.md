# Model Specification: Text-U-Mamba

**Design Philosophy:**
By enforcing linear complexity $O(N)$ across both the encoder and the fusion mechanism, we eliminate the memory bottlenecks that necessitated complex cropping strategies in prior Transformer-based models like SegVol.

## 1\. High-Level Architecture

The system comprises three streamlined modules:

1.  **Text Encoder:** A frozen, domain-specific biomedical BERT.
2.  **Visual Encoder:** A Hybrid CNN-Mamba block (U-Mamba) for feature extraction.
3.  **Fusion Bottleneck:** A novel **Text-Gated Mamba Block** that injects semantic control without quadratic cross-attention.

### System Diagram

```mermaid
graph TD
    classDef visual fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef text fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef fusion fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    subgraph Text_Stream
        Input_Text["Prompt: 'Hepatic Tumor'"]:::text --> PMB[PubMedBERT (Frozen)]:::text
        PMB --> Pooling[Global Average Pooling]:::text
        Pooling --> T_Emb[Text Embedding Vector (1xD)]:::text
    end

    subgraph U_Mamba_Encoder
        Vol[3D Volume Patch (Large Window)]:::visual --> S1[Stage 1: ResNet]:::visual
        S1 --> S2[Stage 2: ResNet]:::visual
        S2 --> S3[Stage 3: Mamba-2]:::visual
        S3 --> S4[Stage 4: Mamba-2]:::visual
    end

    subgraph The_Bottleneck
        S4 & T_Emb --> Gate[Sigmoid Gate Generation]:::fusion
        Gate --> Multiply[Element-wise Modulation]:::fusion
        Multiply --> MambaFusion[Mamba-2 Sequence Modeling]:::fusion
    end

    subgraph Decoder
        MambaFusion --> Up1[Up-Conv]:::visual
        Up1 & S3 --> Cat1[Concat]:::visual
        Cat1 --> Up2[Up-Conv]:::visual
        Up2 --> Head[Segmentation Head]:::visual
    end
```

-----

## 2\. Component Technical Specifications

### A. The Visual Backbone: U-Mamba Enc

We retain the **U-Mamba** encoder structure. [cite\_start]It provides the necessary inductive bias for biomedical imaging (via CNNs) and the long-range dependency for 3D volumes (via Mamba)[cite: 5842].

  * **Stages 1-2 (ResNet-based):** Standard Convolutional blocks to extract local textures (edges, tissue heterogeneity).
  * **Stages 3-4 (Mamba-based):** Replaces the Transformer layers found in SwinUNETR.
      * [cite\_start]**Mechanism:** Flattens the 3D feature map into a 1D sequence and processes it using Structured State Space Models (SSMs)[cite: 5866, 5894].
      * **Advantage:** scales linearly with sequence length $L$, whereas Self-Attention scales quadratically $L^2$. [cite\_start]This allows us to feed much larger input volumes than Transformer-based models[cite: 5875].

### B. The Text Encoder: PubMedBERT (Frozen)

[cite\_start]We utilize **PubMedBERT** as the text encoder, specifically the version used in **BiomedCLIP**[cite: 81].

  * [cite\_start]**Why PubMedBERT?** General domain encoders (like CLIP text encoders) fail on fine-grained medical concepts (e.g., differentiating "neoplastic cells" from "inflammatory cells")[cite: 460, 462]. [cite\_start]PubMedBERT provides superior domain-specific embeddings[cite: 1590].
  * **Configuration:** The weights are **Frozen**. [cite\_start]Fine-tuning the text encoder on segmentation datasets (which have a limited vocabulary of \~200 organ names) risks catastrophic forgetting of the broader medical knowledge base[cite: 2359].

### C. The Fusion Mechanism: Text-Gated Mamba

Instead of standard Cross-Attention (which is $O(N^2)$ regarding image tokens), we implement a **Gated Mamba** bottleneck. [cite\_start]This aligns with the "linear complexity" philosophy found in **VAMBA** [cite: 5184] [cite\_start]and **Cobra**[cite: 1243].

**The Algorithm:**
Let $V \in \mathbb{R}^{L \times D}$ be the flattened visual tokens at the bottleneck.
Let $T \in \mathbb{R}^{D}$ be the global text embedding.

1.  **Project Text:** $T_{proj} = \text{Linear}(T)$ to match visual dimension.
2.  **Generate Gate:** Create a sigmoid activation map based on the interaction between visual features and text.
    $$G = \sigma(V \cdot T_{proj})$$
3.  **Modulate:** Element-wise multiplication to suppress visual features irrelevant to the prompt (e.g., suppressing "Kidney" features when the prompt is "Liver").
    $$V_{mod} = V \odot G$$
4.  **Sequence Modeling:** Pass the modulated sequence through a standard Mamba block to reintegrate global context.
    $$V_{out} = \text{Mamba}(V_{mod})$$

-----

## 3\. Inference Strategy: Large-Window Sliding Window

[cite\_start]We explicitly **reject** the Zoom-out-zoom-in (ZOZI) strategy proposed in SegVol[cite: 2508, 2579].

  * **Critique of ZOZI:** ZOZI is a workaround for the high memory cost of Transformers ($O(N^2)$). It forces the model to process a low-res volume first to find an ROI, then crop. This introduces complexity and potential errors if the low-res pass misses a small lesion.
  * [cite\_start]**The Text-U-Mamba Solution:** Because Mamba is memory efficient ($O(N)$)[cite: 5254], we can fit significantly larger input patches into GPU memory than Transformers.
  * **Implementation:** We use the standard sliding window inference native to nnU-Net, but with **Expanded Window Sizes** (e.g., $128 \times 128 \times 128$ or larger). This reduces the number of patches required and minimizes boundary artifacts, achieving the efficiency goals of VAMBA without the architectural overhead of ZOZI.

-----

## 4\. Data Strategy: The "BiomedParse" Protocol

A universal model requires a universal label space. [cite\_start]We adopt the data curation strategy from **BiomedParse**[cite: 69, 173].

### A. Ontology Harmonization

We must merge the fragmented labels of the 25 datasets (AMOS, MSD, TotalSegmentator, etc.).

  * [cite\_start]*Method:* Use GPT-4 to map dataset-specific synonyms (e.g., "hepatic tumor", "liver mass", "HCC") to a standardized ontology (e.g., "Liver Tumor")[cite: 73, 181].

### B. Negative Prompting (Anti-Hallucination)

To prevent the model from segmenting "everything it sees," we must teach it to output **Blank Masks**.

  * *Implementation:* In 20-25% of training iterations, pair an image (e.g., Abdominal CT) with a mismatching text prompt (e.g., "Brain Tumor").
  * *Objective:* The model must minimize loss against an empty ground truth mask. [cite\_start]This technique achieved 93% precision in rejecting invalid prompts in BiomedParse[cite: 105].

-----

## 5\. Implementation Roadmap

### Phase 1: Data Curation (Critical Path)

  * [cite\_start]**Task:** Download the 25 datasets listed in SegVol[cite: 3194].
  * [cite\_start]**Task:** Run the **BiomedParse** ontology mapping script to unify labels[cite: 185].
  * **Output:** A unified `BiomedParseData` JSON format containing `{image_path, mask_path, text_prompt}` triples.

### Phase 2: Architecture Construction

  * **Task:** Clone the official `U-Mamba` repository.
  * **Task:** Inject the `PubMedBERT` encoder (using HuggingFace `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`).
  * **Task:** Implement the `Text-Gated Mamba` block at the bottleneck.

### Phase 3: Training & Evaluation

  * **Training:** Train on 8x A100 GPUs (or equivalent) using the `nnU-Net` trainer, modified to accept text inputs.
  * **Metric:** Dice Score.
  * **Specific Claim to Test:** Efficiency. [cite\_start]Measure **Training Memory (GB)** vs. **Sequence Length** to demonstrate the linear scaling advantage over SwinUNETR and Qwen2-VL based models[cite: 5439].