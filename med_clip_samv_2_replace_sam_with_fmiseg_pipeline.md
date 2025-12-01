# High-level idea (one sentence)
Replace SAM (promptable mask decoder) with FMISeg (an end-to-end frequency-domain language-guided segmentation network) so the pipeline becomes **BiomedCLIP (fine-tuned) + M2IB saliency/LLM prompts → FMISeg** as the single text-guided segmentation head, with optional pseudo-labeling, ensemble and distillation to keep/replicate SAM’s zero-shot advantages.  

---

# Full pipeline — modules (order of execution)
1. **Input / Preprocessing**
   - Raw image `I` (H×W×3), text prompt `T` (string or tokens).
   - Standardize image (pixel range, mean/std), resize to model resolution (e.g., 512×512 or 384×384 depending on compute). Keep aspect ratio with padding if needed.

2. **BiomedCLIP fine-tuned encoders**
   - Image encoder `Φ_img` → image embedding `Z_img`.
   - Text encoder `Φ_text` → text embedding(s) `Z_text` (word tokens or pooled vector).
   - Fine-tune with DHN-NCE as in MedCLIP-SAMv2 for better domain alignment.

3. **M2IB saliency map generator (Multi-modal Information Bottleneck)**
   - Inputs: `Z_img`, `Z_text` → M2IB → continuous saliency map `S` ∈ [0,1]^{H’×W’} (H’,W’ = pre-decoder spatial resolution, e.g., same as input after upsample/feat-up). Use same formulation as MedCLIP-SAMv2 to produce pixel importance w.r.t. text. `S` will be used as an *attention prior / auxiliary input* to FMISeg.

4. **(Optional) LLM prompt generator**
   - If you use LLM to expand/engineer `T`, generate multiple prompt variants and average or ensemble text features. This is optional but recommended (paper shows prompt engineering helps).

5. **FMISeg preprocessing: wavelet decomposition → HF / LF images**
   - Apply 2D discrete wavelet transform (e.g., Haar, db1) to `I` producing:
     - Low-frequency image `I_LF` (captures coarse semantics), shape H×W×3
     - High-frequency image `I_HF` (captures edges/textures), shape H×W×3  
   - This matches FMISeg design. 

6. **Dual-branch visual encoders (FMISeg)**
   - Two encoders (shared architecture, separate weights): `Enc_LF` and `Enc_HF` (ConvNeXt-Tiny in paper).  
   - Multi-stage feature outputs at four scales (downsample rates 4,8,16,32):
     - For each branch `m ∈ {LF, HF}` produce:  
       F1_m ∈ R^{H/4 × W/4 × C1}, F2_m ∈ R^{H/8 × W/8 × C2}, F3_m ∈ R^{H/16 × W/16 × C3}, F4_m ∈ R^{H/32 × W/32 × C4}. 

7. **Inject M2IB saliency prior**
   - Two options (both valid):
     - **A (strong preference)**: Upsample `S` to each feature map spatial size and **concatenate** as an extra channel into `F*_m` before FFBI/LFFI. Concatenation requires a 1×1 conv to match channel dims. This lets FMISeg use the saliency map as a spatial prior. (Recommended.)  
     - **B**: Multiply `F*_m` by `S` (elementwise), i.e. attention gating. Use sigmoid normalized `S`.  
   - Rationale: reuse MedCLIP saliency info that was previously feeding SAM prompts; now it serves as a learned prior for FMISeg. 

8. **FFBI (Frequency-domain Feature Bidirectional Interaction) module**
   - Input: F4_HF and F4_LF (the deepest features from each branch).  
   - Operation: multi-head cross-attention in both directions (HF→LF and LF→HF), producing enhanced `F4_HF_hat` and `F4_LF_hat`. Formulas in FMISeg:  
     `F4_HF_hat = LN(F4_HF + MHCA(F4_HF, F4_LF, F4_LF))` and analogous for LF. 

9. **Dual-branch decoders with LFFI (Language & Frequency-domain Feature Interaction)**
   - Each branch has decoder upsampling blocks that progressively upsample features from scales 32→16→8→4. At each decoder stage:
     - Inject corresponding skip connections from encoder F{3,2,1}_m.
     - Apply **LFFI**: cross-attention between visual features `F_stage_m` and **linguistic features** `F_T` (word-level embeddings), plus semantically-irrelevant filter (matrix multiply + linear + sigmoid) that produces per-spatial filter weights to reweight visual features. See FMISeg equations (5,6). 
   - Important: FMISeg assumes **CXR-BERT** produces `F_T ∈ R^{L × C}`. MedCLIP uses BiomedCLIP text encoder (PubMedBERT style) producing Z_text in a different embedding space/dimension. You **must** map BiomedCLIP text outputs to the LFFI expected shape. Implement a `TextAdapter` (linear + LayerNorm + optional small Transformer) to convert:
     - `Z_text` (pool vector) or `Z_text_tokens` (L×d_b) → `F_T ∈ R^{L’ × C}` where C matches decoder channel (e.g., 256/384/768). This can be a trainable linear projection (mandatory). 

10. **Dual segmentation heads**
    - FMISeg predicts two masks independently: `Mask_HF` and `Mask_LF` (upsampled to input H×W). Optionally fuse via learned conv or simple average → final `P_mask` (probability map). 

11. **Post-processing**
    - Threshold `P_mask` (e.g., 0.5), morphological ops (remove small connected components), optionally CRF/refinement.
    - If you want uncertainty, keep ensemble of checkpoints or Monte-Carlo Dropout to produce `uncertainty_map`. MedCLIP used checkpoint ensembling for uncertainty — reuse same approach. 

12. **Optional nnUNet / Weakly-supervised refinement**
    - If you want to replicate MedCLIP’s weakly supervised stage: generate pseudo-masks using current pipeline (FMISeg outputs or ensemble of FMISeg+SAM if you run SAM as teacher), then train nnUNet on pseudo-labels with checkpoint ensembling to produce lower-variance / higher-quality masks. This is optional but recommended if you need production-grade segmentation. 

---

# Detailed data flow with tensor shapes (concrete example)
Assume input resolution = 512×512, ConvNeXt channels C1=64, C2=128, C3=256, C4=512, text dim target C=256. Use these as working numbers (you may adapt).

1. Input: `I` ∈ R^{512×512×3}, `T` is raw string.
2. BiomedCLIP:
   - `Z_img` = Φ_img(I) → global vector R^{d_img} (e.g., 1024).
   - `Z_text_tokens` = Φ_text(T) → token features R^{L × d_text} (e.g., L=32, d_text=768). Or pooled vector R^{d_text}. 
3. M2IB:
   - Inputs: `Z_img`, `Z_text_tokens` → M2IB → `S_cont` ∈ R^{64×64} (downsampled map at H/8 or H/4 depending on implementation). Upsample to required sizes: `S_4` (H/4=128), `S_8` (H/8=64), `S_16`(32), `S_32`(16).
4. Wavelet:
   - `I_LF`, `I_HF` ∈ R^{512×512×3}
5. Encoders:
   - `F1_LF` ∈ R^{128×128×64}, `F2_LF` ∈ R^{64×64×128}, `F3_LF` ∈ R^{32×32×256}, `F4_LF` ∈ R^{16×16×512}
   - `F1_HF` … `F4_HF` similarly. 
6. Inject `S`:
   - Upsample `S_4→128×128`, concat to F1_* → conv1 to output same channels (64). For each stage do same. (Concatenate increases channels by 1; conv1 reduces back).
7. FFBI on deepest:
   - Input `F4_HF` (16×16×512), `F4_LF` (16×16×512) → cross-attention → `F4_HF_hat`, `F4_LF_hat` same shapes. 
8. Text adapter:
   - `Z_text_tokens` (L×768) → Linear(768→256) → `F_T` (L×256) (or apply small Transformer to get contextualized tokens). This `F_T` is passed to LFFI. (If BiomedCLIP tokens are different dim, adapt accordingly.)
9. Decoder stage (example for one stage):
   - Upsample `F4_HF_hat` to 32×32, add skip `F3_HF` (32×32×256) via concat → conv → `D3_HF` (32×32×256).
   - Apply LFFI: MHCA(D3_HF queries, keys=F_T, values=F_T) => `D3_HF'`, compute `F_M = D3_HF' ⊗ F_T^T` → Linear → sigmoid => filter map (32×32×L) → elementwise multiply to modulate D3_HF' → combine & conv → output `D3_HF_out`.
   - Repeat for LF branch.
10. Final Heads:
    - Upsample to 512×512 → produce `Mask_HF` and `Mask_LF` (512×512×1) → fuse → `P_mask` (512×512×1).

---

# Training recipe (practical)
1. **Stage 0 — BiomedCLIP fine-tune (if not already fine-tuned)**  
   - Use DHN-NCE with settings from MedCLIP (τ=0.6; β1=β2=0.15; LR=1e-6 for CLIP fine-tune). 

2. **Stage 1 — FMISeg base training (supervised or with pseudo-labels)**
   - If you have ground truth (GT) text→mask pairs (e.g., QaTa-COV19), train FMISeg end-to-end with:
     - Loss = λ1 * DiceLoss(P_mask, GT) + λ2 * CE(P_mask, GT) (paper uses Dice + CrossEntropy). Start λ1=1.0, λ2=1.0. 
   - Optimizer: AdamW, initial LR e.g. 1e-4 decayed with cosine schedule; batch size per GPU 8–16 depending on memory. Use mixed precision.

3. **Stage 2 — Warm start with pseudo-labels (if GT limited)**
   - Use MedCLIP M2IB → (optionally) SAM or existing FMISeg to create pseudo masks.
   - Train FMISeg on mixture of GT and pseudo-labels; upweight GT samples.

4. **Stage 3 — Distillation (optional but recommended)**
   - If you can still run SAM as teacher (even if your final system won’t use it), create teacher soft probability maps `P_teacher` and minimize KL(P_teacher || P_student) (or BCE/soft Dice). This transfers SAM’s generalization to FMISeg. Use loss weight λ_distill (0.1–1.0). 

5. **Stage 4 — Ensemble & nnUNet refinement (optional)**
   - Generate pseudo-masks with trained FMISeg, train nnUNet on these masks for additional smoothing and uncertainty estimation (checkpoint ensembling as in MedCLIP). 

6. **Metrics**
   - Monitor Dice, mIoU, NSD where applicable. Compare to SAM baseline on same prompts/datasets. 

---

# Inference modes (three variants)
1. **End-to-end Text→Mask (no SAM)**  
   - Input `I`, `T` → BiomedCLIP → M2IB `S` → FMISeg → `P_mask`. (Fast, no external prompt engine). This is the pure replacement case. 

2. **M2IB prior + FMISeg (recommended)**  
   - Same as above but explicitly uses `S` as concatenated prior across decoder/encoder skip connections. Improves localization. 

3. **Hybrid (teacher SAM kept for bootstrapping)**  
   - During deployment you can optionally keep SAM as a fallback for difficult cases or for human-in-the-loop prompting. During training use SAM as teacher to distill generalization into FMISeg. This preserves SAM’s zero-shot advantage while migrating to FMISeg. 

---

# Mandatory engineering items you cannot skip
- **TextAdapter**: map BiomedCLIP text features → decoder token features expected by LFFI. Without this the decoder won’t align. (Linear proj + LayerNorm minimum.)   
- **Saliency prior injection**: if you want to reuse MedCLIP’s M2IB investment, incorporate `S` into FMISeg early (encoder) or as attention gating — do **one** of these.   
- **FFBI & LFFI implementations**: replicate FMISeg cross-attention blocks exactly (bi-directional cross-attention + filter design) — these are central to FMISeg’s performance. 

---

# Failure modes & mitigations (be blunt)
- **If you remove SAM entirely and have almost no ground truth**: FMISeg will struggle to generalize zero-shot. Mitigate by using SAM as teacher to distill, or run iterative pseudo-label + nnUNet cycles as MedCLIP did.   
- **Text encoder mismatch**: If you feed BiomedCLIP pooled vectors directly into LFFI, performance will collapse. **Do not** skip the TextAdapter.   
- **Compute**: FMISeg dual branches + cross-attention are heavier than a single branch; reduce backbone or use gradient accumulation. 

---

# Minimal implementable plan (day-by-day checklist)
1. Implement data preprocessing (wavelet HF/LF, text tokenization, M2IB saliency generation).   
2. Clone FMISeg repo; implement `TextAdapter` to accept BiomedCLIP tokens.   
3. Wire BiomedCLIP outputs into pipeline; test forward pass with random data (check shapes).   
4. Train FMISeg on one pulmonary dataset (e.g., QaTa-COV19) to reproduce baseline numbers.   
5. Generate pseudo-labels using M2IB→FMISeg and optionally SAM; iterate with nnUNet refinement. 

---