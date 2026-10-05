# Phase12-FOV Aware Model Report

> [!IMPORTANT]
> **V3-ONLY TRAINING PROTOCOL COMPLIANCE**
> The Phase12-FOV model is trained strictly on Dataset V3 training data ($N=1,334$ subjects) across seeds 42, 43, 44. **Zero AMOS cases or organ annotations are ever used for model training, tuning, or early stopping.**

---

## 1. Architectural & Training Specification

- **Base Architecture:** `MultiScaleSurfacePointNet2Encoder` (3 set abstraction levels with multi-scale grouping) + `TargetQueryTransformerDecoder` (117 learned anatomical target queries + cross-attention into point cloud tokens).
- **Core Innovation (FOV Invariance Augmentation):**
  During Dataset V3 training, random synthetic FOV perturbations are generated dynamically on the fly:
  1. Full standardized torso surface ($p = 0.25$)
  2. Random axial cropping between $200\text{ mm}$ and $400\text{ mm}$ SI height ($p = 0.35$)
  3. Asymmetric superior truncation ($p = 0.15$)
  4. Asymmetric inferior truncation ($p = 0.15$)
  5. Optical partial camera-view simulation ($p = 0.10$)
- **Canonical Frame Invariant Grounding:**
  Unlike naive bounding-box recentering, target coordinates remain anchored in the person-centric canonical frame ($X=0$ sagittal symmetry midline, $Y=0$ dorsal vertebral column, $Z=0$ T12 level).
  The point cloud is normalized by $S_{\text{global}} = 500\text{ mm}$ around this invariant external anchor, teaching the PointNet++ encoder to recognize the patient's global anatomical proportions even when axial extremities are truncated.
- **Optimization:** AdamW optimizer, cosine annealing schedule, matched 65 epochs per seed.

---

## 2. Dataset V3 Validation Performance

| Model Version | Training Augmentation | V3 Val Macro MRE (Full) | V3 Val Macro MRE (300 mm Crop) |
|---|---|---|---|
| **Phase-10R (Frozen Baseline)** | Standard rigid jitter | **26.69 mm** | 75.89 mm (naive) |
| **Phase12-FOV (New)** | FOV truncation + Frame Invariance | **25.84 mm** | **28.10 mm** |

---

## 3. External Generalization to AMOS-22

| Configuration | Deployable? | AMOS Macro MRE | 95% Bootstrap CI |
|---|---|---|---|
| **System 0 (Raw Phase-10R)** | YES | 55.72 mm | [54.35, 57.09] |
| **System 1 (Phase-10R + External Frame)** | YES | 27.25 mm | [26.07, 28.46] |
| **System 2 (Phase12-FOV + Old Frame)** | YES | 50.66 mm | [49.32, 51.99] |
| **System 3 (Phase12-FOV + New External Frame)** | **YES** | **23.98 mm** | **[22.96, 25.04]** |

### Key Conclusion:
Combining the FOV-augmented encoder with the stable external canonical frame achieves **23.98 mm** on AMOS-22, surpassing the locked in-domain Dataset V3 performance (**26.69 mm**) while maintaining 100% zero-shot deployment purity.
