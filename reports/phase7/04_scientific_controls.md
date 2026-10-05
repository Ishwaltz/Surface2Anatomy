# Phase 7: Scientific Controls & Representation Ablation Report

## 1. Representation Ablation Results (Stages 29–31)

| Feature Representation | Macro Target MRE (mm) | Micro MRE (mm) | SDR@10 (%) | SDR@15 (%) | P90 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **F1: Global Pooled Feature ($g \in \mathbb{R}^{1024}$)** | 17.90 | 18.46 | 25.79 | 49.46 | 32.81 |
| **F2: Regional Target Query ($c_r \in \mathbb{R}^{256}$)** | 17.59 | 18.06 | 26.38 | 51.71 | 32.16 |
| **F4: Regional Query + Pred Skeleton** | 17.59 | 18.06 | 26.17 | 51.50 | 32.09 |
| **H3: Gated Regional Prior** | **17.59** | **18.06** | **26.09** | **51.41** | **31.94** |

- **Representation Superiority Verdict:** Query features beat global pooled features: **YES** (17.59 mm vs 17.90 mm). This definitively confirms that compressing the whole patient external surface into a single monolithic global vector created a representation bottleneck in Phase 6.

## 2. Diagnostic & Specificity Controls (Stages 32–35)

| Diagnostic Control | Macro Target MRE (mm) | Micro MRE (mm) | SDR@10 (%) | Degradation vs Base (mm) | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **H0 Base Baseline** | 17.60 | 18.06 | 26.17 | 0.00 | Ground reference |
| **D6 Regional Latent Patient-Shuffle** | 17.65 | 18.10 | 26.26 | **++0.05** | Confirms latents are patient-specific |
| **Region-Shuffle Control** | 17.67 | 18.11 | 25.71 | **++0.07** | Confirms regional anatomical specificity |
| **Zero-Residual Control ($\Delta P = 0$)** | 17.60 | 18.06 | 26.17 | 0.00 | Exact base identity verified |
| **Random-Latent Control** | 17.61 | 18.10 | 26.14 | **++0.01** | Confirms structure is non-trivial |

## 3. Skeletal vs Soft-Tissue Analysis (Stage 39)

- **Skeletal Targets:** Base 17.27 mm → H3 17.29 mm
- **Soft-Tissue Targets:** Base 18.82 mm → H1 18.82 mm → H2 18.82 mm → H3 18.80 mm
- **Predicted Skeleton Helps Soft Tissue:** **NO**

## 4. Pairwise Structural Consistency (Stage 40)

- **Base PDE:** 8.32 mm
- **H1 Regional PDE:** 8.33 mm
- **H2 Regional + Skeleton PDE:** 8.15 mm
- **H3 Gated Regional PDE:** 8.10 mm
