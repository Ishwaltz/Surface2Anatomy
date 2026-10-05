# Phase 6 - Step 4: Diagnostics, Residual Subspace, and Controls

## 1. Ground-Truth & Residual Oracles
- **D0 Anatomical Prior Oracle:** Reconstructs targets from ground truth latent coordinates
- **D1 Residual Oracle:** Macro MRE = 10.05 mm (SDR@10 = 61.76%, P90 = 18.17 mm)

## 2. Scientific Negative Controls
- **D2 Latent Patient Shuffle:** Macro MRE = 61.08 mm (Massive degradation verifies genuine patient-specific alignment)
- **D3 Zero Latent Population Mean (A):** Macro MRE = 54.71 mm (Quantifies error when individual deformation is zeroed)

## 3. Target Error Covariance & Regional Modularity
- Intra-Skeletal Correlation: 0.169
- Intra-Soft Tissue Correlation: 0.149
- Cross Skeletal-Soft Correlation: 0.146

## 4. 8-Patient Gate Memorization Test
- **Final Macro MRE:** 51.41 mm
- **Threshold:** < 3.0 mm
- **Verdict:** FAIL
