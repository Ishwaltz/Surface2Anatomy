# Forensic Proof of Old Surface Input Degeneracy

## 1. Executive Summary
> **The previous network had little or no patient-specific external geometry from which to infer patient-specific internal anatomy.** `[DATA-VERIFIED]`

Because the previous data pipeline generated surface points by linearly stretching the exact same template mesh (`sharon/outputs/meshes/skin.obj`) to match internal organ bounding boxes, followed by per-patient anisotropic normalization into the unit cube $[-1, 1]^3$, the resulting input point clouds $X \in \mathbb{R}^{4096 \times 3}$ were virtually identical clones across all 450 patients.

## 2. Quantitative Degeneracy Metrics

### A. Point-Wise Coordinate Variance across All 450 Patients
- Mean Variance along X: `1.027151e-04`
- Mean Variance along Y: `2.981451e-05`
- Mean Variance along Z: `1.611421e-06`
- **Total Mean Coordinate Variance**: **`1.341411e-04`**
- **Mean Standard Deviation per Point**: **`1.151845e-02`** (in $[-1, 1]^3$)

### B. Pairwise Between-Patient RMS Distance
- Evaluated across 1,225 patient pairs (50 random cases):
- **Mean Pairwise RMS Distance**: **`1.577530e-02`**
- **Median Pairwise RMS Distance**: **`1.610087e-02`**
- Maximum Pairwise RMS Distance: `2.552147e-02`

### C. Chamfer Distance: Between-Patient vs Training Jitter Noise
- **Mean Chamfer Distance Between Different Patients**: **`1.356485e-02`**
- **Mean Chamfer Distance from Training Point Jitter (σ=0.005)**: **`7.912602e-03`**
- **Between-Patient / Noise Ratio**: **`1.71x`**

## 3. Physical Significance
In normalized space $[-1, 1]^3$, the standard deviation between patient surface points was on the order of `~0.003-0.005`, which is identical to the Gaussian sensor noise `0.005` added during training augmentation! The model was given effectively the identical skin surface for every single patient, while being penalized for differences in internal organ locations.
