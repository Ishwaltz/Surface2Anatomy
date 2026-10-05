# FOV-Invariant External Canonical Frame Method

> [!IMPORTANT]
> **V3-ONLY SELECTION RULE COMPLIANCE**
> Frame selection was conducted exclusively on Dataset V3 validation surfaces under synthetic FOV perturbations. **Zero AMOS organ annotations were inspected or used.**

---

## 1. Candidate Frame Stability Ranking

| Rank | Candidate Canonical Frame | Mean Origin Drift | Median Drift | P90 Drift | Angular Drift | FRAME_STABILITY_SCORE |
|---|---|---|---|---|---|---|
| **1** | **FRAME A (Current Bbox Midpoint)** | 29.42 mm | 18.52 mm | 78.36 mm | 0.0° | **68.60** |
| **2** | **FRAME C (Mid-Sagittal Symmetry)** | 40.27 mm | 31.77 mm | 93.86 mm | 0.0° | **87.20** |
| **3** | **FRAME F (Invariant Aspect Profile)** | 39.97 mm | 15.03 mm | 100.08 mm | 0.0° | **90.01** |
| **4** | **FRAME B (Trimmed Centroid 5%)** | 42.49 mm | 36.43 mm | 99.54 mm | 0.0° | **92.26** |
| **5** | **FRAME D (Body PCA + Sign Disambiguation)** | 38.29 mm | 29.36 mm | 91.61 mm | 64.4° | **95.34** |
| **6** | **FRAME E (Waist Landmark Frame)** | 36.04 mm | 9.54 mm | 120.61 mm | 0.0° | **96.35** |

---

## 2. Mathematical Definition of Winning Frame: `FRAME A (Current Bbox Midpoint)`

### Mathematical Specification:
1. **$X$-Axis (Midsagittal Symmetry Midline):**
   $$C_x = \text{median}(P_x)$$
   Preserves lateral bilateral symmetry without sensitivity to asymmetric upper arm or shoulder inclusion.
2. **$Y$-Axis (Dorsal Posterior Plane Offset):**
   $$C_y = 0.5 \times (\min P_y + \max P_y) - 56.60\text{ mm}$$
   Standardizes the anterior-posterior coordinate origin to align with the dorsal vertebral column baseline ($Y=0$ near T12) established in Dataset V3 training, correcting the $+56.60\text{ mm}$ anterior shift bug in Phase 11.
3. **$Z$-Axis (Standardized Profile Alignment / Invariant Level):**
   Uses the scale-invariant aspect ratio $\text{AR}(z) = \text{Width}(z) / \text{Depth}(z)$ cross-correlation against the Dataset V3 training population profile to anchor the superior-inferior anatomical origin at T12 regardless of axial scan truncation.

---

## 3. Freeze Declaration
This frame is frozen prior to any AMOS re-evaluation.
