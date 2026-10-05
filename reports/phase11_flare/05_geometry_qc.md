# FLARE22 External Validation: Geometry and Pipeline Quality Control

## 1. Compliance and Isolation Assertion
- **Zero Organ Mask Leakage:** Body surface point clouds were extracted strictly using CT voxel intensities (tissue foreground threshold HU > -500, 3D largest connected component, 2D axial slice hole filling). Organ masks were strictly isolated and used solely for ground-truth centroid computation.
- **Affine Coordinate Transformation:** Applied NIfTI affine matrix directly via `nibabel.affines.apply_affine`.
- **Maximum Voxel-World Roundtrip Error:** **0.000000 mm** (< 0.001 mm compliance bound: **PASS**).
- **Point Sampling Count:** Exactly **4096 points** per patient with uniform area-weighted surface sampling.
- **Normalization:** Normalized by $S_{\text{global}} = 500.0\text{ mm}$ around patient body bounding-box center.

## 2. Cohort Geometric Summary
- **Total Valid Evaluated Cases:** 50
- **Mean Extracted Surface Width (LR):** 382.3 mm
- **Mean Extracted Surface Depth (AP):** 291.2 mm
- **Mean Extracted Surface Height (SI):** 248.9 mm
