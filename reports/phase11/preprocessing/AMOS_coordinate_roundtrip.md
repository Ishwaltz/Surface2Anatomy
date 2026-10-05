# AMOS Coordinate Transformation & Roundtrip Verification Gate

## Status: PASSED (< 0.001 mm)

### Numerical Roundtrip Gate Results
- **Voxel $\to$ World $\to$ Model Norm $\to$ World Maximum Discrepancy:** `1.61e-13 mm`
- **Voxel $\to$ World $\to$ Voxel Maximum Discrepancy:** `4.65e-13 voxels`
- **Required Threshold:** `< 0.001 mm` (1.00e-03 mm)
- **Safety Margin:** 1000000000.0x below tolerance

### Pipeline Mathematical Guarantees
1. All voxel-to-world mapping operations utilize `nibabel.affines.apply_affine(affine, coords)` directly preserving oblique shear, rotation, and anisotropic spacing.
2. Body centering and scale normalization are purely affine Euclidean operations that preserve metric geometry without anisotropic distortion.
3. Zero spatial inversion or axis permutation errors exist in the pipeline.
