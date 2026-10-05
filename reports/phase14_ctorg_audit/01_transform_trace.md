# Phase 14 CT-ORG Pipeline Forensic Transform Trace

## Coordinate Operations Trace: V3 vs CT-ORG Standard Torso vs CT-ORG Brain/Full-Body

### A. Representative Dataset V3 Cases (N=20)

#### Case: `v3_totalseg_s0016`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 142.5 mm, Width: 354.0 mm, Depth: 282.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-5.62, 154.38, 75.45] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0019`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 421.5 mm, Width: 329.4 mm, Depth: 292.5 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-0.57, 159.82, 379.00] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0029`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 349.5 mm, Width: 445.1 mm, Depth: 291.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-5.53, 160.44, 164.75] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0030`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 381.0 mm, Width: 461.4 mm, Depth: 339.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-17.92, 183.95, 295.80] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0045`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 220.5 mm, Width: 344.8 mm, Depth: 286.5 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [1.92, 156.84, 373.80] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0053`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 515.9 mm, Width: 491.3 mm, Depth: 300.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [1.32, 166.99, 376.10] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0080`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 346.3 mm, Width: 451.0 mm, Depth: 282.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-4.04, 163.45, 222.87] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0107`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 433.5 mm, Width: 432.7 mm, Depth: 248.4 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [9.81, 166.38, 264.75] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0111`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 359.2 mm, Width: 329.9 mm, Depth: 252.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-0.69, 144.57, 357.96] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0117`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 385.4 mm, Width: 391.1 mm, Depth: 241.1 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-2.27, 169.86, -25.12] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0122`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 122.2 mm, Width: 138.0 mm, Depth: 135.7 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-82.17, -1.99, -144.51] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0152`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 327.6 mm, Width: 347.7 mm, Depth: 321.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-0.31, 185.09, 251.64] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0162`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 282.0 mm, Width: 431.3 mm, Depth: 332.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-17.35, 183.88, -379.99] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0167`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 221.1 mm, Width: 263.6 mm, Depth: 259.2 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [0.44, 162.46, 91.79] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0204`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 258.0 mm, Width: 422.5 mm, Depth: 348.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [7.79, 193.79, 416.58] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0211`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 85.5 mm, Width: 327.0 mm, Depth: 285.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [0.38, 167.88, 401.46] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0234`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 191.2 mm, Width: 251.2 mm, Depth: 249.6 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-0.77, 167.78, 550.13] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0236`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 183.0 mm, Width: 369.6 mm, Depth: 289.5 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-0.45, 176.31, 473.54] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0247`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 129.7 mm, Width: 280.6 mm, Depth: 205.5 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [-7.42, 124.83, -717.99] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

#### Case: `v3_totalseg_s0248`
- Raw Voxel -> Original Affine -> Canonical RAS: Standardized V3 ingestion
- Body Surface: HU > -300, 3D connected component, axial hole filling, marching cubes (step=2)
- FOV Standardization: `compute_external_torso_bounds` (Torso height: 416.9 mm, Width: 420.0 mm, Depth: 225.0 mm)
- Surface Sampling: 4096 points uniformly sampled
- Centering vector `c_external`: [0.94, 131.44, 197.34] mm
- Normalization: / 500.0 mm -> Network Input
- Reconstruction: `pred_canonical = pred_norm * 500.0`, `pred_world = pred_canonical + c_external`

### B. Standard CT-ORG Torso Cases (N=20)

#### Case: `volume-0`
- Body Dimensions: W=358.6 mm, D=241.9 mm, H=370.0 mm
- BBox Midpoint `m_vec`: [-7.04, -16.16, -182.95] mm
- Canonical Ridge Offset: [-2.50, -47.63, 122.20] mm
- Centering Vector `c_external`: [-9.54, -63.79, -60.75] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-1`
- Body Dimensions: W=344.6 mm, D=247.3 mm, H=610.0 mm
- BBox Midpoint `m_vec`: [4.44, 3.38, -255.16] mm
- Canonical Ridge Offset: [-3.52, -43.80, 77.39] mm
- Centering Vector `c_external`: [0.92, -40.42, -177.77] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-2`
- Body Dimensions: W=393.1 mm, D=311.7 mm, H=516.0 mm
- BBox Midpoint `m_vec`: [2.54, 166.74, -855.87] mm
- Canonical Ridge Offset: [-0.66, -68.99, 175.03] mm
- Centering Vector `c_external`: [1.88, 97.75, -680.84] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-3`
- Body Dimensions: W=353.6 mm, D=300.9 mm, H=532.0 mm
- BBox Midpoint `m_vec`: [-0.39, 162.53, -793.57] mm
- Canonical Ridge Offset: [-1.06, -71.02, 211.33] mm
- Centering Vector `c_external`: [-1.45, 91.52, -582.25] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-4`
- Body Dimensions: W=453.9 mm, D=312.2 mm, H=672.0 mm
- BBox Midpoint `m_vec`: [2.96, -6.32, -359.28] mm
- Canonical Ridge Offset: [2.77, -55.27, 110.50] mm
- Centering Vector `c_external`: [5.73, -61.59, -248.78] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-5`
- Body Dimensions: W=478.6 mm, D=377.8 mm, H=428.8 mm
- BBox Midpoint `m_vec`: [-7.03, -115.41, -407.29] mm
- Canonical Ridge Offset: [-7.21, -82.48, 169.09] mm
- Centering Vector `c_external`: [-14.25, -197.88, -238.20] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-6`
- Body Dimensions: W=417.3 mm, D=293.3 mm, H=516.0 mm
- BBox Midpoint `m_vec`: [-6.55, 160.25, -1018.64] mm
- Canonical Ridge Offset: [6.31, -65.05, 174.08] mm
- Centering Vector `c_external`: [-0.23, 95.20, -844.56] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-7`
- Body Dimensions: W=381.5 mm, D=294.7 mm, H=540.0 mm
- BBox Midpoint `m_vec`: [-10.37, 160.86, -1199.66] mm
- Canonical Ridge Offset: [0.89, -63.88, 206.68] mm
- Centering Vector `c_external`: [-9.49, 96.99, -992.98] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-8`
- Body Dimensions: W=420.4 mm, D=286.5 mm, H=540.0 mm
- BBox Midpoint `m_vec`: [-7.24, 156.60, -326.53] mm
- Canonical Ridge Offset: [1.36, -57.93, 184.83] mm
- Centering Vector `c_external`: [-5.88, 98.67, -141.70] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-9`
- Body Dimensions: W=406.1 mm, D=283.8 mm, H=548.0 mm
- BBox Midpoint `m_vec`: [-7.30, 156.54, -4.11] mm
- Canonical Ridge Offset: [5.92, -59.23, 190.03] mm
- Centering Vector `c_external`: [-1.38, 97.30, 185.91] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-10`
- Body Dimensions: W=392.5 mm, D=286.3 mm, H=500.0 mm
- BBox Midpoint `m_vec`: [-5.40, 157.47, -336.48] mm
- Canonical Ridge Offset: [0.20, -58.47, 195.86] mm
- Centering Vector `c_external`: [-5.20, 99.00, -140.62] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-11`
- Body Dimensions: W=342.7 mm, D=306.4 mm, H=464.0 mm
- BBox Midpoint `m_vec`: [-10.36, 164.62, -667.05] mm
- Canonical Ridge Offset: [2.30, -68.77, 139.66] mm
- Centering Vector `c_external`: [-8.06, 95.85, -527.39] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-12`
- Body Dimensions: W=333.7 mm, D=282.7 mm, H=454.0 mm
- BBox Midpoint `m_vec`: [-2.28, 151.98, -241.49] mm
- Canonical Ridge Offset: [-2.53, -60.78, 139.71] mm
- Centering Vector `c_external`: [-4.81, 91.20, -101.78] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-13`
- Body Dimensions: W=408.6 mm, D=301.8 mm, H=604.0 mm
- BBox Midpoint `m_vec`: [-10.67, -148.34, 386.61] mm
- Canonical Ridge Offset: [1.25, -62.89, 32.29] mm
- Centering Vector `c_external`: [-9.42, -211.23, 418.90] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-14`
- Body Dimensions: W=349.6 mm, D=264.6 mm, H=586.0 mm
- BBox Midpoint `m_vec`: [5.92, -172.74, 278.54] mm
- Canonical Ridge Offset: [-2.55, -42.50, 112.10] mm
- Centering Vector `c_external`: [3.37, -215.25, 390.65] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-15`
- Body Dimensions: W=333.7 mm, D=260.4 mm, H=564.0 mm
- BBox Midpoint `m_vec`: [1.96, -170.12, 273.49] mm
- Canonical Ridge Offset: [-4.98, -46.06, 132.50] mm
- Centering Vector `c_external`: [-3.02, -216.18, 405.98] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-16`
- Body Dimensions: W=357.9 mm, D=253.1 mm, H=550.4 mm
- BBox Midpoint `m_vec`: [-0.28, -33.75, -273.83] mm
- Canonical Ridge Offset: [10.18, -41.48, -1.54] mm
- Centering Vector `c_external`: [9.90, -75.23, -275.38] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-17`
- Body Dimensions: W=380.5 mm, D=297.0 mm, H=457.6 mm
- BBox Midpoint `m_vec`: [0.59, -155.85, -259.48] mm
- Canonical Ridge Offset: [-3.40, -67.55, 152.48] mm
- Centering Vector `c_external`: [-2.81, -223.40, -107.00] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-18`
- Body Dimensions: W=378.4 mm, D=264.1 mm, H=436.0 mm
- BBox Midpoint `m_vec`: [-4.24, 144.43, -481.62] mm
- Canonical Ridge Offset: [0.44, -61.91, 132.10] mm
- Centering Vector `c_external`: [-3.81, 82.52, -349.52] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

#### Case: `volume-19`
- Body Dimensions: W=492.2 mm, D=349.6 mm, H=804.4 mm
- BBox Midpoint `m_vec`: [-1.17, -3.09, -575.77] mm
- Canonical Ridge Offset: [2.43, -56.04, 143.74] mm
- Centering Vector `c_external`: [1.26, -59.13, -432.02] mm
- Voxel-World Roundtrip Error: 0.00000000 mm
- Normalization: / 500.0 mm -> Model -> Reconstruction: `canonical_mm = norm * 500.0`

### C. CT-ORG Brain-Valid / Full-Body Cases (N=9)

#### Case: `volume-20` (Whole Body Scan)
- Body Dimensions: W=552.3 mm, D=429.3 mm, H=1716.8 mm (Massive Full-Body Scan)
- BBox Midpoint `m_vec`: [5.36, 4.53, 12.97] mm
- Canonical Ridge Offset: [-9.99, -26.97, 197.73] mm (Ridge extrapolating Z=+197.7 mm)
- Centering Vector `c_external`: [-4.63, -22.44, 210.70] mm
- Brain GT (Canonical): [9.48, 7.61, 775.39] mm
- Brain Predicted Seed42 Z: 833.81 mm | Seed43 Z: 790.41 mm | Seed44 Z: 192.27 mm (COLLAPSE/INVERSION) | Ensemble Z: 605.50 mm
- Brain Ensemble Error: 174.71 mm

#### Case: `volume-131` (Whole Body Scan)
- Body Dimensions: W=557.8 mm, D=336.3 mm, H=1715.0 mm (Massive Full-Body Scan)
- BBox Midpoint `m_vec`: [6.93, 40.46, 0.07] mm
- Canonical Ridge Offset: [-20.57, -5.82, 329.40] mm (Ridge extrapolating Z=+329.4 mm)
- Centering Vector `c_external`: [-13.64, 34.64, 329.47] mm
- Brain GT (Canonical): [14.84, -15.91, 704.68] mm
- Brain Predicted Seed42 Z: 926.24 mm | Seed43 Z: 1048.99 mm | Seed44 Z: 246.92 mm (COLLAPSE/INVERSION) | Ensemble Z: 740.71 mm
- Brain Ensemble Error: 122.49 mm

#### Case: `volume-132` (Whole Body Scan)
- Body Dimensions: W=498.1 mm, D=357.4 mm, H=974.5 mm (Massive Full-Body Scan)
- BBox Midpoint `m_vec`: [-249.71, -288.96, 487.12] mm
- Canonical Ridge Offset: [-3.29, -52.34, 303.35] mm (Ridge extrapolating Z=+303.4 mm)
- Centering Vector `c_external`: [-253.00, -341.31, 790.47] mm
- Brain GT (Canonical): [-246.92, -215.59, 949.99] mm
- Brain Predicted Seed42 Z: 1102.02 mm | Seed43 Z: 898.22 mm | Seed44 Z: 477.87 mm (COLLAPSE/INVERSION) | Ensemble Z: 826.04 mm
- Brain Ensemble Error: 133.55 mm

#### Case: `volume-133` (Whole Body Scan)
- Body Dimensions: W=429.3 mm, D=262.5 mm, H=974.5 mm (Massive Full-Body Scan)
- BBox Midpoint `m_vec`: [-7.52, -6.77, -0.06] mm
- Canonical Ridge Offset: [2.89, -31.64, 168.28] mm (Ridge extrapolating Z=+168.3 mm)
- Centering Vector `c_external`: [-4.63, -38.41, 168.22] mm
- Brain GT (Canonical): [-5.53, -24.39, 472.45] mm
- Brain Predicted Seed42 Z: 568.49 mm | Seed43 Z: 572.73 mm | Seed44 Z: 481.82 mm (COLLAPSE/INVERSION) | Ensemble Z: 541.01 mm
- Brain Ensemble Error: 83.02 mm

#### Case: `volume-135` (Whole Body Scan)
- Body Dimensions: W=607.0 mm, D=453.9 mm, H=1749.5 mm (Massive Full-Body Scan)
- BBox Midpoint `m_vec`: [4.26, -15.72, 6.58] mm
- Canonical Ridge Offset: [6.48, -13.83, 111.46] mm (Ridge extrapolating Z=+111.5 mm)
- Centering Vector `c_external`: [10.74, -29.56, 118.04] mm
- Brain GT (Canonical): [13.74, -1.89, 828.15] mm
- Brain Predicted Seed42 Z: 617.61 mm | Seed43 Z: 766.00 mm | Seed44 Z: 239.46 mm (COLLAPSE/INVERSION) | Ensemble Z: 541.02 mm
- Brain Ensemble Error: 289.75 mm

#### Case: `volume-136` (Whole Body Scan)
- Body Dimensions: W=577.0 mm, D=434.8 mm, H=1801.8 mm (Massive Full-Body Scan)
- BBox Midpoint `m_vec`: [-6.08, 5.11, 11.09] mm
- Canonical Ridge Offset: [-5.28, -21.99, 302.71] mm (Ridge extrapolating Z=+302.7 mm)
- Centering Vector `c_external`: [-11.35, -16.88, 313.80] mm
- Brain GT (Canonical): [9.45, 8.67, 802.16] mm
- Brain Predicted Seed42 Z: 900.12 mm | Seed43 Z: 922.46 mm | Seed44 Z: 265.47 mm (COLLAPSE/INVERSION) | Ensemble Z: 696.02 mm
- Brain Ensemble Error: 112.56 mm

#### Case: `volume-137` (Whole Body Scan)
- Body Dimensions: W=563.3 mm, D=352.7 mm, H=1765.0 mm (Massive Full-Body Scan)
- BBox Midpoint `m_vec`: [7.52, 18.43, 0.00] mm
- Canonical Ridge Offset: [-20.69, -13.01, 413.55] mm (Ridge extrapolating Z=+413.6 mm)
- Centering Vector `c_external`: [-13.17, 5.42, 413.55] mm
- Brain GT (Canonical): [-5.52, -33.33, 806.90] mm
- Brain Predicted Seed42 Z: 852.71 mm | Seed43 Z: 962.17 mm | Seed44 Z: 25.66 mm (COLLAPSE/INVERSION) | Ensemble Z: 613.51 mm
- Brain Ensemble Error: 216.97 mm

#### Case: `volume-138` (Whole Body Scan)
- Body Dimensions: W=650.8 mm, D=434.8 mm, H=1682.5 mm (Massive Full-Body Scan)
- BBox Midpoint `m_vec`: [0.96, 28.25, 16.84] mm
- Canonical Ridge Offset: [-1.29, -24.33, 318.11] mm (Ridge extrapolating Z=+318.1 mm)
- Centering Vector `c_external`: [-0.33, 3.92, 334.95] mm
- Brain GT (Canonical): [-1.09, 42.82, 796.25] mm
- Brain Predicted Seed42 Z: 717.24 mm | Seed43 Z: 877.82 mm | Seed44 Z: 189.30 mm (COLLAPSE/INVERSION) | Ensemble Z: 594.79 mm
- Brain Ensemble Error: 203.02 mm

#### Case: `volume-139` (Whole Body Scan)
- Body Dimensions: W=585.2 mm, D=440.2 mm, H=1798.5 mm (Massive Full-Body Scan)
- BBox Midpoint `m_vec`: [1.07, 9.39, 1.17] mm
- Canonical Ridge Offset: [7.59, -11.46, 399.84] mm (Ridge extrapolating Z=+399.8 mm)
- Centering Vector `c_external`: [8.66, -2.07, 401.01] mm
- Brain GT (Canonical): [17.47, 5.04, 816.26] mm
- Brain Predicted Seed42 Z: 660.21 mm | Seed43 Z: 797.08 mm | Seed44 Z: 229.44 mm (COLLAPSE/INVERSION) | Ensemble Z: 562.24 mm
- Brain Ensemble Error: 260.70 mm
