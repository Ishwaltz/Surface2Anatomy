# Forensic Mapping of the Complete Data, Geometry & Coordinate Pipeline

## 1. Complete End-to-End Pipeline Map

```text
Raw Patient DICOM / NIfTI Files (raw_data/*.nii.gz)
        ↓  [SimpleITK / TotalSegmentator API: dataset_preprocessing.py]
Preprocessed CT (ct.nii.gz) & Multiclass Segmentation (segmentation.nii.gz)
        ↓  [Geometric Ellipsoidal Insertion: _extract_female_pelvic_anatomy()]
121-Class Organ Mask Volume (segmentation.nii.gz: shape (Nx, Ny, Nz))
        ↓  [scipy.ndimage.center_of_mass: pointcloud_sampler.py]
Voxel Centers of Mass com = (i, j, k)
        ↓  [CRITICAL BUG: cz, cy, cx = com -> affine @ [cx, cy, cz, 1.0]]
Corrupted Physical Target Centroids (raw_centroids: X & Z swapped, Z ~ 2.5m)
        ↓  [Bounding-Box Rescaling of Static Template: load_canonical_surface_points()]
Synthetic Torso Surface Points (raw_points: shape (4096, 3), stretched to warped bbox)
        ↓  [Anisotropic Bounding-Box Normalization: [-1, 1]^3]
Normalized Tensors: points (B, 4096, 3), centroids (B, 121, 3), scales (B, 3), centers (B, 3)
        ↓  [torch.save: pointcloud_sampler.py]
Consolidated Binary Dataset (sharon/dataset/pointclouds_450.pt)
        ↓  [Dataset Loader + Point Jitter (0.005): dataset.py: PointCloudOrganDataset]
Batch Tensors: pts, ctr, sex, mask, center, scale
        ↓  [Neural Network Forward Pass: model_evidential.py / model_same.py / model_gnn.py]
Predicted Normalized Coordinates (gamma / full_coords in [-1, 1]^3)
        ↓  [Physical De-normalization: preds_mm = preds * scale + center]
Predicted Physical Coordinates (in warped mm coordinate space)
        ↓  [Euclidean Distance: torch.norm(preds_mm - ctr_mm, dim=-1)]
Physical Mean Radial Error Metric (MRE in warped mm)
```

## 2. Stage-by-Stage Forensic Breakdown

### Stage 1: Raw CT & Segmentation Ingestion
- **File Path**: `sharon/dataset_preprocessing.py`
- **Function / Class**: `DatasetPreprocessor.process_case()`
- **Inputs**: Raw CT DICOM directories or NIfTI files in raw_data/
- **Outputs**: sharon/dataset/case_XXX/ct.nii.gz, segmentation.nii.gz
- **Coordinate Units**: Hounsfield Units (HU) for CT; Integer class IDs (1..117) for segmentation
- **Coordinate Frame**: LAS (Left, Anterior, Superior) from NIfTI affine
- **Tensor Shape**: `(768, 768, 90) or (512, 512, 90..132)`
- **Transformation Applied**: SimpleITK DICOM series reader -> NIfTI-1 file export

### Stage 2: Female Pelvic Organ Augmentation (K=121)
- **File Path**: `sharon/dataset_preprocessing.py`
- **Function / Class**: `DatasetPreprocessor._extract_female_pelvic_anatomy()`
- **Inputs**: Base 117-class segmentation.nii.gz, patient sex prior
- **Outputs**: 121-class segmentation with labels 118 (uterus), 119 (ovary_l), 120 (ovary_r), 121 (vagina)
- **Coordinate Units**: Voxel indices / binary spatial ellipsoidal masks
- **Coordinate Frame**: Voxel grid space (D, H, W) assumed to be Z, Y, X (actually X, Y, Z in nibabel)
- **Tensor Shape**: `(Nx, Ny, Nz)`
- **Transformation Applied**: Synthetic geometric ellipsoid rasterization relative to bladder (21) and sacrum (23)

### Stage 3: Target Centroid Extraction & Physical Conversion
- **File Path**: `sharon/pointcloud_sampler.py`
- **Function / Class**: `_worker_normalized_case()`
- **Inputs**: segmentation.nii.gz, NIfTI affine matrix
- **Outputs**: raw_centroids: float32 array (121, 3)
- **Coordinate Units**: Millimeters (mm), but corrupted by axis swap
- **Coordinate Frame**: CORRUPTED: X and Z axes swapped before affine multiplication
- **Tensor Shape**: `(121, 3)`
- **Transformation Applied**: CRITICAL BUG: com = ndi.center_of_mass(seg); cz, cy, cx = com; phys = (affine @ [cx, cy, cz, 1])[:3]. In nibabel, com is (x, y, z). Setting cz=x, cx=z swapped in-plane X and slice Z.

### Stage 4: Skin Surface Generation
- **File Path**: `sharon/pointcloud_sampler.py`
- **Function / Class**: `load_canonical_surface_points() & _worker_normalized_case()`
- **Inputs**: Canonical template mesh (sharon/outputs/meshes/skin.obj) or random ellipsoid
- **Outputs**: raw_points: float32 array (4096, 3)
- **Coordinate Units**: Millimeters (mm), stretched to match warped target bounds
- **Coordinate Frame**: Artificial: Fitted to min/max of corrupted target coordinates
- **Tensor Shape**: `(4096, 3)`
- **Transformation Applied**: raw_points = (base_skin - ref_center)/ref_scale * scale_ras + center_ras + N(0, 0.5)

### Stage 5: Spatial Normalization
- **File Path**: `sharon/pointcloud_sampler.py`
- **Function / Class**: `_worker_normalized_case()`
- **Inputs**: raw_points (4096, 3), raw_centroids (121, 3)
- **Outputs**: norm_points (4096, 3) in [-1, 1], norm_centroids (121, 3) in [-1, 1], center_c (3,), scale_s (3,)
- **Coordinate Units**: Dimensionless normalized coordinates
- **Coordinate Frame**: Unit bounding box [-1, 1]^3 centered at torso midpoint
- **Tensor Shape**: `points: (4096, 3), centroids: (121, 3), center: (3,), scale: (3,)`
- **Transformation Applied**: c = (min + max)/2; s = (max - min)/2; norm = (raw - c) / s

### Stage 6: Batch Loading & Training Augmentation
- **File Path**: `sharon/dataset.py`
- **Function / Class**: `PointCloudOrganDataset.__getitem__()`
- **Inputs**: sharon/dataset/pointclouds_450.pt
- **Outputs**: Dict: points, centroids, sex_prior, mask, center, scale
- **Coordinate Units**: Dimensionless for points/centroids; mm for center/scale
- **Coordinate Frame**: Unit bounding box [-1, 1]^3
- **Tensor Shape**: `points: (4096, 3), centroids: (121, 3)`
- **Transformation Applied**: Gaussian coordinate jitter: pts = clamp(pts + N(0, 0.005), -1.2, 1.2)

### Stage 7: Neural Inference & Physical De-normalization
- **File Path**: `sharon/evaluate_evidential.py (and evaluate_same.py / evaluate_gnn.py)`
- **Function / Class**: `evaluate_evidential_test_set()`
- **Inputs**: Model normalized predictions gamma in [-1, 1]^3, patient center, patient scale
- **Outputs**: preds_mm, ctr_mm, dist_mm
- **Coordinate Units**: Millimeters (in warped coordinate system)
- **Coordinate Frame**: De-normalized space: preds_mm = gamma * scale + center
- **Tensor Shape**: `preds_mm: (B, 121, 3), dist_mm: (B, 121)`
- **Transformation Applied**: Linear affine reversal: preds * scale + center; dist_mm = norm(preds_mm - ctr_mm, dim=-1)

