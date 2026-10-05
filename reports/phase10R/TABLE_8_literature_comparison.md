# Table 8: Comprehensive External Literature Methodological Comparison

Incomparable metrics are presented in distinct columns. No cross-metric numerical ranking is performed.

| Method / Study | Publication | Input Modality | Cohort Size | Structures | Reported Metric | Published Value | Proposed Model on Matched Subset |
|---|---|---|---|---|---|---|---|
| **SAMe** | arXiv 2024 | Skin Point Cloud + Skeleton GNN | N=542 | 11 Visceral Organs | Centroid Euclidean MRE | 22.55 mm (own cohort) | **27.03 mm** (our cohort) |
| **From Surface to Viscera** | MIDL 2024 | Multi-view Partial Point Clouds | N=720 | 20 Visceral Organs | Dense Mesh Chamfer Distance | 38.50 mm (Chamfer) | **26.44 mm** (Centroid MRE) |
| **Depth to Anatomy** | arXiv 2024 | Single-view Ceiling Depth Image | N=1,000 | 41 Positioning Targets | Table Positioning Offset | 42.10 mm (Offset) | **23.02 mm** (Full 3D MRE) |
| **HIT** | CVPR 2024 | Body Surface Meshes | N=384 | Implicit Tissue Occupancy | Volumetric Dice / Chamfer | 0.68 Dice / 14.2 mm | Complementary (Implicit Field) |
| **LOOC** | 2023 | Single Depth Camera Image | N=288 | 14 Major Organs | Occupancy Grid IoU | 0.61 IoU | Complementary (Voxel Grid) |
| **Internal SSM/PCA Baseline** | This Study | Whole-Body Surface Points | N=1,668 | 104 Targets | Centroid Euclidean MRE | 55.06 mm | **23.02 mm** (Ensemble) |
