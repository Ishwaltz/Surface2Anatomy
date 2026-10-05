# Phase 1R Reconstruction Provenance & Environment Snapshot

## 1. Environment Specifications
- **Operating System**: Linux 7.0.0-29-generic x86_64
- **Python**: 3.11.15 (conda-forge, GCC 14.4.0)
- **PyTorch**: 2.13.0+cu130 (CUDA Available: True)
- **NumPy**: 2.4.6
- **SciPy**: 1.17.1
- **Nibabel**: 5.4.2

## 2. Version Control & Workspace Integrity
- **Repository Root**: `/home/sharon/Desktop/3D-Organ-Location-Prediction-Model`
- **VCS Status**: Repository does not contain a `.git` metadata directory.
- **Rule 1 Compliance**: 
  - The old dataset (`sharon/dataset/pointclouds_450.pt`) and splits (`sharon/outputs/splits_pointcloud.json`) remain strictly untouched.
  - All new reconstruction artifacts and pipelines will reside cleanly in:
    - Code: `tools/reconstruct/` and `sharon/data_v2/`
    - Data: `sharon/dataset_v2/`
    - Reports & Figures: `reports/phase1r/`
