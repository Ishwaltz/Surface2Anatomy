# Input Representation and Geometry Ablation Report

## 1. Surface Point Density Scaling (E1)

| Point Count (N) | Density Description | Macro MRE (mm) | Relative vs 4096 (mm) |
| :---: | :--- | :---: | :---: |
| 1024 | Sparse surface point cloud | 33.39 | +10.03 |
| 2048 | Medium surface point cloud | 33.72 | +10.36 |
| 4096 | Standard surface point cloud (Frozen Baseline) | **23.36** | 0.00 |
| 8192 | Dense high-resolution surface point cloud | 28.62 | +5.26 |

## 2. Input Feature Channels (E2: XYZ vs XYZ + Estimated Normals)

| Feature Modality | Channels | Macro MRE (mm) | Observation |
| :--- | :---: | :---: | :--- |
| XYZ Only | 3 | **23.36** | Pure 3D coordinates captured directly by surface scans. |
| XYZ + Estimated Normals | 6 | 29.62 | Normals provide marginal guidance since PointNet++ SA layers already implicitly model local tangent planes. |
