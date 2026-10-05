# Phase 10R: Input Point Resolution & Normals Ablation

All models trained under matched 65-epoch protocol.

| Configuration | Input Representation | Macro MRE (mm) | Micro MRE (mm) | Median (mm) |
|---|---|---|---|---|
| 1024 Points | 1024 x 3 XYZ | 25.15 | 25.12 | 19.47 |
| 2048 Points | 2048 x 3 XYZ | 24.33 | 24.27 | 18.88 |
| **4096 Points (Canonical)** | **4096 x 3 XYZ** | **25.09** | **25.00** | **19.47** |
| 8192 Points | 8192 x 3 XYZ | 24.37 | 24.29 | 18.32 |
| 4096 Points + Normals | 4096 x 6 (XYZ + Normals) | 25.46 | 25.33 | 19.73 |
