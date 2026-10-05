# Table 5: Input Point Resolution and Surface Normal Ablation (Matched 65 Epochs)

| Input Points | Coordinates & Features | Macro MRE (mm) | Micro MRE (mm) | Median (mm) | P90 (mm) |
|---|---|---|---|---|---|
| 1,024 Points | 3D Coordinates (XYZ) | 26.95 | 26.50 | 21.80 | 51.40 |
| 2,048 Points | 3D Coordinates (XYZ) | 25.40 | 25.01 | 20.25 | 48.20 |
| **4,096 Points (Canonical)** | **3D Coordinates (XYZ)** | **24.39** | **24.35** | **19.41** | **46.12** |
| 8,192 Points | 3D Coordinates (XYZ) | 24.28 | 24.18 | 19.30 | 45.90 |
| 4,096 Points + Normals | 6D Coordinates + Normals (XYZ+N) | 24.35 | 24.25 | 19.35 | 46.05 |

- **Finding:** 4,096 points strikes the optimal trade-off between computational latency (88.7 ms) and spatial fidelity. Scaling to 8,192 points yields negligible gain (-0.11 mm, p=0.45), while adding surface normals provides no significant advantage (-0.04 mm, p=0.72) over purely geometric XYZ coordinates.
