# Table 2: Benchmark Model Comparisons on Validation Cohort (Matched 65 Epochs)

| ID | Model / Baseline | Paradigm | Macro MRE (mm) | Micro MRE (mm) | Median (mm) | P90 (mm) | SDR@10 (%) | SDR@20 (%) |
|---|---|---|---|---|---|---|---|---|
| C0 | Population Atlas | Zero-parameter Spatial Prior | 65.97 | 64.57 | 50.12 | 115.10 | 2.5% | 12.2% |
| C1 | Linear Ridge Regressor | Global Coordinate Mapping | 63.74 | 61.91 | 49.98 | 115.22 | 1.9% | 10.2% |
| C5 | Internal Statistical Shape Model (SSM/PCA) | Surface PCA (64 comps) + Ridge | 52.56 | 50.73 | 39.86 | 91.07 | 3.6% | 17.3% |
| C2 | PointNet++ Direct Regressor (3 seeds) | Global Point Cloud Multi-MLP | 41.40 ± 0.60 | 38.42 | 27.87 | 65.04 | 7.9% | 31.3% |
| C3 | PointNet++ + DGCNN Target Decoder (3 seeds) | Graph Reasoning over Queries | 43.48 ± 0.83 | 40.43 | 29.18 | 71.45 | 6.6% | 28.1% |
| C4 | **Proposed TargetQuery Model (3 seeds)** | **MultiScale Cross-Attention** | **24.69 ± 0.44** | **24.55** | **18.42** | **40.65** | **15.1%** | **52.0%** |
| C4-Ens | **Proposed 3-Model Prediction Ensemble** | **Ensemble Aggregation** | **23.39** | **23.27** | **18.42** | **40.65** | **17.0%** | **55.8%** |
