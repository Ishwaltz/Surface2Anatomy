# Phase 10R: Fair Baseline Benchmarking Suite

All neural models trained under matched 65-epoch optimization protocol, AdamW, CosineAnnealingLR, batch size 16, and 0.5 mm train jitter.

| Model ID | Method | Macro MRE (mm) | Micro MRE (mm) | Median (mm) | P90 (mm) | SDR@10 (%) | SDR@20 (%) |
|---|---|---|---|---|---|---|---|
| C0 | Population Atlas | 65.97 | 64.57 | 50.12 | 115.10 | 2.5% | 12.2% |
| C1 | Linear Ridge Regressor | 63.74 | 61.91 | 49.98 | 115.22 | 1.9% | 10.2% |
| C5 | Internal Statistical Shape Model (SSM/PCA) | 52.56 | 50.73 | 39.86 | 91.07 | 3.6% | 17.3% |
| C2 | PointNet++ Direct Regressor (3-seed mean) | 41.40 ± 0.60 | 40.47 | 27.87 | 65.04 | 7.9% | 31.3% |
| C3 | PointNet++ + DGCNN Target Decoder (3-seed mean) | 43.48 ± 0.83 | 42.72 | 29.18 | 71.45 | 6.6% | 28.1% |
| C4 | **Proposed TargetQuery Transformer (3-seed mean)** | **24.69 ± 0.44** | **24.55** | **18.42** | **40.65** | **15.1%** | **52.0%** |
| C4-Ens | **Proposed 3-Model Prediction Ensemble** | **23.39** | **23.27** | **18.42** | **40.65** | **17.0%** | **55.8%** |
