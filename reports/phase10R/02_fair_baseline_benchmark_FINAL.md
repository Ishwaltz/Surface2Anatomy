# Phase 10R: Fair Neural Baseline Benchmark (Matched 65-Epoch Budget)

All neural models trained under identical 65-epoch budget, batch size 16, AdamW, CosineAnnealingLR.

| Architecture / Model | Seed / Config | Macro MRE (mm) | Micro MRE (mm) | Median (mm) | P90 (mm) | SDR@10 (%) | SDR@20 (%) |
|---|---|---|---|---|---|---|---|
| **C2 PointNet++ Direct** | 42 | **41.18** | 39.97 | 30.41 | 65.73 | 6.3% | 28.0% |
| **C2 PointNet++ Direct** | 43 | **40.79** | 40.04 | 29.09 | 69.29 | 7.9% | 30.3% |
| **C2 PointNet++ Direct** | 44 | **42.22** | 41.39 | 30.14 | 73.75 | 6.9% | 27.7% |
| **C2 PointNet++ Direct** | Seed Mean ± SD | **41.40 ± 0.60** | — | — | — | — | — |
| **C2 PointNet++ Direct** | Ensemble | **39.31** | 38.42 | 27.87 | 65.04 | 7.9% | 31.3% |
| **C3 DGCNN Decoder** | 42 | **44.37** | 43.67 | 31.88 | 75.77 | 5.9% | 25.5% |
| **C3 DGCNN Decoder** | 43 | **42.38** | 41.78 | 31.38 | 71.40 | 5.6% | 25.4% |
| **C3 DGCNN Decoder** | 44 | **43.71** | 42.71 | 30.42 | 75.45 | 6.0% | 25.7% |
| **C3 DGCNN Decoder** | Seed Mean ± SD | **43.48 ± 0.83** | — | — | — | — | — |
| **C3 DGCNN Decoder** | Ensemble | **41.24** | 40.43 | 29.18 | 71.45 | 6.6% | 28.1% |
| **C4 Proposed TargetQuery** | 42 | **25.09** | 25.00 | 19.47 | 43.61 | 15.2% | 51.7% |
| **C4 Proposed TargetQuery** | 43 | **24.08** | 23.91 | 19.12 | 42.11 | 15.5% | 52.9% |
| **C4 Proposed TargetQuery** | 44 | **24.91** | 24.74 | 19.60 | 44.02 | 14.5% | 51.4% |
| **C4 Proposed TargetQuery** | Seed Mean ± SD | **24.69 ± 0.44** | 24.55 ± 0.46 | — | 43.24 ± 0.82 | 15.1 ± 0.4% | 52.0 ± 0.7% |
| **C4 Proposed TargetQuery** | Ensemble | **23.39** | 23.27 | 18.42 | 40.65 | 17.0% | 55.8% |
