# Baseline Benchmark Suite Evaluation Report

## 1. Primary Benchmark Results Across Models

| Model ID | Method Architecture | Macro MRE (mm) | Micro MRE (mm) | Median (mm) | P90 (mm) | SDR@5mm (%) | SDR@10mm (%) | SDR@15mm (%) | SDR@20mm (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0** | Population Mean Atlas | 65.97 | 64.57 | 50.12 | 115.10 | 0.46% | 2.49% | 6.74% | 12.17% |
| **C1** | Torso Dimension Ridge Regressor | 61.43 | 60.19 | 48.19 | 106.44 | 0.49% | 3.27% | 8.35% | 14.35% |
| **C5** | Statistical Shape Model (SSM / PCA) | 55.06 | 53.16 | 41.77 | 97.32 | 0.63% | 3.44% | 8.29% | 14.78% |
| **C2** | PointNet++ Direct Multi-Target Regressor | 44.41 | 44.38 | 37.89 | 78.42 | 1.12% | 8.41% | 19.20% | 30.85% |
| **C3** | Surface PointNet++ + DGCNN Target Decoder | 44.63 | 44.60 | 38.12 | 79.10 | 1.05% | 8.15% | 18.90% | 30.22% |
| **C4 (Seed 42)** | Proposed Geometric Attention ($S_{FULL}$) | 24.69 | 24.69 | 19.04 | 42.92 | 2.68% | 15.69% | 35.24% | 53.26% |
| **C4 (Seed 43)** | Proposed Geometric Attention ($S_{FULL}$) | 24.42 | 24.19 | 19.43 | 42.54 | 2.26% | 15.05% | 33.84% | 51.95% |
| **C4 (Seed 44)** | Proposed Geometric Attention ($S_{FULL}$) | 25.15 | 24.98 | 19.89 | 43.58 | 2.42% | 14.58% | 32.30% | 50.33% |
| **C4 (Ensemble)**| Proposed 3-Seed Ensemble Mean | **23.36** | **23.27** | **18.35** | **40.64** | **2.98%** | **17.15%** | **37.03%** | **55.94%** |

## 2. Subdomain Breakdown (V2 vs TotalSegmentator)

| Model ID | Description | Full Val Macro (mm) | V2 Val Macro (mm) | TotalSegmentator Val Macro (mm) |
| :--- | :--- | :---: | :---: | :---: |
| C0 | Population Atlas | 65.97 | 52.09 | 75.62 |
| C1 | Torso Ridge | 61.43 | 51.72 | 68.10 |
| C5 | Statistical Shape Model | 55.06 | 43.42 | 62.57 |
| C4 (Ensemble) | Proposed Model | **23.36** | **18.89** | **26.61** |
