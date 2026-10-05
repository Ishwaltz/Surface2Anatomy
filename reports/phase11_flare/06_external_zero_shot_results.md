# FLARE22 Zero-Shot External Evaluation Results

## 1. Primary Benchmark Results
- **Ensemble Macro MRE (13 Overlapping Targets):** **60.14 mm**
- **95% Patient Bootstrap CI:** **[57.63 mm, 62.70 mm]** (5,000 resamples)
- **Ensemble Micro MRE:** **60.14 mm**
- **Median Error:** **59.17 mm**
- **P75 / P90 / P95 Error:** 68.00 mm / 76.32 mm / 81.65 mm

## 2. Success Detection Rates (SDR)
- **SDR @ 5 mm:** 0.0%
- **SDR @ 10 mm:** 0.0%
- **SDR @ 15 mm:** 0.0%
- **SDR @ 20 mm:** 0.0%
- **SDR @ 25 mm:** 0.2%
- **SDR @ 30 mm:** 0.2%

## 3. Seed-Level Stability
- **Seed 42 Macro MRE:** 62.64 mm
- **Seed 43 Macro MRE:** 61.17 mm
- **Seed 44 Macro MRE:** 58.78 mm
- **Seed Mean ± SD:** **60.86 ± 1.59 mm**

## 4. Sanity Controls
- **Training Population Atlas Baseline:** 64.78 mm
- **Patient Surface Shuffle Control:** 68.24 mm
- **Target Query Permutation Control:** 126.07 mm

## 5. Target-Wise Breakdown (Sorted Best to Worst)
| Target Name | Support ($N$) | Mean Radial Error (mm) | Median (mm) | P90 (mm) | SDR@20 (%) | SDR@30 (%) |
|---|---|---|---|---|---|---|
| **esophagus** | 50 | **52.41** | 52.87 | 63.10 | 0.0% | 0.0% |
| **stomach** | 50 | **54.77** | 52.80 | 71.62 | 0.0% | 0.0% |
| **pancreas** | 50 | **57.43** | 57.51 | 72.31 | 0.0% | 0.0% |
| **inferior_vena_cava** | 50 | **60.02** | 60.12 | 72.52 | 0.0% | 0.0% |
| **aorta** | 50 | **60.18** | 59.26 | 72.59 | 0.0% | 0.0% |
| **kidney_left** | 50 | **60.24** | 59.50 | 77.12 | 0.0% | 0.0% |
| **kidney_right** | 50 | **61.45** | 62.17 | 79.78 | 0.0% | 0.0% |
| **duodenum** | 50 | **61.54** | 60.11 | 75.74 | 0.0% | 0.0% |
| **liver** | 50 | **61.66** | 58.97 | 76.04 | 0.0% | 0.0% |
| **gallbladder** | 50 | **62.14** | 61.68 | 81.30 | 0.0% | 0.0% |
| **spleen** | 50 | **62.24** | 61.24 | 83.31 | 0.0% | 2.0% |
| **adrenal_gland_left** | 50 | **63.49** | 64.47 | 78.26 | 0.0% | 0.0% |
| **adrenal_gland_right** | 50 | **64.24** | 63.78 | 76.50 | 0.0% | 0.0% |
