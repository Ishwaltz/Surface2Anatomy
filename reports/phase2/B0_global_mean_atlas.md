# B0: Global Mean Atlas Baseline Report

## 1. Description
The Global Mean Atlas computes the empirical centroid for each anatomical target across all valid training cases ($N=352$) in the patient-centered metric coordinate frame, and predicts this constant coordinate for every validation patient.

## 2. Validation Metrics (Primary 107 Targets)
- **Macro-Target MRE**: **54.71 mm**
- **Micro MRE**: **50.14 mm**
- **Macro-Patient MRE**: **48.68 mm**
- **Median Radial Error**: **36.74 mm**
- **P75 / P90 Error**: 55.66 mm / 92.84 mm
- **SDR@10 / SDR@20**: 3.38% / 17.52%
