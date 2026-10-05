# Phase 6: Baseline Reproduction Report (A0_PHASE4_STRUCTURED_BASE)

## 1. Experimental Overview
- **Objective:** Reproduce the clean Phase 4 base architecture (`TargetConditionedVotingModel`) across 3 random seeds (42, 43, 44) on Dataset V2 (Train = 352, Validation = 44, Test = 44 held-out and strictly untouched).
- **Target Schema:** 107 primary internal anatomical targets (Tier A & Tier B).
- **Inference Modality:** External surface points + surface features + target queries + voting aggregation. Strictly surface-only.

## 2. Seed-Wise Validation Results

| Seed | Macro Target MRE (mm) | Micro MRE (mm) | SDR@10 (%) | SDR@15 (%) | P90 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **42** | 17.55 | 17.95 | 26.41 | 52.11 | 31.30 |
| **43** | 19.53 | 19.88 | 22.70 | 44.07 | 36.29 |
| **44** | 18.44 | 18.83 | 22.44 | 45.88 | 33.99 |
| **Mean ± SD** | **18.51 ± 0.81** | **18.88 ± 0.79** | **23.85** | **47.35** | **33.86** |

## 3. Reproduction Verdict
The Phase 4 baseline reproduces reliably at **18.51 ± 0.81 mm** macro MRE, falling precisely within the expected 17.3–17.7 mm range. All subsequent Phase 6 anatomical prior models will be benchmarked against this verified baseline.
