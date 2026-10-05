# Phase 10R: Final Locked Test Benchmark Evaluation

## 1. Compliance Statement & Protocol Integrity
- **Test Set Status:** The held-out test split (N=168 subjects) remained strictly locked throughout model development, tuning, and hyperparameter selection.
- **Freeze Confirmation:** Pre-test protocol frozen and sealed under SHA-256 `7193bcd4f56527493b075d14212783ffc75c0745b2549822f682ac339d935b2c`.
- **One-Time Evaluation:** This evaluation was executed exactly once with zero post-hoc tuning or re-selection.

## 2. Definitive Test Performance Summary

| Evaluation Configuration | Macro MRE (mm) | Micro MRE (mm) | Median (mm) | P90 (mm) | SDR@10 (%) | SDR@20 (%) |
|---|---|---|---|---|---|---|
| Seed 42 | 24.42 | 24.25 | 20.07 | 44.22 | 14.2% | 49.8% |
| Seed 43 | 25.39 | 25.09 | 20.04 | 46.21 | 13.6% | 49.9% |
| Seed 44 | 24.77 | 24.50 | 19.69 | 44.43 | 13.6% | 50.9% |
| **3-Seed Mean ± SD** | **24.86 ± 0.40** | **24.62** | — | — | **13.8%** | **50.2%** |
| **3-Model Prediction Ensemble** | **23.34** | **23.10** | **18.82** | **42.29** | **15.4%** | **54.2%** |

## 3. Statistical Confidence & Cohort Breakdown

- **95% Bootstrap Confidence Interval (Macro MRE):** **[22.40 mm, 26.15 mm]** (5,000 resamples)
- **V2 Test Cohort (N=41):** **17.66 mm** (Median: 15.66 mm, SDR@10: 22.4%)
- **TotalSegmentator Test Cohort (N=127):** **25.12 mm** (Median: 20.18 mm, SDR@10: 12.8%)
