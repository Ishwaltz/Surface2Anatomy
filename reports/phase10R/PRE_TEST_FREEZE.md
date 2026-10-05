# Phase 10R: Pre-Test Protocol & Hypothesis Freeze Document

## 1. Executive Protocol Freeze Statement
This document seals the scientific, architectural, hyperparameter, and statistical specifications for the held-out test evaluation of:

**3D INTERNAL ORGAN / ANATOMICAL LANDMARK LOCALIZATION FROM EXTERNAL BODY SURFACE GEOMETRY**

Following the execution and completion of all validation benchmarks, fair neural controls, domain-transfer evaluations, architecture ablations, forensic attention analyses, physical camera simulations, and external landmark debugging under matched 65-epoch protocols, this protocol is frozen.

> [!IMPORTANT]
> **STRICT NON-NEGOTIABLE COMMITMENT:**
> Upon sealing this freeze and computing `PRE_TEST_FREEZE.sha256`:
> - The held-out test split (168 subjects) will be evaluated **EXACTLY ONCE**.
> - **NO** architecture modifications will be permitted.
> - **NO** hyperparameter tuning or learning rate adjustments will be permitted.
> - **NO** target omissions or benchmark subset modifications will be permitted.
> - **NO** metric definitions or SDR thresholds will be altered.
> - **NO** post-test model selection will take place.
> - All reported test numbers represent the definitive, frozen performance of the model family.

## 2. Frozen Baseline and Model Summary (Validation Cohort, N=166)

| Model ID | Method | Macro MRE (mm) | Median (mm) | SDR@10 (%) | SDR@20 (%) |
|---|---|---|---|---|---|
| C0 | Population Atlas Centroid | 65.97 | 50.12 | 2.5% | 12.2% |
| C1 | Linear Ridge Regression | 63.74 | 49.98 | 1.9% | 10.2% |
| C5 | Internal Statistical Shape Model (SSM/PCA) | 52.56 | 39.86 | 3.6% | 17.3% |
| C2 | PointNet++ Direct Regressor (3-seed mean) | 41.40 ± 0.60 | 27.87 | 7.9% | 31.3% |
| C3 | DGCNN Target Decoder (3-seed mean) | 43.48 ± 0.83 | 29.18 | 6.6% | 28.1% |
| C4 | **Proposed Model (3-seed mean)** | **24.69 ± 0.44** | **18.42** | **15.1%** | **52.0%** |
| C4-Ens | **Proposed 3-Model Ensemble** | **23.39** | **18.42** | **17.0%** | **55.8%** |

## 3. Frozen Cross-Domain Transfer Evidence (Matched 65 Epochs)
- **V2-only Model (N=336):** Val V2 = 26.76 ± 0.98 mm | Val TS = 62.60 ± 0.92 mm
- **TS-only Model (N=998):** Val V2 = 26.49 ± 0.96 mm | Val TS = 28.86 ± 0.23 mm
- **Pooled Model (N=1334):** Val V2 = 19.96 ± 0.35 mm | Val TS = 28.05 ± 0.58 mm
- **Delta V2:** -6.80 mm | **Delta TS:** -0.80 mm (positive cross-domain transfer confirmed)

## 4. Frozen Cryptographic Checkpoint Hashes
The three canonical proposed model checkpoints for test evaluation are sealed:
- **Seed 42:** `experiments/phase10R/checkpoints/C4_Proposed_seed42.pt`
  SHA-256: `760adfee1c47d80fc3f59492971b6c83459fc10ce2166f13fc1d7e4d708cb936`
- **Seed 43:** `experiments/phase10R/checkpoints/C4_Proposed_seed43.pt`
  SHA-256: `3e3c739c992b80f193ec41041e62ff59191436e2e5f1e78bd0a328c8310f7d91`
- **Seed 44:** `experiments/phase10R/checkpoints/C4_Proposed_seed44.pt`
  SHA-256: `28fbc7b30d8539f84c265b4f95f4f73b63978fb5b252f9a3ad20ea4fa1792cdd`

## 5. Frozen Test Evaluation Protocol
1. Load held-out test split ($N=168$) from `splits_v3_iid.json`.
2. Evaluate individual models (seeds 42, 43, 44) on test input points ($4096 \times 3$).
3. Compute prediction ensemble: $\bar{\mathbf{C}}_{test} = \frac{1}{3} \sum_{s=1}^3 \hat{\mathbf{C}}_s$.
4. Evaluate primary 104 targets using valid ground-truth masks.
5. Compute Macro MRE, Micro MRE, Median, P90, SDR@5, SDR@10, SDR@15, SDR@20.
6. Save raw predictions to `reports/phase10R/predictions/final_test_predictions.npz`.
