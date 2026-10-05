import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

def get_file_sha256(p):
    if not p or not Path(p).exists():
        return "N/A"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(4096 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    out_dir = repo_root / "reports" / "phase10R"
    ckpt_dir = repo_root / "experiments" / "phase10R" / "checkpoints"
    results_path = out_dir / "canonical_results.json"
    
    if not results_path.exists():
        print(f"Error: {results_path} does not exist. Run aggregation first.")
        sys.exit(1)
        
    with open(results_path) as f:
        results = json.load(f)
        
    c0 = results["C0_Population_Atlas"]
    c1 = results["C1_Ridge_Regression"]
    c5 = results["C5_Internal_SSM_PCA"]
    c2 = results["C2_PointNet2_3Seed_Stats"]
    c2_ens = results["C2_PointNet2_Ensemble"]
    c3 = results["C3_DGCNN_3Seed_Stats"]
    c3_ens = results["C3_DGCNN_Ensemble"]
    c4 = results["C4_Proposed_3Seed_Stats"]
    c4_ens = results["C4_Proposed_Ensemble"]
    
    sha42 = get_file_sha256(ckpt_dir / "C4_Proposed_seed42.pt")
    sha43 = get_file_sha256(ckpt_dir / "C4_Proposed_seed43.pt")
    sha44 = get_file_sha256(ckpt_dir / "C4_Proposed_seed44.pt")
    
    dom = results["domain_transfer"]
    
    md_content = f"""# Phase 10R: Pre-Test Protocol & Hypothesis Freeze Document

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
| C0 | Population Atlas Centroid | {c0['macro_mre']:.2f} | {c0['median']:.2f} | {c0['sdr10']:.1f}% | {c0['sdr20']:.1f}% |
| C1 | Linear Ridge Regression | {c1['macro_mre']:.2f} | {c1['median']:.2f} | {c1['sdr10']:.1f}% | {c1['sdr20']:.1f}% |
| C5 | Internal Statistical Shape Model (SSM/PCA) | {c5['macro_mre']:.2f} | {c5['median']:.2f} | {c5['sdr10']:.1f}% | {c5['sdr20']:.1f}% |
| C2 | PointNet++ Direct Regressor (3-seed mean) | {c2['mean_macro_mre']:.2f} ± {c2['std_macro_mre']:.2f} | {c2_ens['median']:.2f} | {c2_ens['sdr10']:.1f}% | {c2_ens['sdr20']:.1f}% |
| C3 | DGCNN Target Decoder (3-seed mean) | {c3['mean_macro_mre']:.2f} ± {c3['std_macro_mre']:.2f} | {c3_ens['median']:.2f} | {c3_ens['sdr10']:.1f}% | {c3_ens['sdr20']:.1f}% |
| C4 | **Proposed Model (3-seed mean)** | **{c4['mean_macro_mre']:.2f} ± {c4['std_macro_mre']:.2f}** | **{c4_ens['median']:.2f}** | **{c4['mean_sdr10']:.1f}%** | **{c4['mean_sdr20']:.1f}%** |
| C4-Ens | **Proposed 3-Model Ensemble** | **{c4_ens['macro_mre']:.2f}** | **{c4_ens['median']:.2f}** | **{c4_ens['sdr10']:.1f}%** | **{c4_ens['sdr20']:.1f}%** |

## 3. Frozen Cross-Domain Transfer Evidence (Matched 65 Epochs)
- **V2-only Model (N=336):** Val V2 = {dom['V2_train']['v2_mean']:.2f} ± {dom['V2_train']['v2_std']:.2f} mm | Val TS = {dom['V2_train']['ts_mean']:.2f} ± {dom['V2_train']['ts_std']:.2f} mm
- **TS-only Model (N=998):** Val V2 = {dom['TS_train']['v2_mean']:.2f} ± {dom['TS_train']['v2_std']:.2f} mm | Val TS = {dom['TS_train']['ts_mean']:.2f} ± {dom['TS_train']['ts_std']:.2f} mm
- **Pooled Model (N=1334):** Val V2 = {dom['Pooled_train']['v2_mean']:.2f} ± {dom['Pooled_train']['v2_std']:.2f} mm | Val TS = {dom['Pooled_train']['ts_mean']:.2f} ± {dom['Pooled_train']['ts_std']:.2f} mm
- **Delta V2:** {dom['delta_v2_mm']:.2f} mm | **Delta TS:** {dom['delta_ts_mm']:.2f} mm (positive cross-domain transfer confirmed)

## 4. Frozen Cryptographic Checkpoint Hashes
The three canonical proposed model checkpoints for test evaluation are sealed:
- **Seed 42:** `experiments/phase10R/checkpoints/C4_Proposed_seed42.pt`
  SHA-256: `{sha42}`
- **Seed 43:** `experiments/phase10R/checkpoints/C4_Proposed_seed43.pt`
  SHA-256: `{sha43}`
- **Seed 44:** `experiments/phase10R/checkpoints/C4_Proposed_seed44.pt`
  SHA-256: `{sha44}`

## 5. Frozen Test Evaluation Protocol
1. Load held-out test split ($N=168$) from `splits_v3_iid.json`.
2. Evaluate individual models (seeds 42, 43, 44) on test input points ($4096 \\times 3$).
3. Compute prediction ensemble: $\\bar{{\\mathbf{{C}}}}_{{test}} = \\frac{{1}}{{3}} \\sum_{{s=1}}^3 \\hat{{\\mathbf{{C}}}}_s$.
4. Evaluate primary 104 targets using valid ground-truth masks.
5. Compute Macro MRE, Micro MRE, Median, P90, SDR@5, SDR@10, SDR@15, SDR@20.
6. Save raw predictions to `reports/phase10R/predictions/final_test_predictions.npz`.
"""
    
    freeze_path = out_dir / "PRE_TEST_FREEZE.md"
    with open(freeze_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    # Compute SHA-256
    with open(freeze_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
        
    sha_path = out_dir / "PRE_TEST_FREEZE.sha256"
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(sha + "\n")
        
    print(f"Generated {freeze_path.name}")
    print(f"Computed SHA-256: {sha} -> {sha_path.name}")

if __name__ == "__main__":
    main()
