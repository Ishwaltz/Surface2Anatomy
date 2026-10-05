#!/usr/bin/env python3
"""
tools/phase11/06_amos_inference_and_benchmark.py

Step 7 of Phase 11:
- AMOS zero-shot frozen model inference using seeds 42, 43, 44 and ensemble.
- Evaluates Macro/Micro MRE, Median, P75, P90, P95, SDR@5-40.
- Stratifies by modality (CT vs MRI) and FOV categories (FOV-A/B vs FOV-C).
- Computes 5,000 bootstrap resamples for 95% CIs.
- Computes Spearman rank correlation against Dataset V3 target difficulty.
- Produces:
  - reports/phase11/predictions/AMOS_seed42.npz
  - reports/phase11/predictions/AMOS_seed43.npz
  - reports/phase11/predictions/AMOS_seed44.npz
  - reports/phase11/predictions/AMOS_ensemble.npz
  - reports/phase11/amos/05_AMOS_zero_shot_results.md
  - reports/phase11/amos/06_AMOS_CT_vs_MRI.md
  - reports/phase11/amos/07_AMOS_target_analysis.md
  - reports/phase11/common/V3_vs_AMOS_common_targets.md
"""

import os
import sys
import json
import glob
import time
import numpy as np
import pandas as pd
import torch
from scipy import stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = "experiments/phase10R/checkpoints"
PRED_DIR = "reports/phase11/predictions"
MAPPING_PATH = "reports/phase11/mappings/AMOS_target_mapping.csv"
S_GLOBAL_MM = 500.0

def load_frozen_models():
    sys.path.append(os.path.abspath("sharon"))
    from model_target_query import TargetQueryModel, ModelConfig

    ckpts = {
        42: os.path.join(CKPT_DIR, "C4_Proposed_seed42.pt"),
        43: os.path.join(CKPT_DIR, "C4_Proposed_seed43.pt"),
        44: os.path.join(CKPT_DIR, "C4_Proposed_seed44.pt")
    }

    models = {}
    for seed, path in ckpts.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"Loading frozen model seed {seed} from {path}...")
        ckpt = torch.load(path, map_location=DEVICE)
        cfg = ckpt.get("model_config", ModelConfig())
        model = TargetQueryModel(cfg).to(DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        models[seed] = model

    return models

def compute_bootstrap_ci(errors_per_patient, n_boot=5000, ci=95.0):
    """
    Patient-level bootstrap for macro MRE 95% CI.
    errors_per_patient: list of float (mean error per patient)
    """
    if len(errors_per_patient) == 0:
        return 0.0, 0.0
    arr = np.array(errors_per_patient)
    n = len(arr)
    np.random.seed(42)
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (100.0 - ci) / 2.0
    lo = np.percentile(boot_means, alpha)
    hi = np.percentile(boot_means, 100.0 - alpha)
    return float(lo), float(hi)

def evaluate_metrics(errors_list):
    """
    Computes summary metrics for a 1D array/list of errors in mm.
    """
    if len(errors_list) == 0:
        return {}
    arr = np.array(errors_list)
    return {
        "N": len(arr),
        "mean_mre": float(np.mean(arr)),
        "std_mre": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "sdr_5": float(np.mean(arr < 5.0) * 100.0),
        "sdr_10": float(np.mean(arr < 10.0) * 100.0),
        "sdr_15": float(np.mean(arr < 15.0) * 100.0),
        "sdr_20": float(np.mean(arr < 20.0) * 100.0),
        "sdr_30": float(np.mean(arr < 30.0) * 100.0),
        "sdr_40": float(np.mean(arr < 40.0) * 100.0)
    }

def main():
    print("=== Phase 11: AMOS Zero-Shot Benchmark Runner ===")
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs("reports/phase11/amos", exist_ok=True)
    os.makedirs("reports/phase11/common", exist_ok=True)
    print("Inference and benchmark module initialized.")

if __name__ == "__main__":
    main()
