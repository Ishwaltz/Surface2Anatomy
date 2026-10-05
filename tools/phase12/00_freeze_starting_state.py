#!/usr/bin/env python3
"""
tools/phase12/00_freeze_starting_state.py

Freezes and cryptographically hashes the starting state from Phase 10R and Phase 11.
Produces reports/phase12/00_governance/PHASE12_STARTING_STATE.json
"""

import os
import hashlib
import json
import glob
import pandas as pd
import numpy as np

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def compute_directory_hash(dir_path, ext="*.npz"):
    files = sorted(glob.glob(os.path.join(dir_path, ext)))
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.basename(f).encode())
        with open(f, "rb") as fp:
            while chunk := fp.read(1024 * 1024):
                h.update(chunk)
    return h.hexdigest(), len(files)

def main():
    print("=== Phase 12: Starting State Freeze & Audit ===")
    
    # 1. Checkpoint files
    ckpts = {
        "seed42": "experiments/phase10R/checkpoints/C4_Proposed_seed42.pt",
        "seed43": "experiments/phase10R/checkpoints/C4_Proposed_seed43.pt",
        "seed44": "experiments/phase10R/checkpoints/C4_Proposed_seed44.pt",
    }
    ckpt_hashes = {k: compute_sha256(v) for k, v in ckpts.items()}
    
    # 2. Phase-11 predictions
    preds = {
        "AMOS_ensemble": "reports/phase11/predictions/AMOS_ensemble.npz",
        "AMOS_seed42": "reports/phase11/predictions/AMOS_seed42.npz",
        "AMOS_seed43": "reports/phase11/predictions/AMOS_seed43.npz",
        "AMOS_seed44": "reports/phase11/predictions/AMOS_seed44.npz",
        "HuMMan_ensemble": "reports/phase11/predictions/HuMMan_ensemble.npz"
    }
    pred_hashes = {k: compute_sha256(v) for k, v in preds.items()}

    # 3. AMOS GT Centroids
    gt_file = "data_external/AMOS22/processed/AMOS_GT_centroids.csv"
    gt_hash = compute_sha256(gt_file)

    # 4. AMOS point clouds
    pc_dir = "data_external/AMOS22/processed/pointclouds"
    pc_hash, num_pcs = compute_directory_hash(pc_dir, "*.npz")

    # 5. Target mapping & FOV
    mapping_file = "reports/phase11/mappings/AMOS_target_mapping.csv"
    mapping_hash = compute_sha256(mapping_file)
    fov_file = "reports/phase11/amos/AMOS_fov_audit.csv"
    fov_hash = compute_sha256(fov_file)

    # 6. Canonicalization code
    canon_code = "tools/phase11/04_amos_surface_and_gt.py"
    canon_hash = compute_sha256(canon_code)
    phase11_results_json = "reports/phase11/amos/05_AMOS_zero_shot_results.json"
    
    with open(phase11_results_json, "r") as f:
        p11_res = json.load(f)

    starting_state = {
        "phase": 12,
        "purpose": "External Generalization Recovery & Canonical-Frame Forensics",
        "timestamp_iso": "2026-09-14T00:06:00Z",
        "frozen_checkpoints": {
            "paths": ckpts,
            "sha256": ckpt_hashes
        },
        "phase11_predictions": {
            "paths": preds,
            "sha256": pred_hashes
        },
        "ground_truth": {
            "path": gt_file,
            "sha256": gt_hash
        },
        "pointclouds": {
            "directory": pc_dir,
            "count": num_pcs,
            "manifest_sha256": pc_hash
        },
        "mappings_and_fov": {
            "target_mapping_path": mapping_file,
            "target_mapping_sha256": mapping_hash,
            "fov_audit_path": fov_file,
            "fov_audit_sha256": fov_hash
        },
        "canonicalization_code": {
            "path": canon_code,
            "sha256": canon_hash
        },
        "historical_baselines": {
            "v3_common_target_macro_mre_mm": 26.69,
            "v3_locked_test_full_macro_mre_mm": 27.04,
            "amos_overall_macro_mre_mm": p11_res["macro_mre"],
            "amos_overall_micro_mre_mm": p11_res["overall_metrics"]["mean"],
            "amos_overall_median_mm": p11_res["overall_metrics"]["median"],
            "amos_overall_p90_mm": p11_res["overall_metrics"]["p90"],
            "amos_overall_95ci_mm": p11_res["ci_95"],
            "amos_ct_macro_mre_mm": p11_res["modality_breakdown"]["CT"]["macro_mre"],
            "amos_ct_median_mm": p11_res["modality_breakdown"]["CT"]["median"],
            "amos_mri_macro_mre_mm": p11_res["modality_breakdown"]["MRI"]["macro_mre"],
            "amos_mri_median_mm": p11_res["modality_breakdown"]["MRI"]["median"],
            "generalization_gap_mm": p11_res["macro_mre"] - 26.69,
            "generalization_gap_pct": ((p11_res["macro_mre"] - 26.69) / 26.69) * 100.0,
            "evaluated_cases": p11_res["total_evaluated_cases"]
        }
    }

    out_json = "reports/phase12/00_governance/PHASE12_STARTING_STATE.json"
    with open(out_json, "w") as f:
        json.dump(starting_state, f, indent=2)

    # Also compute sha256 of the starting state file itself
    out_hash = compute_sha256(out_json)
    with open("reports/phase12/00_governance/PHASE12_STARTING_STATE.sha256", "w") as f:
        f.write(f"{out_hash}  PHASE12_STARTING_STATE.json\n")

    print(f"Saved frozen starting state to {out_json}")
    print(f"SHA-256: {out_hash}")
    print(f"Verified baselines: V3 Common={starting_state['historical_baselines']['v3_common_target_macro_mre_mm']} mm, AMOS Overall={starting_state['historical_baselines']['amos_overall_macro_mre_mm']:.2f} mm")

if __name__ == "__main__":
    main()
