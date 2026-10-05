#!/usr/bin/env python3
"""
tools/phase11/01_freeze_phase10r.py

Step 2 of Phase 11: Freeze Phase-10R model artifacts, architecture, checkpoints,
normalization, ontology, and conventions before external validation.
Produces reports/phase11/00_PHASE11_EXTERNAL_FREEZE.json and .sha256.
"""

import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime, timezone

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit():
    try:
        res = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_GIT_COMMIT"

def main():
    print("=== Phase 11: Freezing Phase-10R Model Artifacts ===")

    # Verify files
    checkpoints = {
        "seed42": "experiments/phase10R/checkpoints/C4_Proposed_seed42.pt",
        "seed43": "experiments/phase10R/checkpoints/C4_Proposed_seed43.pt",
        "seed44": "experiments/phase10R/checkpoints/C4_Proposed_seed44.pt",
        "atlas_prior": "experiments/phase10R/checkpoints/C0_Population_Atlas.pt"
    }

    files_to_verify = {
        "model_architecture_target_query": "sharon/model_target_query.py",
        "model_architecture_gnn": "sharon/model_gnn.py",
        "target_ontology": "sharon/dataset_v3/target_ontology_v3.csv",
        "benchmark_primary_targets": "sharon/dataset_v3/benchmark_primary_targets_v3.json",
        "splits_v3": "sharon/dataset_v3/splits_v3_iid.json",
        "pointclouds_v3": "sharon/dataset_v3/pointclouds_v3.pt",
        "manifest_v3": "sharon/dataset_v3/manifest_v3.csv",
        "phase10r_final_test_predictions": "reports/phase10R/predictions/final_test_predictions.npz",
        "phase10r_pre_test_freeze": "reports/phase10R/PRE_TEST_FREEZE.md"
    }

    verified_hashes = {}
    for name, p in checkpoints.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing checkpoint: {p}")
        verified_hashes[f"checkpoint_{name}"] = {
            "path": p,
            "size_bytes": os.path.getsize(p),
            "sha256": sha256_file(p)
        }

    for name, p in files_to_verify.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artifact: {p}")
        verified_hashes[name] = {
            "path": p,
            "size_bytes": os.path.getsize(p),
            "sha256": sha256_file(p)
        }

    git_commit = get_git_commit()

    freeze_record = {
        "freeze_name": "PHASE11_EXTERNAL_VALIDATION_FREEZE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "statement": "This Phase 11 analysis is strictly post hoc external validation. No external dataset is used for fitting or model selection.",
        "scientific_context": {
            "model_architecture": {
                "encoder": "MultiScaleSurfacePointNet2Encoder (SA1, SA2, SA3)",
                "decoder": "TargetQueryTransformerDecoder (3 cross-attention layers, 8 heads, d_model=256)",
                "total_model_slots": 117,
                "primary_v3_targets": 104
            },
            "input_specification": {
                "points": 4096,
                "channels": 3,
                "format": "XYZ coordinates",
                "normalization_scale_S_global_mm": 500.0,
                "canonical_coordinate_frame": "+X Right, +Y Anterior, +Z Superior",
                "body_centering_convention": "Centered at midpoint of surface bounding box [X_mid, Y_mid, Z_mid]"
            },
            "ensemble_definition": "Unweighted coordinate-wise average of predictions from seeds 42, 43, and 44",
            "git_commit": git_commit
        },
        "verified_file_hashes": verified_hashes
    }

    out_json = "reports/phase11/00_PHASE11_EXTERNAL_FREEZE.json"
    with open(out_json, "w") as f:
        json.dump(freeze_record, f, indent=2)
    print(f"Wrote freeze record to {out_json}")

    freeze_hash = sha256_file(out_json)
    out_sha = "reports/phase11/00_PHASE11_EXTERNAL_FREEZE.sha256"
    with open(out_sha, "w") as f:
        f.write(f"{freeze_hash}  {os.path.basename(out_json)}\n")
    print(f"Sealed freeze hash: {freeze_hash}")
    print("=== Model Freeze Complete ===")

if __name__ == "__main__":
    main()
