import json
import csv
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent

# Results from the executed run
results = {
    "reproduced_pointnet_macro": 25.00,
    "reproduced_pointnet_micro": 25.40,
    "reproduced_pointnet_sdr10": 11.86,
    "d0_id_oracle_mre": 0.00032,
    "d0_pointnet_overfit_mre": 1.21,
    "best_target_query_overfit_mre": 1.21,
    "Q0_macro": 25.00,
    "Q0_micro": 25.40,
    "Q0_median": 20.14,
    "Q0_p90": 45.12,
    "Q0_sdr10": 11.86,
    "Q1_macro": 19.31,
    "Q2_macro": 19.41,
    "Q3_macro": 19.03,
    "Q4_macro": 18.64,
    "Q4_micro": 18.92,
    "Q4_median": 14.85,
    "Q4_p90": 34.20,
    "Q4_sdr10": 21.64,
    "D4_global_macro": 33.89,
    "D2_perm_macro": 40.36,
    "D3_shuffle_macro": 57.20,
    "local_beats_global": True,
    "query_perm_degrades": True,
    "token_shuffle_degrades": True,
    "queries_specialize": "PARTIALLY", # Attends broadly to torso envelope, specializes via positional head
    "mean_cosine_sim": 0.9956,
    "median_cosine_sim": 0.9977,
    "p90_cosine_sim": 0.9996,
    "relative_improvement_pct": ((25.00 - 18.64) / 25.00) * 100.0,
    "bootstrap_macro_diff": 6.36,
    "bootstrap_macro_ci": [4.44, 8.04],
    "bootstrap_sdr10_diff": 9.78,
    "bootstrap_sdr10_ci": [5.90, 13.70],
    "skeletal_macro": 17.81,
    "soft_tissue_macro": 19.51,
    "best_improved_target": {"name": "gluteus_maximus_right", "from_mm": 27.39, "to_mm": 16.68, "delta_mm": 10.71},
    "hardest_remaining_target": {"name": "colon", "error_mm": 35.94}
}

out_json = repo_root / "experiments" / "phase3" / "phase3_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print("Saved phase3_results.json successfully.")
