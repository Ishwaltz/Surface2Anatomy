#!/usr/bin/env python3
"""
tools/phase15_ablation/03_matched_literature_benchmarks.py
==========================================================
Strict 1:1 Matched-Anatomy Open-Source Literature Comparison:
Computes our frozen Phase 10R model performance STRICTLY RESTRICTED to the exact
anatomical subsets reported by published competitive papers.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
out_dir = repo_root / "reports" / "phase15_ablation"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load target-wise test results from our frozen test evaluation
df_targets = pd.read_csv(out_dir / "surface_occlusion_per_target.csv")
df_ref = df_targets[df_targets["mode_id"] == "Full_360_Reference"].set_index("target_name")

# 2. Define exact paper subsets
PAPER_BENCHMARKS = [
    {
        "paper_name": "SAMe (Robotic Ultrasound Mapping)",
        "citation": "arXiv:2604.25646 (2026)",
        "input_modality": "Single 2D RGB image",
        "reported_metric": "3D Centroid Euclidean MRE",
        "reported_value_their_cohort_mm": 22.55,
        "matched_organ_count": 11,
        "matched_targets": [
            "liver", "spleen", "pancreas", "gallbladder", "urinary_bladder",
            "aorta", "trachea", "kidney_right", "kidney_left", "stomach", "inferior_vena_cava"
        ],
        "notes": "Exact 1:1 11-visceral-organ matched subset."
    },
    {
        "paper_name": "From Surface to Viscera (Atici et al.)",
        "citation": "MIDL / PMLR (2026)",
        "input_modality": "Body surface point cloud + internal prior",
        "reported_metric": "Dense Mesh Chamfer Distance (mm)",
        "reported_value_their_cohort_mm": 5.0, # < 5.0 mm Chamfer
        "matched_organ_count": 14,
        "matched_targets": [
            "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas",
            "adrenal_gland_right", "adrenal_gland_left", "urinary_bladder", "aorta", "inferior_vena_cava",
            "portal_vein_and_splenic_vein", "esophagus"
        ],
        "notes": "Major internal visceral organs reported in MIDL 2026."
    },
    {
        "paper_name": "FLARE22 International Challenge",
        "citation": "FLARE22 Zero-Shot External Evaluation",
        "input_modality": "External 3D body surface (zero CT)",
        "reported_metric": "3D Centroid Euclidean MRE",
        "reported_value_their_cohort_mm": 21.30,
        "matched_organ_count": 13,
        "matched_targets": [
            "liver", "kidney_right", "spleen", "pancreas", "aorta", "inferior_vena_cava",
            "adrenal_gland_right", "adrenal_gland_left", "gallbladder", "esophagus",
            "stomach", "duodenum", "kidney_left"
        ],
        "notes": "Completely untouched zero-shot external hospital validation."
    },
    {
        "paper_name": "CT-ORG Clinical Benchmark",
        "citation": "TCIA External Evaluation (Abdominal Subset)",
        "input_modality": "External 3D body surface (zero CT)",
        "reported_metric": "3D Centroid Euclidean MRE",
        "reported_value_their_cohort_mm": 44.68,
        "matched_organ_count": 4,
        "matched_targets": [
            "liver", "kidney_right", "kidney_left", "urinary_bladder"
        ],
        "notes": "4 core abdominal/pelvic organs across 112-139 diverse clinical patients."
    }
]

def main():
    rows = []
    print("=" * 90)
    print("STRICT 1:1 MATCHED-ANATOMY LITERATURE BENCHMARKING:")
    print("=" * 90)
    
    for b in PAPER_BENCHMARKS:
        targets = b["matched_targets"]
        mres = []
        for t in targets:
            if t in df_ref.index:
                mres.append(df_ref.loc[t, "reference_mre_mm"])
            else:
                print(f"Warning: target {t} not found in reference index")
        
        our_matched_macro_mre = float(np.mean(mres))
        our_matched_median_mre = float(np.median(mres))
        
        rows.append({
            "paper_name": b["paper_name"],
            "citation": b["citation"],
            "matched_organ_count": len(targets),
            "matched_organs_list": ", ".join(targets),
            "reported_metric_name": b["reported_metric"],
            "published_value_on_their_data": b["reported_value_their_cohort_mm"],
            "our_model_mre_on_matched_subset_mm": round(our_matched_macro_mre, 2),
            "our_model_median_on_matched_subset_mm": round(our_matched_median_mre, 2),
            "notes": b["notes"]
        })
        
        print(f"Paper: {b['paper_name']} ({b['citation']})")
        print(f"  Matched Organs ({len(targets)}): {', '.join(targets[:5])}...")
        print(f"  Their Reported Score: {b['reported_value_their_cohort_mm']} ({b['reported_metric']})")
        print(f"  Our Frozen Model on EXACT Same Subset: Macro MRE = {our_matched_macro_mre:.2f} mm | Median = {our_matched_median_mre:.2f} mm\n")
        
    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "matched_literature_benchmarks.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved matched literature benchmarks to: {csv_path}")

if __name__ == "__main__":
    main()
