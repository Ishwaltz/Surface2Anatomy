#!/usr/bin/env python3
"""
tools/phase11_flare/01_setup_label_mapping.py
Establishes the official FLARE22 label map and cross-references it with
our primary 104 benchmark target ontology from Dataset V3.
"""

import os, sys, csv, json
from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
out_dir = repo_root / "reports" / "phase11_flare"
out_dir.mkdir(parents=True, exist_ok=True)

# Primary ontology
onto_path = repo_root / "sharon/dataset_v3/target_ontology_v3.csv"
onto_df = pd.read_csv(onto_path)

FLARE22_MAPPINGS = [
    {"flare_label_id": 1,  "flare_name": "liver",               "our_target_name": "liver",               "mapping_status": "EXACT", "notes": "Direct parenchymal equivalence"},
    {"flare_label_id": 2,  "flare_name": "right kidney",         "our_target_name": "kidney_right",        "mapping_status": "EXACT", "notes": "Direct anatomical identity (patient right)"},
    {"flare_label_id": 3,  "flare_name": "spleen",               "our_target_name": "spleen",               "mapping_status": "EXACT", "notes": "Direct parenchymal equivalence"},
    {"flare_label_id": 4,  "flare_name": "pancreas",             "our_target_name": "pancreas",             "mapping_status": "EXACT", "notes": "Consistent retroperitoneal landmark"},
    {"flare_label_id": 5,  "flare_name": "aorta",                "our_target_name": "aorta",                "mapping_status": "EXACT", "notes": "Abdominal aorta segment"},
    {"flare_label_id": 6,  "flare_name": "inferior vena cava",   "our_target_name": "inferior_vena_cava",   "mapping_status": "EXACT", "notes": "Direct anatomical identity"},
    {"flare_label_id": 7,  "flare_name": "right adrenal gland",  "our_target_name": "adrenal_gland_right",  "mapping_status": "EXACT", "notes": "Localized retroperitoneal gland"},
    {"flare_label_id": 8,  "flare_name": "left adrenal gland",   "our_target_name": "adrenal_gland_left",   "mapping_status": "EXACT", "notes": "Localized retroperitoneal gland"},
    {"flare_label_id": 9,  "flare_name": "gallbladder",          "our_target_name": "gallbladder",          "mapping_status": "EXACT", "notes": "Direct anatomical identity"},
    {"flare_label_id": 10, "flare_name": "esophagus",            "our_target_name": "esophagus",            "mapping_status": "EXACT", "notes": "Distal abdominal/lower thoracic esophagus"},
    {"flare_label_id": 11, "flare_name": "stomach",              "our_target_name": "stomach",              "mapping_status": "EXACT", "notes": "Direct parenchymal/luminal identity"},
    {"flare_label_id": 12, "flare_name": "duodenum",             "our_target_name": "duodenum",             "mapping_status": "EXACT", "notes": "Duodenal C-loop identity"},
    {"flare_label_id": 13, "flare_name": "left kidney",          "our_target_name": "kidney_left",         "mapping_status": "EXACT", "notes": "Direct anatomical identity (patient left)"},
]

def main():
    csv_path = out_dir / "01_flare_label_map.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["flare_label_id", "flare_name", "our_target_name", "mapping_status", "notes"])
        writer.writeheader()
        for row in FLARE22_MAPPINGS:
            writer.writerow(row)
    print(f"Generated: {csv_path}")

    # Verify all in ontology
    verified_overlap = []
    for m in FLARE22_MAPPINGS:
        tname = m["our_target_name"]
        match = onto_df[onto_df["canonical_name"] == tname]
        assert len(match) > 0, f"Target {tname} not in ontology!"
        t_idx = int(match.iloc[0]["canonical_target_id"] - 1)
        verified_overlap.append({
            "target": tname,
            "our_index": t_idx,
            "flare_label": m["flare_label_id"],
            "flare_name": m["flare_name"],
            "mapping_status": m["mapping_status"]
        })
    
    json_path = out_dir / "04_frozen_overlap_targets.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_overlapping_targets": len(verified_overlap),
            "ontology_scope": "Primary 104 benchmark targets",
            "targets": verified_overlap
        }, f, indent=2)
    print(f"Generated: {json_path}")

if __name__ == "__main__":
    main()
