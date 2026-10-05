#!/usr/bin/env python3
"""
tools/phase11/03_amos_target_mapping.py

Step 4 of Phase 11: Semantic target mapping between official AMOS22 label IDs (1-15)
and the frozen Phase-10R 117-slot ontology.

Produces:
- reports/phase11/mappings/AMOS_target_mapping.csv
- reports/phase11/mappings/AMOS_target_mapping.sha256
- reports/phase11/amos/04_AMOS_target_mapping.md
"""

import os
import hashlib
import pandas as pd

CSV_PATH = "reports/phase11/mappings/AMOS_target_mapping.csv"
SHA_PATH = "reports/phase11/mappings/AMOS_target_mapping.sha256"
REPORT_PATH = "reports/phase11/amos/04_AMOS_target_mapping.md"

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Phase 11: AMOS Target Mapping Generation ===")
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    # Load Phase 10R ontology to verify target indices and names
    ontology_path = "sharon/dataset_v3/target_ontology_v3.csv"
    onto_df = pd.read_csv(ontology_path)
    onto_name_to_idx = {row['canonical_name']: idx for idx, row in onto_df.iterrows()}
    onto_primary_set = set(onto_df[onto_df['include_primary'] == 'YES']['canonical_name'])

    # Official AMOS 2022 label definitions (1 to 15)
    # References: Ji et al., "AMOS: A Large-Scale Abdominal Multi-Organ Benchmark", IEEE TMI / NeurIPS 2022
    amos_definitions = [
        (1, "spleen", "spleen", "EXACT_MATCH", "Direct anatomical identity", "NO"),
        (2, "right kidney", "kidney_right", "EXACT_MATCH", "Direct anatomical identity (patient right)", "NO"),
        (3, "left kidney", "kidney_left", "EXACT_MATCH", "Direct anatomical identity (patient left)", "NO"),
        (4, "gallbladder", "gallbladder", "EXACT_MATCH", "Direct anatomical identity", "NO"),
        (5, "esophagus", "esophagus", "EXACT_MATCH", "Direct anatomical identity (distal/abdominal segment)", "NO"),
        (6, "liver", "liver", "EXACT_MATCH", "Direct anatomical identity", "NO"),
        (7, "stomach", "stomach", "EXACT_MATCH", "Direct anatomical identity", "NO"),
        (8, "aorta", "aorta", "EXACT_MATCH", "Direct anatomical identity (abdominal aorta)", "NO"),
        (9, "inferior vena cava", "inferior_vena_cava", "EXACT_MATCH", "Direct anatomical identity", "NO"),
        (10, "pancreas", "pancreas", "EXACT_MATCH", "Direct anatomical identity", "NO"),
        (11, "right adrenal gland", "adrenal_gland_right", "EXACT_MATCH", "Direct anatomical identity (patient right)", "NO"),
        (12, "left adrenal gland", "adrenal_gland_left", "EXACT_MATCH", "Direct anatomical identity (patient left)", "NO"),
        (13, "duodenum", "duodenum", "EXACT_MATCH", "Direct anatomical identity", "NO"),
        (14, "urinary bladder", "urinary_bladder", "EXACT_MATCH", "Direct anatomical identity", "NO"),
        (15, "prostate/uterus", "prostate", "SEX_SPECIFIC_MATCH", "Male prostate maps to slot 21; female uterus not in 117-ontology", "YES")
    ]

    records = []
    for amos_id, amos_name, phase10_name, status, reason, sex_spec in amos_definitions:
        if phase10_name not in onto_name_to_idx:
            raise KeyError(f"Mapped name '{phase10_name}' not in Phase-10R target ontology!")
        
        idx = onto_name_to_idx[phase10_name]
        is_primary = "YES" if phase10_name in onto_primary_set else "NO"

        records.append({
            "AMOS_label_id": amos_id,
            "AMOS_label_name": amos_name,
            "Phase10R_target_name": phase10_name,
            "Phase10R_target_index": idx,
            "mapping_status": status,
            "reason": reason,
            "sex_specific": sex_spec,
            "primary_external_target_yes_no": is_primary
        })

    mapping_df = pd.DataFrame(records)
    mapping_df.to_csv(CSV_PATH, index=False)
    print(f"Wrote target mapping to {CSV_PATH}")

    sha = sha256_file(CSV_PATH)
    with open(SHA_PATH, "w") as f:
        f.write(f"{sha}  {os.path.basename(CSV_PATH)}\n")
    print(f"Sealed mapping SHA256: {sha}")

    # Generate Markdown Report
    with open(REPORT_PATH, "w") as f:
        f.write("# AMOS22 Target Mapping & Semantic Alignment\n\n")
        f.write("## 1. Protocol Integrity & Freezing\n")
        f.write("This mapping was constructed strictly prior to running model inference, based exclusively on official anatomical definitions and nomenclature.\n")
        f.write(f"- **SHA-256 Checksum:** `{sha}`\n\n")
        f.write("## 2. Definitive Target Alignment Table\n\n")
        f.write("| AMOS ID | AMOS Label | Phase 10R Target | Model Index | Primary V3 Target? | Mapping Status | Sex Specific? |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, r in mapping_df.iterrows():
            f.write(f"| {r['AMOS_label_id']} | {r['AMOS_label_name']} | `{r['Phase10R_target_name']}` | {r['Phase10R_target_index']} | {r['primary_external_target_yes_no']} | {r['mapping_status']} | {r['sex_specific']} |\n")
        f.write("\n## 3. Analysis of Target Support\n")
        f.write("- **Primary Benchmark Intersection:** 14 organs (`spleen`, `kidney_right`, `kidney_left`, `gallbladder`, `esophagus`, `liver`, `stomach`, `aorta`, `inferior_vena_cava`, `pancreas`, `adrenal_gland_right`, `adrenal_gland_left`, `duodenum`, `urinary_bladder`) are members of the primary 104 benchmark target set.\n")
        f.write("- **Secondary / Auxiliary Targets:** AMOS label 15 (`prostate/uterus`) represents male prostate and female uterus. In Phase 10R, slot 21 corresponds to `prostate` (non-primary target). For female patients, uterus is excluded from model error evaluation as it is not in the 117-target ontology.\n")
        f.write("- **Common Intersection:** The headline common-target cross-dataset benchmark between Dataset V3 test set and AMOS will evaluate the matched 14 abdominal organ targets.\n")

    print(f"Wrote report to {REPORT_PATH}")
    print("=== AMOS Target Mapping Complete ===")

if __name__ == "__main__":
    main()
