import sys
import csv
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import (
    ORGAN_NAMES, TOTAL_CLASSES_121, SKELETAL_CLASSES,
    SOFT_TISSUE_CLASSES, ANATOMICAL_REGIONS, REGION_NAMES,
    MALE_SPECIFIC_ORGANS, FEMALE_SPECIFIC_ORGANS
)

def generate_target_definition_table():
    out_csv = repo_root / "reports" / "phase1" / "02_target_definition_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, name in enumerate(ORGAN_NAMES):
        organ_id = TOTAL_CLASSES_121[name]
        r_id = ANATOMICAL_REGIONS[name]
        region_str = REGION_NAMES[r_id]
        skel_or_soft = "Skeletal" if name in SKELETAL_CLASSES else "Soft-Tissue"

        if organ_id <= 117:
            gen_method = "scipy.ndimage.center_of_mass on TotalSegmentator v2 mask"
            source_file = "segmentation.nii.gz (TotalSegmentator labels 1-117)"
            validity_rule = "Present in CT field of view; nonzero mask voxels"
        else:
            gen_method = "Synthetic ellipsoidal landmark placement relative to bladder & sacrum"
            source_file = "dataset_preprocessing.py (_extract_female_pelvic_anatomy)"
            validity_rule = "Present only if biological sex is Female (sex==0)"

        if name in MALE_SPECIFIC_ORGANS:
            sex_spec = "Male-only (absent in females)"
        elif name in FEMALE_SPECIFIC_ORGANS:
            sex_spec = "Female-only (absent in males)"
        else:
            sex_spec = "Both sexes"

        notes = []
        if organ_id in [118, 119, 120, 121]:
            notes.append("Synthetic fallback geometry in dataset_preprocessing.py")
        if "vertebrae_" in name or "rib_" in name:
            notes.append("Elongated/curved bone; centroid is volume midpoint")
        if name in ["aorta", "inferior_vena_cava"]:
            notes.append("Longitudinal tubular vessel; centroid sensitive to CT slice coverage")

        note_str = "; ".join(notes) if notes else "Standard anatomical structure"

        rows.append({
            "target_index": idx,
            "target_name": name,
            "anatomical_group": region_str,
            "skeletal_or_soft_tissue": skel_or_soft,
            "generation_method": gen_method,
            "source_file": source_file,
            "validity_rule": validity_rule,
            "sex_specific": sex_spec,
            "notes": note_str
        })

    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "target_index", "target_name", "anatomical_group", "skeletal_or_soft_tissue",
            "generation_method", "source_file", "validity_rule", "sex_specific", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Done] Generated target definition table: {out_csv} ({len(rows)} entries)")

if __name__ == "__main__":
    generate_target_definition_table()
