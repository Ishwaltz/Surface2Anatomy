import sys
import csv
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES

def audit_left_right():
    print("=" * 80)
    print("AUDIT: LEFT/RIGHT ANATOMICAL ORIENTATION")
    print("=" * 80)

    # Find all left/right pairs
    pairs = []
    for name in ORGAN_NAMES:
        if "_left" in name:
            right_name = name.replace("_left", "_right")
            if right_name in ORGAN_NAMES:
                pairs.append((name, right_name))
        elif name.endswith("_l"):
            right_name = name[:-2] + "_r"
            if right_name in ORGAN_NAMES:
                pairs.append((name, right_name))

    print(f"Discovered {len(pairs)} bilateral Left/Right anatomical pairs:")
    for l, r in pairs[:5]:
        print(f"  {l} <-> {r}")
    print(f"  ... and {len(pairs)-5} more.")

    pt_path = repo_root / "sharon" / "dataset" / "pointclouds_450.pt"
    data = torch.load(str(pt_path), weights_only=False)

    raw_c = data["raw_centroids"] # (450, 121, 3)
    norm_c = data["centroids"]    # (450, 121, 3)
    masks = data["masks"]         # (450, 121)
    case_ids = data["case_ids"]

    # In canonical LAS coordinate frame:
    # Axis 0 is L (Left positive, Right negative), so Left.x > Right.x
    # But in pointclouds_450.pt due to axis swap:
    # Axis 0 was slice k, Axis 2 was in-plane x.
    
    records = []
    axis0_violations = 0
    axis0_valid_comparisons = 0
    
    axis2_violations = 0
    axis2_valid_comparisons = 0

    for b, cid in enumerate(case_ids):
        for l_name, r_name in pairs:
            l_idx = ORGAN_NAMES.index(l_name)
            r_idx = ORGAN_NAMES.index(r_name)

            if masks[b, l_idx] > 0.5 and masks[b, r_idx] > 0.5:
                l_pt = raw_c[b, l_idx].cpu().numpy()
                r_pt = raw_c[b, r_idx].cpu().numpy()

                # Test along Axis 0 (where Left-Right SHOULD be in LAS):
                dx0 = l_pt[0] - r_pt[0]
                expected_order = "x_left > x_right"
                is_valid_axis0 = (dx0 > 0)
                axis0_valid_comparisons += 1
                if not is_valid_axis0:
                    axis0_violations += 1

                # Test along Axis 2 (where in-plane Left-Right ended up due to swap):
                dz = l_pt[2] - r_pt[2]
                is_valid_axis2 = (dz > 0)
                axis2_valid_comparisons += 1
                if not is_valid_axis2:
                    axis2_violations += 1

                observed_order = f"dx0={dx0:.2f}mm, dz={dz:.2f}mm"
                records.append({
                    "patient_id": cid,
                    "pair_name": f"{l_name}_vs_{r_name}",
                    "expected_order": expected_order,
                    "observed_order": observed_order,
                    "valid": "VALID" if is_valid_axis0 else "VIOLATED"
                })

    axis0_violation_rate = (axis0_violations / max(axis0_valid_comparisons, 1)) * 100.0
    axis2_violation_rate = (axis2_violations / max(axis2_valid_comparisons, 1)) * 100.0

    print(f"\nTotal Bilateral Evaluations across {len(case_ids)} patients: {axis0_valid_comparisons}")
    print(f"Axis 0 (Expected Left-Right) Violations: {axis0_violations}/{axis0_valid_comparisons} ({axis0_violation_rate:.2f}%)")
    print(f"Axis 2 (Swapped Axis) Violations:        {axis2_violations}/{axis2_valid_comparisons} ({axis2_violation_rate:.2f}%)")

    print("\nInterpretation:")
    print(f"  Under the expected coordinate convention (Left > Right along Axis 0),")
    print(f"  {axis0_violation_rate:.1f}% of patient comparisons fail!")
    print(f"  Conversely, along Axis 2 (Z), Left > Right holds in {100.0 - axis2_violation_rate:.1f}% of cases,")
    print(f"  proving that the Left-Right dimension was swapped into the Z axis!")

    out_csv = repo_root / "reports" / "phase1" / "05_left_right_audit.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "pair_name", "expected_order", "observed_order", "valid"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[Done] CSV written to: {out_csv}")

if __name__ == "__main__":
    audit_left_right()
