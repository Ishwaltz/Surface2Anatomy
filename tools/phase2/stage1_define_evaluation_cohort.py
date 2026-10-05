import sys
import csv
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES

def main():
    print("=" * 80)
    print("STAGE 1: DEFINING PRIMARY EVALUATION COHORT TIERS")
    print("=" * 80)

    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"

    data = torch.load(str(pt_path), weights_only=False)
    import json
    with open(splits_path) as f:
        splits = json.load(f)

    tr_idx = splits["train_indices"]
    val_idx = splits["val_indices"]

    p_mask = data["target_primary_mask"].numpy() # (440, 121)
    t_mask = data["target_mask"].numpy()         # (440, 121)

    tier_records = []
    tier_counts = {"TIER_A": 0, "TIER_B": 0, "TIER_C": 0, "UNUSABLE": 0, "SYNTHETIC": 0}

    for idx, name in enumerate(ORGAN_NAMES):
        organ_id = idx + 1
        is_synthetic = (organ_id in [118, 119, 120, 121])
        
        tr_count = int(p_mask[tr_idx, idx].sum())
        val_count = int(p_mask[val_idx, idx].sum())
        tot_count = int(p_mask[:, idx].sum())

        if is_synthetic:
            tier = "SYNTHETIC"
            desc = "Procedural female pelvic structure; excluded from primary benchmark"
        elif val_count == 0 or tot_count < 5:
            tier = "UNUSABLE"
            desc = "Zero or sparse validation support (<5 observations total)"
        elif tr_count >= 100 and val_count >= 15:
            tier = "TIER_A"
            desc = "Robust support: >=100 train and >=15 val genuine observations"
        elif tr_count >= 40 and val_count >= 5:
            tier = "TIER_B"
            desc = "Moderate support: >=40 train and >=5 val observations"
        else:
            tier = "TIER_C"
            desc = "Low support: genuine target with sparse observation count"

        tier_counts[tier] += 1
        tier_records.append({
            "target_index": idx,
            "target_name": name,
            "tier": tier,
            "train_valid_count": tr_count,
            "val_valid_count": val_count,
            "total_genuine_count": tot_count,
            "description": desc
        })

    out_csv = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"
    with open(out_csv, "w", newline="") as f:
        fieldnames = ["target_index", "target_name", "tier", "train_valid_count", "val_valid_count", "total_genuine_count", "description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tier_records)

    print("Tier Summary across 121 Output Schema Structures:")
    for t, c in tier_counts.items():
        print(f"  {t:12s}: {c:3d} structures")

    # Primary Benchmark Target Set: TIER_A + TIER_B
    primary_targets = [r["target_index"] for r in tier_records if r["tier"] in ["TIER_A", "TIER_B"]]
    print(f"\nPrimary Phase 2 Evaluation Set (TIER_A + TIER_B): {len(primary_targets)} structures")
    print(f"Tier A Primary Targets: {tier_counts['TIER_A']} structures")
    print(f"Tier B Primary Targets: {tier_counts['TIER_B']} structures")
    print(f"[Done] Output written to: {out_csv}")

if __name__ == "__main__":
    main()
