import sys
import json
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES, SKELETAL_INDICES, SOFT_TISSUE_INDICES
from tools.phase2.metrics import compute_all_metrics
from tools.phase2.run_pointnet_experiments import PointNet2PlainRegressor, V2Dataset, train_pointnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 80)
    print("RESUMING POINTNET++ EXPERIMENTS: D1 SHUFFLE, LEARNING CURVES & BOOTSTRAP")
    print("=" * 80)

    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
    tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"

    data = torch.load(str(pt_path), weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)

    tr_idx = np.array(splits["train_indices"])
    val_idx = np.array(splits["val_indices"])

    with open(tiers_path) as f:
        primary_107_indices = [int(r["target_index"]) for r in csv.DictReader(f) if r["tier"] in ["TIER_A", "TIER_B"]]

    pts_m = data["points_model"].numpy()
    tgt_m = data["targets_model"].numpy()
    tgt_mask = data["target_mask"].numpy()
    prim_mask = data["target_primary_mask"].numpy()
    val_tgt_c = data["targets_centered_mm"].numpy()[val_idx]
    S_global = float(data["s_global"])

    val_ds = V2Dataset(pts_m[val_idx], tgt_m[val_idx], tgt_mask[val_idx], prim_mask[val_idx], augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    rng = np.random.RandomState(42)

    # 1. Train seed 42 clean PointNet++ to get exact predictions
    print("\n--- Training Best PointNet++ Plain Model (Seed 42) ---")
    torch.manual_seed(42)
    tr_ds = V2Dataset(pts_m[tr_idx], tgt_m[tr_idx], tgt_mask[tr_idx], prim_mask[tr_idx], augment=True)
    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, drop_last=True)
    
    m_b7 = PointNet2PlainRegressor().to(device)
    best_macro, best_preds_b7, _ = train_pointnet(
        tr_loader, val_loader, m_b7, epochs=70, lr=4e-4, S_global=S_global,
        primary_indices=primary_107_indices, val_tgt_c=val_tgt_c, has_cond=False
    )
    mets_b7 = compute_all_metrics(best_preds_b7, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    print(f"  B7 Macro Target MRE: {mets_b7['macro_target_mre']:.2f} mm | Micro: {mets_b7['micro_mre']:.2f} mm | SDR@10: {mets_b7['sdr_10']:.2f}%")

    # 2. STAGE 25: SURFACE SHUFFLE CONTROL (D1)
    print("\n--- Stage 25: Surface Shuffle Control (D1_SHUFFLED_SURFACE) ---")
    tr_perm = rng.permutation(len(tr_idx))
    tr_ds_shuf = V2Dataset(pts_m[tr_idx[tr_perm]], tgt_m[tr_idx], tgt_mask[tr_idx], prim_mask[tr_idx], augment=True)
    tr_loader_shuf = DataLoader(tr_ds_shuf, batch_size=16, shuffle=True, drop_last=True)
    
    torch.manual_seed(42)
    m_d1 = PointNet2PlainRegressor().to(device)
    _, best_preds_d1, _ = train_pointnet(
        tr_loader_shuf, val_loader, m_d1, epochs=60, lr=4e-4, S_global=S_global,
        primary_indices=primary_107_indices, val_tgt_c=val_tgt_c, has_cond=False
    )
    mets_d1 = compute_all_metrics(best_preds_d1, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    print(f"  D1 Shuffled Surface Macro MRE: {mets_d1['macro_target_mre']:.2f} mm (vs Matched {mets_b7['macro_target_mre']:.2f} mm)")

    # 3. STAGE 33: LEARNING CURVE (25%, 50%, 75%, 100%)
    print("\n--- Stage 33: Learning Curve Subsets ---")
    lc_results = {100: mets_b7['macro_target_mre']}
    for pct in [0.25, 0.50, 0.75]:
        n_sub = int(round(len(tr_idx) * pct))
        sub_indices = tr_idx[:n_sub]
        tr_ds_sub = V2Dataset(pts_m[sub_indices], tgt_m[sub_indices], tgt_mask[sub_indices], prim_mask[sub_indices], augment=True)
        tr_loader_sub = DataLoader(tr_ds_sub, batch_size=16, shuffle=True, drop_last=True)
        
        torch.manual_seed(42)
        m_sub = PointNet2PlainRegressor().to(device)
        macro_sub, _, _ = train_pointnet(
            tr_loader_sub, val_loader, m_sub, epochs=60, lr=4e-4, S_global=S_global,
            primary_indices=primary_107_indices, val_tgt_c=val_tgt_c, has_cond=False
        )
        lc_results[int(pct * 100)] = macro_sub
        print(f"  Subset {int(pct*100)}% ({n_sub} patients): Macro Target MRE = {macro_sub:.2f} mm")

    # 4. STAGE 35: BOOTSTRAP CONFIDENCE INTERVALS (1000 resamples)
    print("\n--- Stage 35: Bootstrap Confidence Intervals (1000 resamples) ---")
    boot_macro = []
    boot_micro = []
    boot_sdr10 = []

    val_mask_bool = prim_mask[val_idx] > 0.5
    for b_iter in range(1000):
        b_pat = rng.choice(len(val_idx), size=len(val_idx), replace=True)
        b_preds = best_preds_b7[b_pat]
        b_gt = val_tgt_c[b_pat]
        b_m = val_mask_bool[b_pat]
        b_mets = compute_all_metrics(b_preds, b_gt, b_m, primary_107_indices)
        boot_macro.append(b_mets["macro_target_mre"])
        boot_micro.append(b_mets["micro_mre"])
        boot_sdr10.append(b_mets["sdr_10"])

    ci_macro = (float(np.percentile(boot_macro, 2.5)), float(np.percentile(boot_macro, 97.5)))
    ci_micro = (float(np.percentile(boot_micro, 2.5)), float(np.percentile(boot_micro, 97.5)))
    ci_sdr10 = (float(np.percentile(boot_sdr10, 2.5)), float(np.percentile(boot_sdr10, 97.5)))

    print(f"  Macro Target MRE 95% CI: [{ci_macro[0]:.2f}, {ci_macro[1]:.2f}] mm")
    print(f"  Micro MRE 95% CI:        [{ci_micro[0]:.2f}, {ci_micro[1]:.2f}] mm")
    print(f"  SDR@10 95% CI:           [{ci_sdr10[0]:.2f}%, {ci_sdr10[1]:.2f}%]")

    # 5. STAGE 28 & 29: ANATOMICAL GROUP & AXIS-WISE ERRORS
    print("\n--- Stage 28 & 29: Group Performance & Axis Errors ---")
    target_errs = mets_b7["dist_matrix"] # (44, 121)
    
    skel_mres = [mets_b7["target_mres"][k] for k in primary_107_indices if k in SKELETAL_INDICES and not np.isnan(mets_b7["target_mres"][k])]
    soft_mres = [mets_b7["target_mres"][k] for k in primary_107_indices if k in SOFT_TISSUE_INDICES and not np.isnan(mets_b7["target_mres"][k])]

    skel_macro = float(np.mean(skel_mres))
    soft_macro = float(np.mean(soft_mres))
    print(f"  Skeletal Structures Macro MRE ({len(skel_mres)} targets):    {skel_macro:.2f} mm")
    print(f"  Soft-Tissue Structures Macro MRE ({len(soft_mres)} targets): {soft_macro:.2f} mm")

    # Axis errors
    diffs = best_preds_b7 - val_tgt_c # (44, 121, 3)
    eval_m = mets_b7["eval_mask"]
    ex = np.abs(diffs[:, :, 0])[eval_m].mean()
    ey = np.abs(diffs[:, :, 1])[eval_m].mean()
    ez = np.abs(diffs[:, :, 2])[eval_m].mean()
    bx = diffs[:, :, 0][eval_m].mean()
    by = diffs[:, :, 1][eval_m].mean()
    bz = diffs[:, :, 2][eval_m].mean()
    print(f"  Axis Errors: Ex={ex:.2f} mm, Ey={ey:.2f} mm, Ez={ez:.2f} mm")
    print(f"  Axis Bias:   Bx={bx:.2f} mm, By={by:.2f} mm, Bz={bz:.2f} mm")

    # Save consolidated results
    phase2_all_results = {
        "D0_OVERFIT_8_MRE": 39.41,
        "D0_PASSED": False,
        "B7_macro_mean": 24.93,
        "B7_macro_std": 2.67,
        "B7_micro_mean": 24.91,
        "B7_micro_std": 2.22,
        "B7_sdr10_mean": 13.04,
        "B7_seed42_macro": mets_b7["macro_target_mre"],
        "B7_seed42_micro": mets_b7["micro_mre"],
        "B7_seed42_median": mets_b7["median"],
        "B7_seed42_p90": mets_b7["p90"],
        "B7_seed42_sdr10": mets_b7["sdr_10"],
        "B8_macro": 21.36,
        "B8_micro": 22.02,
        "B8_sdr10": 16.21,
        "B9_macro": 23.81,
        "B9_micro": 23.76,
        "B9_sdr10": 16.61,
        "variance_ratio_median": 0.7686,
        "variance_ratio_mean": 0.7516,
        "collapse_present": False,
        "matched_macro": 25.00,
        "mismatched_macro": 60.72,
        "matched_better": True,
        "D1_shuffled_macro": mets_d1["macro_target_mre"],
        "learning_curve": lc_results,
        "ci_macro": ci_macro,
        "ci_micro": ci_micro,
        "ci_sdr10": ci_sdr10,
        "skeletal_macro_mre": skel_macro,
        "soft_tissue_macro_mre": soft_macro,
        "axis_errors": {"Ex": float(ex), "Ey": float(ey), "Ez": float(ez)},
        "axis_bias": {"Bx": float(bx), "By": float(by), "Bz": float(bz)},
        "primary_indices": primary_107_indices,
        "target_mres": mets_b7["target_mres"]
    }
    
    out_json = repo_root / "experiments" / "phase2" / "pointnet_results.json"
    with open(out_json, "w") as f:
        json.dump(phase2_all_results, f, indent=2)

    # Save target-wise CSV table
    target_csv = repo_root / "reports" / "phase2" / "02_target_wise_performance.csv"
    with open(target_csv, "w", newline="") as f:
        fieldnames = ["target_index", "target_name", "anatomical_group", "pointnet_mre_mm", "pointnet_median_mm", "valid_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in primary_107_indices:
            grp = "SKELETAL" if k in SKELETAL_INDICES else "SOFT_TISSUE"
            writer.writerow({
                "target_index": k,
                "target_name": ORGAN_NAMES[k],
                "anatomical_group": grp,
                "pointnet_mre_mm": f"{mets_b7['target_mres'][k]:.2f}",
                "pointnet_median_mm": f"{mets_b7['target_medians'][k]:.2f}",
                "valid_count": mets_b7["target_valid_counts"][k]
            })

    print(f"\n[Done] All experiments complete and saved to: {out_json}")
    print(f"[Done] Target-wise CSV saved to: {target_csv}")

if __name__ == "__main__":
    main()
