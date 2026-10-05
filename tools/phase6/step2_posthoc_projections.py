import os
import sys
import json
import csv
import time
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES
from tools.phase2.metrics import compute_all_metrics

def paired_bootstrap_test(preds_a, preds_b, targets, masks, n_resamples=1000, seed=42):
    np.random.seed(seed)
    N = len(preds_a)
    diffs = []
    for _ in range(n_resamples):
        idx = np.random.choice(N, size=N, replace=True)
        mets_a = compute_all_metrics(preds_a[idx], targets[idx], masks[idx])
        mets_b = compute_all_metrics(preds_b[idx], targets[idx], masks[idx])
        diffs.append(mets_a["macro_target_mre"] - mets_b["macro_target_mre"])
    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    p_val = 2.0 * min(float(np.mean(diffs >= 0)), float(np.mean(diffs <= 0)))
    p_val = min(1.0, max(p_val, 1.0 / n_resamples))
    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val
    }

def clean_dict(d):
    res = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            res[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            res[k] = int(v)
        elif isinstance(v, np.ndarray):
            res[k] = v.tolist()
        elif isinstance(v, dict):
            res[k] = clean_dict(v)
        else:
            res[k] = v
    return res

def main():
    print("=" * 80)
    print("PHASE 6 - PART M: POST-HOC MODEL-PROJECTION EVALUATION (A1 & A2)")
    print("=" * 80)

    # 1. Load Data and Prior Basis
    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
    tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"
    basis_path = repo_root / "experiments" / "phase6" / "anatomical_prior_basis.pt"
    a0_results_path = repo_root / "experiments" / "phase6" / "a0_baseline_reproduction_results.json"

    if not basis_path.exists():
        print(f"Error: Anatomical prior basis not found at {basis_path}. Run step1 first.")
        sys.exit(1)

    data = torch.load(str(pt_path), weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)
    with open(tiers_path) as f:
        primary_107_indices = [int(r["target_index"]) for r in csv.DictReader(f) if r["tier"] in ["TIER_A", "TIER_B"]]

    K_prim = len(primary_107_indices)
    val_idx = np.array(splits["val_indices"])
    val_tgt = data["targets_centered_mm"].numpy()[val_idx][:, primary_107_indices, :] # (44, 107, 3)
    val_mask = data["target_primary_mask"].numpy()[val_idx][:, primary_107_indices]   # (44, 107)

    prior_data = torch.load(str(basis_path), weights_only=False)
    A = prior_data["A_mean_mm"]
    A_3d = A.reshape(K_prim, 3)
    U = prior_data["U_basis"]   # (321, d)
    d = prior_data["latent_dim"]
    A_vec = A.reshape(-1)       # (321,)

    print(f"Loaded prior basis: K={K_prim}, d={d}, U shape={U.shape}")

    # Load A0 checkpoints to get validation predictions across seeds
    seeds = [42, 43, 44]
    a0_preds_per_seed = {}

    from sharon.model_voting import TargetConditionedVotingModel
    from tools.phase3.run_phase3_experiments import Phase3Dataset
    from torch.utils.data import DataLoader

    pts_m = data["points_model"].numpy()
    tgt_m = data["targets_model"].numpy()

    body_dims = data["body_dimensions_mm"].numpy()
    sex = data["sex"].numpy()
    man_csv = repo_root / "sharon" / "dataset_v2" / "manifest_v2.csv"
    areas = np.zeros(len(sex), dtype=np.float32)
    volumes = np.zeros(len(sex), dtype=np.float32)
    with open(man_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            areas[i] = float(row["surface_area_mm2"])
            volumes[i] = float(row["body_volume_liters"])

    tr_idx = np.array(splits["train_indices"])
    raw_meta = np.column_stack([
        body_dims[:, 0] / 500.0,
        body_dims[:, 1] / 500.0,
        body_dims[:, 2] / 500.0,
        areas / 1e5,
        volumes / 10.0,
        sex.astype(np.float32)
    ])
    tr_meta_mean = raw_meta[tr_idx].mean(axis=0)
    tr_meta_std = raw_meta[tr_idx].std(axis=0) + 1e-6
    meta_norm = (raw_meta - tr_meta_mean) / tr_meta_std
    S_global = 500.0

    val_ds = Phase3Dataset(pts_m[val_idx], tgt_m[val_idx], val_mask, meta_norm[val_idx], augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in seeds:
        ckpt_path = repo_root / "experiments" / "phase6" / f"a0_phase4_base_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"Error: Baseline checkpoint not found: {ckpt_path}")
            sys.exit(1)
        
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = TargetConditionedVotingModel(atlas_coords=ckpt["atlas_coords"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        val_preds_list = []
        with torch.no_grad():
            for b in val_loader:
                pts = b["pts"].to(device)
                m = b["meta"].to(device)
                out = model(pts, metadata=m)
                val_preds_list.append((out["p_final"] * S_global).cpu().numpy())

        p_full = np.concatenate(val_preds_list, axis=0) # (44, 121, 3)
        p_prim = p_full[:, primary_107_indices, :]     # (44, 107, 3)
        a0_preds_per_seed[seed] = p_prim

    # 2. Evaluate Hard Projection (A1) and Soft Projection (A2)
    # Hard projection:
    # y = vec(P0 - A) -> z_hat = U^T y -> P_proj = A + reshape(U z_hat)
    a1_preds_per_seed = {}
    a2_preds_per_seed = {}
    results_by_seed = {}

    # Grid of gating weights g
    gate_values = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]

    for seed in seeds:
        P0 = a0_preds_per_seed[seed] # (44, 107, 3)
        N = P0.shape[0]

        # Center relative to training mean A
        y0 = (P0 - A_3d).reshape(N, -1) # (44, 321)
        # Project onto subspace: U has shape (321, d), U^T U = I
        z_hat = y0 @ U               # (44, d)
        y_proj = z_hat @ U.T         # (44, 321)
        P_proj = (y_proj.reshape(N, K_prim, 3)) + A_3d # (44, 107, 3)

        a1_preds_per_seed[seed] = P_proj

        # Evaluate A0
        mets_a0 = compute_all_metrics(P0, val_tgt, val_mask)
        # Evaluate A1 (Hard projection g = 1.0)
        mets_a1 = compute_all_metrics(P_proj, val_tgt, val_mask)

        # Evaluate soft projection across grid of g
        gate_metrics = {}
        for g in gate_values:
            P_soft = (1.0 - g) * P0 + g * P_proj
            m = compute_all_metrics(P_soft, val_tgt, val_mask)
            gate_metrics[f"g_{g:.2f}"] = {
                "g": g,
                "macro_target_mre": float(m["macro_target_mre"]),
                "micro_mre": float(m["micro_mre"]),
                "sdr_10": float(m["sdr_10"]),
                "sdr_15": float(m["sdr_15"]),
                "p90": float(m["p90"])
            }

        # Find best g for this seed
        best_g_key = min(gate_metrics.keys(), key=lambda k: gate_metrics[k]["macro_target_mre"])
        best_g = gate_metrics[best_g_key]["g"]
        P_best_soft = (1.0 - best_g) * P0 + best_g * P_proj
        a2_preds_per_seed[seed] = P_best_soft
        mets_a2_best = gate_metrics[best_g_key]

        # Also get standard g=0.5
        mets_a2_g50 = gate_metrics["g_0.50"]
        mets_a2_g25 = gate_metrics["g_0.25"]

        results_by_seed[seed] = {
            "a0": clean_dict(mets_a0),
            "a1_hard_proj": clean_dict(mets_a1),
            "a2_best_g": mets_a2_best,
            "a2_g0.25": mets_a2_g25,
            "a2_g0.50": mets_a2_g50,
            "gate_sweep": gate_metrics
        }

        print(f"Seed {seed}:")
        print(f"  A0 (Phase 4 Base):   Macro MRE: {mets_a0['macro_target_mre']:.2f} mm | SDR@10: {mets_a0['sdr_10']:.2f}% | P90: {mets_a0['p90']:.2f} mm")
        print(f"  A1 (Hard Proj g=1.0): Macro MRE: {mets_a1['macro_target_mre']:.2f} mm | SDR@10: {mets_a1['sdr_10']:.2f}% | P90: {mets_a1['p90']:.2f} mm")
        print(f"  A2 (Soft g={best_g:.2f}):   Macro MRE: {mets_a2_best['macro_target_mre']:.2f} mm | SDR@10: {mets_a2_best['sdr_10']:.2f}% | P90: {mets_a2_best['p90']:.2f} mm")

    # 3. Aggregate 3-seed statistics
    def agg_seeds(key_path):
        vals = []
        for s in seeds:
            cur = results_by_seed[s]
            for p in key_path:
                cur = cur[p]
            vals.append(cur)
        return float(np.mean(vals)), float(np.std(vals))

    a0_macro_mean, a0_macro_std = agg_seeds(["a0", "macro_target_mre"])
    a1_macro_mean, a1_macro_std = agg_seeds(["a1_hard_proj", "macro_target_mre"])
    a2_best_macro_mean, a2_best_macro_std = agg_seeds(["a2_best_g", "macro_target_mre"])
    a2_g50_macro_mean, a2_g50_macro_std = agg_seeds(["a2_g0.50", "macro_target_mre"])

    print("\n" + "=" * 80)
    print("3-SEED AGGREGATE RESULTS:")
    print(f"A0 (Phase 4 Base):       {a0_macro_mean:.2f} ± {a0_macro_std:.2f} mm")
    print(f"A1 (Hard Projection):   {a1_macro_mean:.2f} ± {a1_macro_std:.2f} mm (Δ = {a1_macro_mean - a0_macro_mean:+.2f} mm)")
    print(f"A2 (Soft Best-g):       {a2_best_macro_mean:.2f} ± {a2_best_macro_std:.2f} mm (Δ = {a2_best_macro_mean - a0_macro_mean:+.2f} mm)")
    print(f"A2 (Soft g=0.50):       {a2_g50_macro_mean:.2f} ± {a2_g50_macro_std:.2f} mm (Δ = {a2_g50_macro_mean - a0_macro_mean:+.2f} mm)")
    print("=" * 80)

    # 4. Statistical Tests (Bootstrap 1000 resamples on Seed 42 and pooled)
    print("\nRunning Paired Bootstrap Tests (1000 resamples)...")
    boot_a0_vs_a1 = paired_bootstrap_test(
        a1_preds_per_seed[42], a0_preds_per_seed[42], val_tgt, val_mask, n_resamples=1000, seed=42
    )
    boot_a0_vs_a2 = paired_bootstrap_test(
        a2_preds_per_seed[42], a0_preds_per_seed[42], val_tgt, val_mask, n_resamples=1000, seed=42
    )

    print(f"Seed 42 A1 vs A0: Mean Diff = {boot_a0_vs_a1['mean_diff']:+.2f} mm (95% CI: [{boot_a0_vs_a1['ci_lower']:+.2f}, {boot_a0_vs_a1['ci_upper']:+.2f}]), p = {boot_a0_vs_a1['p_value']:.4f}")
    print(f"Seed 42 A2 vs A0: Mean Diff = {boot_a0_vs_a2['mean_diff']:+.2f} mm (95% CI: [{boot_a0_vs_a2['ci_lower']:+.2f}, {boot_a0_vs_a2['ci_upper']:+.2f}]), p = {boot_a0_vs_a2['p_value']:.4f}")

    # 5. Save results
    final_output = {
        "seeds": seeds,
        "results_by_seed": results_by_seed,
        "aggregate": {
            "a0_macro_mre": {"mean": a0_macro_mean, "std": a0_macro_std},
            "a1_hard_proj_macro_mre": {"mean": a1_macro_mean, "std": a1_macro_std, "diff": a1_macro_mean - a0_macro_mean},
            "a2_best_macro_mre": {"mean": a2_best_macro_mean, "std": a2_best_macro_std, "diff": a2_best_macro_mean - a0_macro_mean},
            "a2_g50_macro_mre": {"mean": a2_g50_macro_mean, "std": a2_g50_macro_std, "diff": a2_g50_macro_mean - a0_macro_mean}
        },
        "bootstrap_tests_seed42": {
            "a1_vs_a0": clean_dict(boot_a0_vs_a1),
            "a2_vs_a0": clean_dict(boot_a0_vs_a2)
        }
    }

    out_file = repo_root / "experiments" / "phase6" / "a1_a2_projection_results.json"
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved projection results to {out_file}")

    # Write report
    report_file = repo_root / "reports" / "phase6" / "02_posthoc_projections.md"
    with open(report_file, "w") as f:
        f.write("# Phase 6 - Step 2: Post-Hoc Model-Projection Evaluation (A1 & A2)\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"- **A0 Baseline (Phase 4 Base):** {a0_macro_mean:.2f} ± {a0_macro_std:.2f} mm\n")
        f.write(f"- **A1 Hard Projection (g=1.0):** {a1_macro_mean:.2f} ± {a1_macro_std:.2f} mm (Δ = {a1_macro_mean - a0_macro_mean:+.2f} mm)\n")
        f.write(f"- **A2 Gated/Soft Projection (Best g):** {a2_best_macro_mean:.2f} ± {a2_best_macro_std:.2f} mm (Δ = {a2_best_macro_mean - a0_macro_mean:+.2f} mm)\n")
        f.write(f"- **A2 Gated Projection (g=0.50):** {a2_g50_macro_mean:.2f} ± {a2_g50_macro_std:.2f} mm (Δ = {a2_g50_macro_mean - a0_macro_mean:+.2f} mm)\n\n")
        f.write("## Statistical Analysis\n\n")
        f.write(f"- A1 vs A0: p-value = {boot_a0_vs_a1['p_value']:.4f}, 95% CI [{boot_a0_vs_a1['ci_lower']:+.2f}, {boot_a0_vs_a1['ci_upper']:+.2f}] mm\n")
        f.write(f"- A2 vs A0: p-value = {boot_a0_vs_a2['p_value']:.4f}, 95% CI [{boot_a0_vs_a2['ci_lower']:+.2f}, {boot_a0_vs_a2['ci_upper']:+.2f}] mm\n")
    print(f"Wrote report to {report_file}")

if __name__ == "__main__":
    main()
