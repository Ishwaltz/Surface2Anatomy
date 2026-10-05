import os
import sys
import copy
import json
import csv
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

from sharon.anatomical_prior import MaskedLowRankAnatomyModel
from sharon.model_prior_hybrid import SurfaceToLatentModel, HybridPriorLocalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Phase6Dataset(Dataset):
    def __init__(self, pts_m, tgt_mm, prim_mask, meta_norm, body_dims_mm, z_stars=None, augment=False):
        self.pts_m = torch.tensor(pts_m, dtype=torch.float32)
        self.tgt_mm = torch.tensor(tgt_mm, dtype=torch.float32)
        self.prim = torch.tensor(prim_mask, dtype=torch.float32)
        self.meta = torch.tensor(meta_norm, dtype=torch.float32)
        self.b_dims = torch.tensor(body_dims_mm, dtype=torch.float32)
        self.z_stars = torch.tensor(z_stars, dtype=torch.float32) if z_stars is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.pts_m)

    def __getitem__(self, idx):
        pts = self.pts_m[idx].clone()
        tgt = self.tgt_mm[idx].clone()
        if self.augment:
            jitter = torch.randn_like(pts) * 0.005
            pts = pts + jitter
        item = {
            "pts": pts,
            "tgt": tgt,
            "prim": self.prim[idx],
            "meta": self.meta[idx],
            "b_dims": self.b_dims[idx]
        }
        if self.z_stars is not None:
            item["z_star"] = self.z_stars[idx]
        return item

def compute_mre_loss(preds_mm, targets_mm, mask, eps_sq=1.0):
    diff = torch.sum((preds_mm - targets_mm) ** 2, dim=-1) # (B, K)
    dist = torch.sqrt(diff + eps_sq)
    return (dist * mask).sum() / mask.sum().clamp(min=1.0)

def train_a3_model(
    tr_loader, val_loader, model, epochs=45, lr=3e-4, lambda_z=0.02,
    val_tgt_mm=None, val_mask=None, primary_indices=None, model_name="A3_PriorOnly"
):
    enc_params = list(model.encoder.parameters())
    mlp_params = list(model.latent_mlp.parameters())
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": lr * 0.5},
        {"params": mlp_params, "lr": lr}
    ], weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_macro = 999.0
    best_preds = None
    best_state = None

    print(f"  Training {model_name} ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        for b in tr_loader:
            pts = b["pts"].to(device)
            tgt = b["tgt"].to(device)
            prim = b["prim"].to(device)
            meta = b["meta"].to(device)
            b_dims = b["b_dims"].to(device)
            z_gt = b.get("z_star", None)
            if z_gt is not None:
                z_gt = z_gt.to(device)

            opt.zero_grad()
            out = model(pts, metadata=meta, body_dimensions=b_dims)
            l_mre = compute_mre_loss(out["p_final"], tgt, prim)

            if z_gt is not None:
                l_z = F.mse_loss(out["z_hat"], z_gt)
                loss = l_mre + lambda_z * l_z
            else:
                loss = l_mre

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Validation
        model.eval()
        v_preds = []
        with torch.no_grad():
            for b in val_loader:
                pts = b["pts"].to(device)
                meta = b["meta"].to(device)
                b_dims = b["b_dims"].to(device)
                out = model(pts, metadata=meta, body_dimensions=b_dims)
                v_preds.append(out["p_final"].cpu().numpy())

        preds_arr = np.concatenate(v_preds, axis=0) # (44, 107, 3)
        mets = compute_all_metrics(preds_arr, val_tgt_mm, val_mask)
        macro_v = mets["macro_target_mre"]

        if macro_v < best_macro:
            best_macro = macro_v
            best_preds = preds_arr.copy()
            best_state = copy.deepcopy(model.state_dict())

    print(f"  -> Best Validation Macro MRE: {best_macro:.2f} mm")
    if best_state is not None:
        model.load_state_dict(best_state)
    best_mets = compute_all_metrics(best_preds, val_tgt_mm, val_mask)
    return best_mets, best_preds, model


def train_a4_model(
    tr_loader, val_loader, model, epochs=50, lr_enc=2e-4, lr_dec=4e-4, lambda_z=0.02,
    val_tgt_mm=None, val_mask=None, primary_indices=None, model_name="A4_Hybrid"
):
    enc_params = list(model.encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": lr_enc},
        {"params": other_params, "lr": lr_dec}
    ], weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_macro = 999.0
    best_preds = None
    best_state = None

    print(f"  Training {model_name} ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        for b in tr_loader:
            pts = b["pts"].to(device)
            tgt = b["tgt"].to(device)
            prim = b["prim"].to(device)
            meta = b["meta"].to(device)
            b_dims = b["b_dims"].to(device)
            z_gt = b.get("z_star", None)
            if z_gt is not None:
                z_gt = z_gt.to(device)

            opt.zero_grad()
            out = model(pts, metadata=meta, body_dimensions=b_dims)
            l_final = compute_mre_loss(out["p_final"], tgt, prim)
            l_prior = compute_mre_loss(out["p_prior"], tgt, prim)

            loss = l_final + 0.3 * l_prior
            if z_gt is not None:
                l_z = F.mse_loss(out["z_hat"], z_gt)
                loss = loss + lambda_z * l_z

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Validation
        model.eval()
        v_preds = []
        with torch.no_grad():
            for b in val_loader:
                pts = b["pts"].to(device)
                meta = b["meta"].to(device)
                b_dims = b["b_dims"].to(device)
                out = model(pts, metadata=meta, body_dimensions=b_dims)
                v_preds.append(out["p_final"].cpu().numpy())

        preds_arr = np.concatenate(v_preds, axis=0) # (44, 107, 3)
        mets = compute_all_metrics(preds_arr, val_tgt_mm, val_mask)
        macro_v = mets["macro_target_mre"]

        if macro_v < best_macro:
            best_macro = macro_v
            best_preds = preds_arr.copy()
            best_state = copy.deepcopy(model.state_dict())

    print(f"  -> Best Validation Macro MRE: {best_macro:.2f} mm")
    if best_state is not None:
        model.load_state_dict(best_state)
    best_mets = compute_all_metrics(best_preds, val_tgt_mm, val_mask)
    return best_mets, best_preds, model


def main():
    print("=" * 80)
    print("PHASE 6 - PARTS N & O: TRAIN PRIOR-ONLY (A3) & HYBRID (A4) MODELS")
    print("=" * 80)

    # 1. Load Data and Prior Basis
    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
    tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"
    basis_path = repo_root / "experiments" / "phase6" / "anatomical_prior_basis.pt"
    oracle_res_path = repo_root / "experiments" / "phase6" / "latent_search_and_oracle_results.json"

    if not basis_path.exists():
        print(f"Error: Anatomical prior basis not found at {basis_path}. Run step1 first.")
        sys.exit(1)

    data = torch.load(str(pt_path), weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)
    with open(tiers_path) as f:
        primary_107_indices = [int(r["target_index"]) for r in csv.DictReader(f) if r["tier"] in ["TIER_A", "TIER_B"]]

    K_prim = len(primary_107_indices)
    tr_idx = np.array(splits["train_indices"])
    val_idx = np.array(splits["val_indices"])

    # Load prior basis
    prior_data = torch.load(str(basis_path), weights_only=False)
    A_raw = prior_data["A_mean_mm"]
    A_mean = torch.tensor(A_raw, dtype=torch.float32).view(K_prim, 3) # (107, 3)
    U_basis = torch.tensor(prior_data["U_basis"], dtype=torch.float32)   # (321, d)
    latent_dim = prior_data["latent_dim"]

    # Compute ground truth latent codes z* for train and val
    model_prior = MaskedLowRankAnatomyModel(num_targets=K_prim, latent_dim=latent_dim)
    model_prior.A = prior_data["A_mean_mm"]
    model_prior.U = prior_data["U_basis"]

    tgt_all_mm = data["targets_centered_mm"].numpy()[:, primary_107_indices, :] # (440, 107, 3)
    prim_mask_all = data["target_primary_mask"].numpy()[:, primary_107_indices]  # (440, 107)

    tr_z_stars, _ = model_prior.project_ground_truth(tgt_all_mm[tr_idx], prim_mask_all[tr_idx])
    val_z_stars, _ = model_prior.project_ground_truth(tgt_all_mm[val_idx], prim_mask_all[val_idx])

    # Extract features
    pts_m = data["points_model"].numpy()
    body_dims = data["body_dimensions_mm"].numpy()
    sex = data["sex"].numpy()
    man_csv = repo_root / "sharon" / "dataset_v2" / "manifest_v2.csv"
    areas = np.zeros(len(sex), dtype=np.float32)
    volumes = np.zeros(len(sex), dtype=np.float32)
    with open(man_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            areas[i] = float(row["surface_area_mm2"])
            volumes[i] = float(row["body_volume_liters"])

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

    tr_ds = Phase6Dataset(
        pts_m[tr_idx], tgt_all_mm[tr_idx], prim_mask_all[tr_idx],
        meta_norm[tr_idx], body_dims[tr_idx], z_stars=tr_z_stars, augment=True
    )
    val_ds = Phase6Dataset(
        pts_m[val_idx], tgt_all_mm[val_idx], prim_mask_all[val_idx],
        meta_norm[val_idx], body_dims[val_idx], z_stars=val_z_stars, augment=False
    )

    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    val_tgt_mm = tgt_all_mm[val_idx]
    val_mask = prim_mask_all[val_idx]

    seeds = [42, 43, 44]
    a3_results_by_seed = {}
    a4_results_by_seed = {}
    a3_preds_by_seed = {}
    a4_preds_by_seed = {}

    # 2. Train Model A3 (Prior Only) across 3 seeds
    print("\n" + "=" * 50)
    print("TRAINING MODEL A3: SURFACE-TO-LATENT PRIOR ONLY")
    print("=" * 50)

    for s in seeds:
        print(f"\n--- Model A3 (Seed {s}) ---")
        torch.manual_seed(s)
        np.random.seed(s)
        m_a3 = SurfaceToLatentModel(
            A_mean=A_mean, U_basis=U_basis, latent_dim=latent_dim, use_metadata=True
        ).to(device)

        mets, preds, trained_m = train_a3_model(
            tr_loader, val_loader, m_a3, epochs=40, lr=3e-4, lambda_z=0.02,
            val_tgt_mm=val_tgt_mm, val_mask=val_mask, primary_indices=primary_107_indices,
            model_name=f"A3-PriorOnly-Seed{s}"
        )
        a3_results_by_seed[s] = clean_dict(mets)
        a3_preds_by_seed[s] = preds

        # Save checkpoint
        ckpt_p = repo_root / "experiments" / "phase6" / f"a3_prior_only_seed{s}.pt"
        torch.save({"state_dict": trained_m.state_dict(), "latent_dim": latent_dim}, str(ckpt_p))
        print(f"Seed {s} -> Macro MRE: {mets['macro_target_mre']:.2f} mm | SDR@10: {mets['sdr_10']:.2f}% | P90: {mets['p90']:.2f} mm")

    # 3. Train Model A4 (Hybrid Prior + Local Residual) across 3 seeds
    print("\n" + "=" * 50)
    print("TRAINING MODEL A4: PHASE 4 + JOINT PRIOR HYBRID")
    print("=" * 50)

    for s in seeds:
        print(f"\n--- Model A4 (Seed {s}) ---")
        torch.manual_seed(s)
        np.random.seed(s)
        m_a4 = HybridPriorLocalModel(
            A_mean=A_mean, U_basis=U_basis, latent_dim=latent_dim,
            d_model=256, nhead=8, num_layers=4, use_metadata=True
        ).to(device)

        mets, preds, trained_m = train_a4_model(
            tr_loader, val_loader, m_a4, epochs=45, lr_enc=2e-4, lr_dec=4e-4, lambda_z=0.02,
            val_tgt_mm=val_tgt_mm, val_mask=val_mask, primary_indices=primary_107_indices,
            model_name=f"A4-Hybrid-Seed{s}"
        )
        a4_results_by_seed[s] = clean_dict(mets)
        a4_preds_by_seed[s] = preds

        ckpt_p = repo_root / "experiments" / "phase6" / f"a4_hybrid_seed{s}.pt"
        torch.save({"state_dict": trained_m.state_dict(), "latent_dim": latent_dim}, str(ckpt_p))
        print(f"Seed {s} -> Macro MRE: {mets['macro_target_mre']:.2f} mm | SDR@10: {mets['sdr_10']:.2f}% | P90: {mets['p90']:.2f} mm")

    # 4. Aggregate 3-Seed Statistics
    a3_macro_vals = [a3_results_by_seed[s]["macro_target_mre"] for s in seeds]
    a3_micro_vals = [a3_results_by_seed[s]["micro_mre"] for s in seeds]
    a3_sdr10_vals = [a3_results_by_seed[s]["sdr_10"] for s in seeds]
    a3_sdr15_vals = [a3_results_by_seed[s]["sdr_15"] for s in seeds]
    a3_p90_vals = [a3_results_by_seed[s]["p90"] for s in seeds]

    a4_macro_vals = [a4_results_by_seed[s]["macro_target_mre"] for s in seeds]
    a4_micro_vals = [a4_results_by_seed[s]["micro_mre"] for s in seeds]
    a4_sdr10_vals = [a4_results_by_seed[s]["sdr_10"] for s in seeds]
    a4_sdr15_vals = [a4_results_by_seed[s]["sdr_15"] for s in seeds]
    a4_p90_vals = [a4_results_by_seed[s]["p90"] for s in seeds]

    # Load A0 baseline
    with open(repo_root / "experiments" / "phase6" / "a0_baseline_reproduction_results.json") as f:
        a0_summary = json.load(f)
    a0_mean = a0_summary["macro_mean"]

    print("\n" + "=" * 80)
    print("PHASE 6 MULTI-SEED SUMMARY:")
    print(f"A0 (Phase 4 Base):       {a0_mean:.2f} ± {a0_summary['macro_std']:.2f} mm")
    print(f"A3 (Prior-Only A3):     {np.mean(a3_macro_vals):.2f} ± {np.std(a3_macro_vals):.2f} mm (Δ = {np.mean(a3_macro_vals) - a0_mean:+.2f} mm)")
    print(f"A4 (Hybrid A4):         {np.mean(a4_macro_vals):.2f} ± {np.std(a4_macro_vals):.2f} mm (Δ = {np.mean(a4_macro_vals) - a0_mean:+.2f} mm)")
    print("=" * 80)

    # 5. Paired Bootstrap Tests (Seed 42)
    # Need A0 predictions
    # Load A0 model for seed 42 to get exact preds
    from sharon.model_voting import TargetConditionedVotingModel
    from tools.phase3.run_phase3_experiments import Phase3Dataset

    ckpt_a0 = torch.load(str(repo_root / "experiments" / "phase6" / "a0_phase4_base_seed42.pt"), weights_only=False)
    m_a0 = TargetConditionedVotingModel(atlas_coords=ckpt_a0["atlas_coords"]).to(device)
    m_a0.load_state_dict(ckpt_a0["state_dict"])
    m_a0.eval()
    val_ds_a0 = Phase3Dataset(pts_m[val_idx], data["targets_model"].numpy()[val_idx], val_mask, meta_norm[val_idx], augment=False)
    val_ldr_a0 = DataLoader(val_ds_a0, batch_size=16, shuffle=False)
    a0_preds_42_l = []
    with torch.no_grad():
        for b in val_ldr_a0:
            p = b["pts"].to(device)
            m = b["meta"].to(device)
            out = m_a0(p, metadata=m)
            a0_preds_42_l.append((out["p_final"] * 500.0).cpu().numpy())
    a0_preds_42 = np.concatenate(a0_preds_42_l, axis=0)[:, primary_107_indices, :]

    boot_a0_vs_a3 = paired_bootstrap_test(a3_preds_by_seed[42], a0_preds_42, val_tgt_mm, val_mask, n_resamples=1000, seed=42)
    boot_a0_vs_a4 = paired_bootstrap_test(a4_preds_by_seed[42], a0_preds_42, val_tgt_mm, val_mask, n_resamples=1000, seed=42)

    print(f"\nA3 vs A0 Bootstrap: Diff = {boot_a0_vs_a3['mean_diff']:+.2f} mm (95% CI: [{boot_a0_vs_a3['ci_lower']:+.2f}, {boot_a0_vs_a3['ci_upper']:+.2f}]), p = {boot_a0_vs_a3['p_value']:.4f}")
    print(f"A4 vs A0 Bootstrap: Diff = {boot_a0_vs_a4['mean_diff']:+.2f} mm (95% CI: [{boot_a0_vs_a4['ci_lower']:+.2f}, {boot_a0_vs_a4['ci_upper']:+.2f}]), p = {boot_a0_vs_a4['p_value']:.4f}")

    # 6. Save JSON Results
    results_dict = {
        "seeds": seeds,
        "a3_results_by_seed": a3_results_by_seed,
        "a4_results_by_seed": a4_results_by_seed,
        "a3_summary": {
            "macro_mean": float(np.mean(a3_macro_vals)),
            "macro_std": float(np.std(a3_macro_vals)),
            "micro_mean": float(np.mean(a3_micro_vals)),
            "micro_std": float(np.std(a3_micro_vals)),
            "sdr10_mean": float(np.mean(a3_sdr10_vals)),
            "sdr15_mean": float(np.mean(a3_sdr15_vals)),
            "p90_mean": float(np.mean(a3_p90_vals))
        },
        "a4_summary": {
            "macro_mean": float(np.mean(a4_macro_vals)),
            "macro_std": float(np.std(a4_macro_vals)),
            "micro_mean": float(np.mean(a4_micro_vals)),
            "micro_std": float(np.std(a4_micro_vals)),
            "sdr10_mean": float(np.mean(a4_sdr10_vals)),
            "sdr15_mean": float(np.mean(a4_sdr15_vals)),
            "p90_mean": float(np.mean(a4_p90_vals))
        },
        "bootstrap_tests_seed42": {
            "a3_vs_a0": clean_dict(boot_a0_vs_a3),
            "a4_vs_a0": clean_dict(boot_a0_vs_a4)
        }
    }

    out_json = repo_root / "experiments" / "phase6" / "a3_a4_training_results.json"
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved training results to {out_json}")

    # Write report
    report_file = repo_root / "reports" / "phase6" / "03_prior_models_evaluation.md"
    with open(report_file, "w") as f:
        f.write(f"""# Phase 6 - Step 3: Prior-Only (A3) and Hybrid (A4) Models Evaluation

## Executive Summary

- **A0 Baseline (Phase 4 Base):** {a0_mean:.2f} ± {a0_summary['macro_std']:.2f} mm
- **A3 Surface-to-Latent Prior-Only:** {np.mean(a3_macro_vals):.2f} ± {np.std(a3_macro_vals):.2f} mm (Δ = {np.mean(a3_macro_vals) - a0_mean:+.2f} mm)
- **A4 Phase 4 + Joint Prior Hybrid:** {np.mean(a4_macro_vals):.2f} ± {np.std(a4_macro_vals):.2f} mm (Δ = {np.mean(a4_macro_vals) - a0_mean:+.2f} mm)

## Bootstrap Hypothesis Tests (Seed 42)
- **A3 vs A0:** Mean Diff = {boot_a0_vs_a3['mean_diff']:+.2f} mm (95% CI: [{boot_a0_vs_a3['ci_lower']:+.2f}, {boot_a0_vs_a3['ci_upper']:+.2f}] mm), p = {boot_a0_vs_a3['p_value']:.4f}
- **A4 vs A0:** Mean Diff = {boot_a0_vs_a4['mean_diff']:+.2f} mm (95% CI: [{boot_a0_vs_a4['ci_lower']:+.2f}, {boot_a0_vs_a4['ci_upper']:+.2f}] mm), p = {boot_a0_vs_a4['p_value']:.4f}
""")
    print(f"Wrote report to {report_file}")

if __name__ == "__main__":
    main()
