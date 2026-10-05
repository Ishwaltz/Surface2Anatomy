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

from labels import ORGAN_NAMES, SKELETAL_INDICES, SOFT_TISSUE_INDICES
from tools.phase2.metrics import compute_all_metrics
from sharon.model_regional_prior import RegionalResidualPriorModel

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
        elif isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().tolist()
        elif isinstance(v, dict):
            res[k] = clean_dict(v)
        else:
            res[k] = v
    return res

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

class RegionalPriorDataset(Dataset):
    def __init__(self, queries, p_base, p_query, p_vote, tgt_mm, prim_mask, s_pred=None, s_gt=None):
        self.queries = torch.tensor(queries, dtype=torch.float32)
        self.p_base = torch.tensor(p_base, dtype=torch.float32)
        self.p_query = torch.tensor(p_query, dtype=torch.float32)
        self.p_vote = torch.tensor(p_vote, dtype=torch.float32)
        self.tgt_mm = torch.tensor(tgt_mm, dtype=torch.float32)
        self.prim = torch.tensor(prim_mask, dtype=torch.float32)
        self.s_pred = torch.tensor(s_pred, dtype=torch.float32) if s_pred is not None else None
        self.s_gt = torch.tensor(s_gt, dtype=torch.float32) if s_gt is not None else None

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        item = {
            "queries": self.queries[idx],
            "p_base": self.p_base[idx],
            "p_query": self.p_query[idx],
            "p_vote": self.p_vote[idx],
            "tgt_mm": self.tgt_mm[idx],
            "prim": self.prim[idx]
        }
        if self.s_pred is not None:
            item["s_pred"] = self.s_pred[idx]
        if self.s_gt is not None:
            item["s_gt"] = self.s_gt[idx]
        return item

def compute_coordinate_loss(p_final, tgt_mm, mask, delta_p=None, lambda_delta=1e-4):
    diff = torch.sum((p_final - tgt_mm) ** 2, dim=-1) # (B, K)
    dist = torch.sqrt(diff + 1.0)
    mre_loss = (dist * mask).sum() / mask.sum().clamp(min=1.0)
    if delta_p is not None:
        reg_loss = lambda_delta * torch.mean(torch.norm(delta_p, dim=-1))
        return mre_loss + reg_loss
    return mre_loss

def train_regional_model(
    tr_loader, val_loader, model, epochs=45, lr=3e-4,
    val_tgt_mm=None, val_mask=None, model_name="RegionalPrior"
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_macro = 999.0
    best_preds = None
    best_state = None

    print(f"  Training {model_name} ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        for b in tr_loader:
            q = b["queries"].to(device)
            p_b = b["p_base"].to(device)
            p_q = b["p_query"].to(device)
            p_v = b["p_vote"].to(device)
            tgt = b["tgt_mm"].to(device)
            prim = b["prim"].to(device)
            s_p = b.get("s_pred", None)
            if s_p is not None:
                s_p = s_p.to(device)

            opt.zero_grad()
            out = model(queries=q, p_base=p_b, p_query=p_q, p_vote=p_v, s_pred=s_p)
            loss = compute_coordinate_loss(out["p_final"], tgt, prim, delta_p=out["delta_p"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Validation step
        model.eval()
        v_preds = []
        with torch.no_grad():
            for b in val_loader:
                q = b["queries"].to(device)
                p_b = b["p_base"].to(device)
                p_q = b["p_query"].to(device)
                p_v = b["p_vote"].to(device)
                s_p = b.get("s_pred", None)
                if s_p is not None:
                    s_p = s_p.to(device)
                out = model(queries=q, p_base=p_b, p_query=p_q, p_vote=p_v, s_pred=s_p)
                v_preds.append(out["p_final"].cpu().numpy())

        preds_arr = np.concatenate(v_preds, axis=0)
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
    print("PHASE 7 - STAGES 14 TO 28: TRAIN REGIONAL PRIORS (H1, H2, D5, H3)")
    print("=" * 80)

    # 1. Load Extracted Features and Regional Bases
    feats_path = repo_root / "experiments" / "phase7" / "phase4_base_features.pt"
    basis_path = repo_root / "experiments" / "phase7" / "regional_basis_and_oracle_results.pt"

    feats = torch.load(str(feats_path), weights_only=False)
    basis_data = torch.load(str(basis_path), weights_only=False)

    primary_107_indices = feats["primary_107_indices"]
    K_prim = len(primary_107_indices)
    canonical_regions = basis_data["canonical_regions"]
    selected_dims = basis_data["selected_dims"]
    regional_bases = basis_data["regional_bases"]

    # Extract primary subset
    tr_q = feats["tr_reps"]["queries"][:, primary_107_indices, :]
    tr_pb = feats["tr_reps"]["p_final"][:, primary_107_indices, :]
    tr_pq = feats["tr_reps"]["p_query"][:, primary_107_indices, :]
    tr_pv = feats["tr_reps"]["p_vote"][:, primary_107_indices, :]
    tr_tgt = feats["tr_tgt_mm"][:, primary_107_indices, :]
    tr_mask = feats["tr_prim_mask"][:, primary_107_indices]

    val_q = feats["val_reps"]["queries"][:, primary_107_indices, :]
    val_pb = feats["val_reps"]["p_final"][:, primary_107_indices, :]
    val_pq = feats["val_reps"]["p_query"][:, primary_107_indices, :]
    val_pv = feats["val_reps"]["p_vote"][:, primary_107_indices, :]
    val_tgt = feats["val_tgt_mm"][:, primary_107_indices, :]
    val_mask = feats["val_prim_mask"][:, primary_107_indices]

    # Skeletal coordinates (indices in canonical_regions["skeletal"])
    skel_indices = canonical_regions["skeletal"]
    tr_skel_pred = tr_pb[:, skel_indices, :]
    val_skel_pred = val_pb[:, skel_indices, :]
    val_skel_gt = val_tgt[:, skel_indices, :]

    # Datasets
    # Standard (without skeleton)
    tr_ds_std = RegionalPriorDataset(tr_q, tr_pb, tr_pq, tr_pv, tr_tgt, tr_mask)
    val_ds_std = RegionalPriorDataset(val_q, val_pb, val_pq, val_pv, val_tgt, val_mask)

    # With skeletal conditioning
    tr_ds_skel = RegionalPriorDataset(tr_q, tr_pb, tr_pq, tr_pv, tr_tgt, tr_mask, s_pred=tr_skel_pred)
    val_ds_skel = RegionalPriorDataset(val_q, val_pb, val_pq, val_pv, val_tgt, val_mask, s_pred=val_skel_pred, s_gt=val_skel_gt)

    tr_ldr_std = DataLoader(tr_ds_std, batch_size=16, shuffle=True, drop_last=True)
    val_ldr_std = DataLoader(val_ds_std, batch_size=16, shuffle=False)

    tr_ldr_skel = DataLoader(tr_ds_skel, batch_size=16, shuffle=True, drop_last=True)
    val_ldr_skel = DataLoader(val_ds_skel, batch_size=16, shuffle=False)

    seeds = [42, 43, 44]

    # 2. Experiment H1: Regional Residual Prior (Frozen Base, No Gating, No Skeleton)
    print("\n" + "=" * 60)
    print("EXPERIMENT H1: REGIONAL RESIDUAL PRIOR (FROZEN BASE)")
    print("=" * 60)
    h1_results = {}
    h1_preds_by_seed = {}

    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        m_h1 = RegionalResidualPriorModel(
            regions=canonical_regions, regional_bases=regional_bases, regional_dims=selected_dims,
            use_skeletal_conditioning=False, use_gating=False
        ).to(device)

        mets, preds, trained_m = train_regional_model(
            tr_ldr_std, val_ldr_std, m_h1, epochs=40, lr=3e-4,
            val_tgt_mm=val_tgt, val_mask=val_mask, model_name=f"H1-Seed{s}"
        )
        h1_results[s] = clean_dict(mets)
        h1_preds_by_seed[s] = preds
        torch.save(trained_m.state_dict(), str(repo_root / "experiments" / "phase7" / f"h1_model_seed{s}.pt"))

    h1_macro_vals = [h1_results[s]["macro_target_mre"] for s in seeds]
    print(f"H1 Multi-Seed Aggregate: {np.mean(h1_macro_vals):.2f} ± {np.std(h1_macro_vals):.2f} mm")

    # 3. Experiment H2: Regional Prior + Predicted Skeleton
    print("\n" + "=" * 60)
    print("EXPERIMENT H2: REGIONAL PRIOR + PREDICTED SKELETON (S_pred)")
    print("=" * 60)
    h2_results = {}
    h2_preds_by_seed = {}

    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        m_h2 = RegionalResidualPriorModel(
            regions=canonical_regions, regional_bases=regional_bases, regional_dims=selected_dims,
            use_skeletal_conditioning=True, use_gating=False
        ).to(device)

        mets, preds, trained_m = train_regional_model(
            tr_ldr_skel, val_ldr_skel, m_h2, epochs=40, lr=3e-4,
            val_tgt_mm=val_tgt, val_mask=val_mask, model_name=f"H2-Seed{s}"
        )
        h2_results[s] = clean_dict(mets)
        h2_preds_by_seed[s] = preds
        torch.save(trained_m.state_dict(), str(repo_root / "experiments" / "phase7" / f"h2_model_seed{s}.pt"))

    h2_macro_vals = [h2_results[s]["macro_target_mre"] for s in seeds]
    print(f"H2 Multi-Seed Aggregate: {np.mean(h2_macro_vals):.2f} ± {np.std(h2_macro_vals):.2f} mm")

    # 4. Diagnostic D5: Ground-Truth Skeleton Regional Oracle (Upper Bound)
    print("\n" + "=" * 60)
    print("DIAGNOSTIC D5: GROUND-TRUTH SKELETON REGIONAL ORACLE")
    print("=" * 60)
    # Evaluate best H2 model (Seed 42) with s_gt instead of s_pred
    m_h2_best = RegionalResidualPriorModel(
        regions=canonical_regions, regional_bases=regional_bases, regional_dims=selected_dims,
        use_skeletal_conditioning=True, use_gating=False
    ).to(device)
    m_h2_best.load_state_dict(torch.load(str(repo_root / "experiments" / "phase7" / "h2_model_seed42.pt")))
    m_h2_best.eval()

    d5_preds_l = []
    with torch.no_grad():
        for b in val_ldr_skel:
            q = b["queries"].to(device)
            p_b = b["p_base"].to(device)
            s_gt = b["s_gt"].to(device)
            out = m_h2_best(queries=q, p_base=p_b, s_pred=s_gt)
            d5_preds_l.append(out["p_final"].cpu().numpy())

    d5_preds = np.concatenate(d5_preds_l, axis=0)
    d5_mets = compute_all_metrics(d5_preds, val_tgt, val_mask)
    print(f"D5 GT-Skeleton Oracle: Macro MRE = {d5_mets['macro_target_mre']:.2f} mm | SDR@10 = {d5_mets['sdr_10']:.2f}%")

    # 5. Experiment H3: Gated Regional Prior (H3)
    print("\n" + "=" * 60)
    print("EXPERIMENT H3: GATED REGIONAL PRIOR (CONSERVATIVE GATING)")
    print("=" * 60)
    h3_results = {}
    h3_preds_by_seed = {}

    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        m_h3 = RegionalResidualPriorModel(
            regions=canonical_regions, regional_bases=regional_bases, regional_dims=selected_dims,
            use_skeletal_conditioning=True, use_gating=True
        ).to(device)

        mets, preds, trained_m = train_regional_model(
            tr_ldr_skel, val_ldr_skel, m_h3, epochs=45, lr=2.5e-4,
            val_tgt_mm=val_tgt, val_mask=val_mask, model_name=f"H3-Seed{s}"
        )
        h3_results[s] = clean_dict(mets)
        h3_preds_by_seed[s] = preds
        torch.save(trained_m.state_dict(), str(repo_root / "experiments" / "phase7" / f"h3_model_seed{s}.pt"))

    h3_macro_vals = [h3_results[s]["macro_target_mre"] for s in seeds]
    h3_micro_vals = [h3_results[s]["micro_mre"] for s in seeds]
    h3_sdr10_vals = [h3_results[s]["sdr_10"] for s in seeds]
    h3_sdr15_vals = [h3_results[s]["sdr_15"] for s in seeds]
    h3_p90_vals = [h3_results[s]["p90"] for s in seeds]

    h0_base_mre = feats["h0_seed42_metrics"]["macro_target_mre"] # 17.33 mm
    print(f"\nH3 Gated Regional Multi-Seed Aggregate: {np.mean(h3_macro_vals):.2f} ± {np.std(h3_macro_vals):.2f} mm (Δ vs Base = {np.mean(h3_macro_vals) - h0_base_mre:+.2f} mm)")

    # 6. Statistical Significance Tests (Bootstrap 1000 resamples on Seed 42)
    boot_h1 = paired_bootstrap_test(h1_preds_by_seed[42], val_pb, val_tgt, val_mask, n_resamples=1000, seed=42)
    boot_h2 = paired_bootstrap_test(h2_preds_by_seed[42], val_pb, val_tgt, val_mask, n_resamples=1000, seed=42)
    boot_h3 = paired_bootstrap_test(h3_preds_by_seed[42], val_pb, val_tgt, val_mask, n_resamples=1000, seed=42)

    print(f"\nBootstrap H3 vs Base (Seed 42): Diff = {boot_h3['mean_diff']:+.2f} mm (95% CI: [{boot_h3['ci_lower']:+.2f}, {boot_h3['ci_upper']:+.2f}]), p = {boot_h3['p_value']:.4f}")

    # 7. Subgroup & Specific Organ Errors on Best Model (H3 Seed 42)
    best_h3_preds = h3_preds_by_seed[42]
    skel_prim = [i for i, idx in enumerate(primary_107_indices) if idx in SKELETAL_INDICES]
    soft_prim = [i for i, idx in enumerate(primary_107_indices) if idx in SOFT_TISSUE_INDICES]

    skel_errs_h3 = np.linalg.norm(best_h3_preds[:, skel_prim, :] - val_tgt[:, skel_prim, :], axis=-1)
    skel_mask = val_mask[:, skel_prim]
    skel_mre_h3 = float(np.mean(skel_errs_h3[skel_mask > 0]))

    soft_errs_h3 = np.linalg.norm(best_h3_preds[:, soft_prim, :] - val_tgt[:, soft_prim, :], axis=-1)
    soft_mask = val_mask[:, soft_prim]
    soft_mre_h3 = float(np.mean(soft_errs_h3[soft_mask > 0]))

    colon_idx_local = [i for i, idx in enumerate(primary_107_indices) if ORGAN_NAMES[idx] == "colon"]
    gb_idx_local = [i for i, idx in enumerate(primary_107_indices) if ORGAN_NAMES[idx] == "gallbladder"]

    colon_mre_h3 = float(np.mean(np.linalg.norm(best_h3_preds[:, colon_idx_local[0], :] - val_tgt[:, colon_idx_local[0], :], axis=-1)[val_mask[:, colon_idx_local[0]] > 0])) if colon_idx_local else 0.0
    gb_mre_h3 = float(np.mean(np.linalg.norm(best_h3_preds[:, gb_idx_local[0], :] - val_tgt[:, gb_idx_local[0], :], axis=-1)[val_mask[:, gb_idx_local[0]] > 0])) if gb_idx_local else 0.0

    print(f"H3 Skeletal MRE:     {skel_mre_h3:.2f} mm (Base: {feats['h0_seed42_metrics']['macro_target_mre']:.2f} mm)")
    print(f"H3 Soft-Tissue MRE:  {soft_mre_h3:.2f} mm")
    print(f"H3 Colon MRE:        {colon_mre_h3:.2f} mm")
    print(f"H3 Gallbladder MRE:  {gb_mre_h3:.2f} mm")

    # 8. Save All Training Results
    train_summary = {
        "seeds": seeds,
        "h1_results": h1_results,
        "h2_results": h2_results,
        "h3_results": h3_results,
        "d5_gt_skeleton_oracle": clean_dict(d5_mets),
        "h1_summary": {
            "macro_mean": float(np.mean(h1_macro_vals)),
            "macro_std": float(np.std(h1_macro_vals))
        },
        "h2_summary": {
            "macro_mean": float(np.mean(h2_macro_vals)),
            "macro_std": float(np.std(h2_macro_vals))
        },
        "h3_summary": {
            "macro_mean": float(np.mean(h3_macro_vals)),
            "macro_std": float(np.std(h3_macro_vals)),
            "micro_mean": float(np.mean(h3_micro_vals)),
            "micro_std": float(np.std(h3_micro_vals)),
            "sdr10_mean": float(np.mean(h3_sdr10_vals)),
            "sdr15_mean": float(np.mean(h3_sdr15_vals)),
            "p90_mean": float(np.mean(h3_p90_vals)),
            "skel_mre_seed42": skel_mre_h3,
            "soft_mre_seed42": soft_mre_h3,
            "colon_mre_seed42": colon_mre_h3,
            "gb_mre_seed42": gb_mre_h3
        },
        "bootstrap_seed42": {
            "h1_vs_base": clean_dict(boot_h1),
            "h2_vs_base": clean_dict(boot_h2),
            "h3_vs_base": clean_dict(boot_h3)
        }
    }

    out_file = repo_root / "experiments" / "phase7" / "regional_prior_training_results.json"
    with open(out_file, "w") as f:
        json.dump(train_summary, f, indent=2)
    print(f"\nSaved regional prior training summary to: {out_file}")

    # Write report
    rep_file = repo_root / "reports" / "phase7" / "04_regional_priors_evaluation.md"
    with open(rep_file, "w") as f:
        f.write(f"""# Phase 7: Regional Residual Priors Evaluation Report (H1, H2, D5, H3)

## 1. Multi-Seed Experimental Benchmark

| Model | Conditioning | Gating | Macro Target MRE (mm) | Micro MRE (mm) | SDR@10 (%) | SDR@15 (%) | P90 (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H_0$ Stable Base** | None | Fixed Base | {h0_base_mre:.2f} | 17.70 | 27.46 | 52.14 | 30.74 |
| **$H_1$ Regional Prior** | Query Context | None | {np.mean(h1_macro_vals):.2f} ± {np.std(h1_macro_vals):.2f} | — | — | — | — |
| **$H_2$ Regional + Skel** | Query + Pred Skel | None | {np.mean(h2_macro_vals):.2f} ± {np.std(h2_macro_vals):.2f} | — | — | — | — |
| **$D_5$ GT-Skel Oracle** | Query + GT Skel | None | **{d5_mets['macro_target_mre']:.2f}** | {d5_mets['micro_mre']:.2f} | {d5_mets['sdr_10']:.2f} | {d5_mets['sdr_15']:.2f} | {d5_mets['p90']:.2f} |
| **$H_3$ Gated Regional** | Query + Pred Skel | Conservative | **{np.mean(h3_macro_vals):.2f} ± {np.std(h3_macro_vals):.2f}** | **{np.mean(h3_micro_vals):.2f}** | **{np.mean(h3_sdr10_vals):.2f}** | **{np.mean(h3_sdr15_vals):.2f}** | **{np.mean(h3_p90_vals):.2f}** |

## 2. Statistical Bootstrap Hypothesis Tests (Seed 42)
- **$H_3$ vs $H_0$:** Diff = {boot_h3['mean_diff']:+.2f} mm (95% CI: [{boot_h3['ci_lower']:+.2f}, {boot_h3['ci_upper']:+.2f}] mm), $p = {boot_h3['p_value']:.4f}$

## 3. Subgroup Breakdown ($H_3$ Best Model)
- Skeletal Targets MRE: **{skel_mre_h3:.2f} mm**
- Soft-Tissue Viscera MRE: **{soft_mre_h3:.2f} mm**
- Colon MRE: **{colon_mre_h3:.2f} mm**
- Gallbladder MRE: **{gb_mre_h3:.2f} mm**
""")
    print(f"Generated regional priors report: {rep_file}")

if __name__ == "__main__":
    main()
