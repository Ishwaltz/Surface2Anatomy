import sys
import json
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES
from tools.phase2.metrics import compute_all_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 80)
    print("PHASE 5 - STEP 1: RESIDUAL ORACLE & CONTROLS D0 (BIAS), D1 (COARSE MLP)")
    print("=" * 80)

    # 1. Load Data & Precomputed Coarse Outputs
    coarse_path = repo_root / "experiments" / "phase5" / "coarse_predictions_r0.pt"
    ckpt_path = repo_root / "experiments" / "phase5" / "r0_phase4_base_seed42.pt"
    tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"

    coarse_data = torch.load(str(coarse_path), weights_only=False)
    ckpt = torch.load(str(ckpt_path), weights_only=False)

    with open(tiers_path) as f:
        primary_107_indices = [int(r["target_index"]) for r in csv.DictReader(f) if r["tier"] in ["TIER_A", "TIER_B"]]

    val_p0_m = coarse_data["val_coarse"]["p_final"]   # (44, 121, 3) in model coords
    tr_p0_m = coarse_data["tr_coarse"]["p_final"]     # (352, 121, 3) in model coords
    
    val_tgt_mm = coarse_data["val_tgt_c_mm"]          # (44, 121, 3) in centered mm
    tr_tgt_m = coarse_data["tr_tgt_m"]                # (352, 121, 3) in model coords
    
    tr_prim = coarse_data["tr_prim_mask"]             # (352, 121)
    val_prim = coarse_data["val_prim_mask"]           # (44, 121)
    S_global = coarse_data["S_global"]                # 500.0

    val_p0_mm = val_p0_m * S_global
    tr_p0_mm = tr_p0_m * S_global
    tr_tgt_mm = tr_tgt_m * S_global

    # -------------------------------------------------------------
    # STAGE 16: RESIDUAL ORACLE DIAGNOSTIC (ON VALIDATION)
    # -------------------------------------------------------------
    print("\n--- STAGE 16: Residual Oracle Diagnostic (Validation Set) ---")
    val_errors = np.linalg.norm(val_p0_mm - val_tgt_mm, axis=-1) # (44, 121)
    valid_val_errs = val_errors[val_prim > 0.5]

    oracle_stats = {
        "mean_mm": float(np.mean(valid_val_errs)),
        "median_mm": float(np.median(valid_val_errs)),
        "p75_mm": float(np.percentile(valid_val_errs, 75)),
        "p90_mm": float(np.percentile(valid_val_errs, 90)),
        "p95_mm": float(np.percentile(valid_val_errs, 95)),
        "max_mm": float(np.max(valid_val_errs))
    }
    print(f"  Coarse Residual Oracle: Mean = {oracle_stats['mean_mm']:.2f} mm | Median = {oracle_stats['median_mm']:.2f} mm | P75 = {oracle_stats['p75_mm']:.2f} mm | P90 = {oracle_stats['p90_mm']:.2f} mm | P95 = {oracle_stats['p95_mm']:.2f} mm")

    # -------------------------------------------------------------
    # STAGE 17 & 18: SIMPLE TARGET BIAS CORRECTION CONTROL (D0)
    # -------------------------------------------------------------
    print("\n--- STAGE 18: Control D0 - Static Target Bias Correction ---")
    # Compute mean coarse residual per target using TRAIN ONLY: b_k = E_train[tgt - p0]
    b_k_mm = np.zeros((121, 3), dtype=np.float32)
    b_k_mag = np.zeros(121, dtype=np.float32)
    p95_k_mm = np.zeros(121, dtype=np.float32)

    for k in range(121):
        mask_k = tr_prim[:, k] > 0.5
        if mask_k.sum() > 0:
            diff_k = tr_tgt_mm[mask_k, k] - tr_p0_mm[mask_k, k] # (N_k, 3)
            b_k_mm[k] = diff_k.mean(axis=0)
            b_k_mag[k] = np.linalg.norm(b_k_mm[k])
            p95_k_mm[k] = np.percentile(np.linalg.norm(diff_k, axis=-1), 95)
        else:
            p95_k_mm[k] = 30.0

    print(f"  Train Coarse Residual Bias magnitude across 107 primary targets: Mean = {b_k_mag[primary_107_indices].mean():.2f} mm | Max = {b_k_mag[primary_107_indices].max():.2f} mm")
    print(f"  Train Coarse P95 Error Bound R_k across 107 primary targets: Mean = {p95_k_mm[primary_107_indices].mean():.2f} mm | Max = {p95_k_mm[primary_107_indices].max():.2f} mm")

    # Apply D0 static bias correction on validation
    val_p_d0_mm = val_p0_mm + b_k_mm.reshape(1, 121, 3)
    mets_d0 = compute_all_metrics(val_p_d0_mm, val_tgt_mm, val_prim, primary_107_indices)
    sdr15_d0 = float((np.abs(val_p_d0_mm - val_tgt_mm) < 15.0).mean() * 100.0)

    print(f"  -> D0 Static Bias Macro MRE: {mets_d0['macro_target_mre']:.2f} mm | Micro: {mets_d0['micro_mre']:.2f} mm | SDR@10: {mets_d0['sdr_10']:.2f}% | SDR@15: {sdr15_d0:.2f}%")
    print(f"     (vs Phase 4 R0 Base: 17.11 mm)")

    # -------------------------------------------------------------
    # STAGE 19: SMALL MLP COARSE-ONLY RESIDUAL CONTROL (D1)
    # -------------------------------------------------------------
    print("\n--- STAGE 19: Control D1 - Coarse-Only Residual MLP (No Surface Support) ---")
    # D1 inputs for each target k: [p_k^0, a_k, (p_k^0 - a_k), (p_k^query - p_k^vote)] (12-D per target)
    atlas_m = ckpt["atlas_coords"].cpu().numpy() # (121, 3)
    atlas_mm = atlas_m * S_global

    def construct_d1_features(p0_mm, coarse_dict):
        B = p0_mm.shape[0]
        # p0_mm: (B, 121, 3)
        diff_atlas = p0_mm - atlas_mm.reshape(1, 121, 3)
        qv_diff = (coarse_dict["p_query"] - coarse_dict["p_vote"]) * S_global
        disp = coarse_dict["dispersion"][:, :, None] * S_global
        # Combine: (B, 121, 3 + 3 + 3 + 3 + 1) = 13 features
        feats = np.concatenate([
            p0_mm / 250.0,
            atlas_mm.reshape(1, 121, 3).repeat(B, axis=0) / 250.0,
            diff_atlas / 100.0,
            qv_diff / 50.0,
            disp / 50.0
        ], axis=-1)
        return torch.from_numpy(feats).float()

    tr_d1_x = construct_d1_features(tr_p0_mm, coarse_data["tr_coarse"]) # (352, 121, 13)
    val_d1_x = construct_d1_features(val_p0_mm, coarse_data["val_coarse"]) # (44, 121, 13)

    tr_d1_y = torch.from_numpy((tr_tgt_mm - tr_p0_mm) / S_global).float() # (352, 121, 3) in model space
    tr_d1_mask = torch.from_numpy(tr_prim).float() # (352, 121)

    class CoarseOnlyMLP(nn.Module):
        def __init__(self, in_dim=13, hidden=128):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, 3)
            )
        def forward(self, x):
            return self.mlp(x) # (B, 121, 3)

    torch.manual_seed(42)
    m_d1 = CoarseOnlyMLP().to(device)
    opt_d1 = torch.optim.AdamW(m_d1.parameters(), lr=1e-3, weight_decay=1e-4)

    d1_dataset = TensorDataset(tr_d1_x, tr_d1_y, tr_d1_mask)
    d1_loader = DataLoader(d1_dataset, batch_size=32, shuffle=True)

    for ep in range(60):
        m_d1.train()
        for bx, by, bm in d1_loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            opt_d1.zero_grad()
            pred_delta = m_d1(bx)
            loss = (torch.norm(pred_delta - by, dim=-1) * bm).sum() / bm.sum()
            loss.backward()
            opt_d1.step()

    m_d1.eval()
    with torch.no_grad():
        val_delta_m = m_d1(val_d1_x.to(device)).cpu().numpy()
        val_delta_mm = val_delta_m * S_global

    val_p_d1_mm = val_p0_mm + val_delta_mm
    mets_d1 = compute_all_metrics(val_p_d1_mm, val_tgt_mm, val_prim, primary_107_indices)
    sdr15_d1 = float((np.abs(val_p_d1_mm - val_tgt_mm) < 15.0).mean() * 100.0)

    print(f"  -> D1 Coarse-Only MLP Macro MRE: {mets_d1['macro_target_mre']:.2f} mm | Micro: {mets_d1['micro_mre']:.2f} mm | SDR@10: {mets_d1['sdr_10']:.2f}% | SDR@15: {sdr15_d1:.2f}%")
    print(f"     (vs Phase 4 R0 Base: 17.11 mm)")

    # Save results to experiments/phase5/step1_baseline_results.json
    results = {
        "oracle_stats": oracle_stats,
        "D0_bias_macro_mm": mets_d0["macro_target_mre"],
        "D0_bias_micro_mm": mets_d0["micro_mre"],
        "D0_bias_sdr10": mets_d0["sdr_10"],
        "D0_bias_sdr15": sdr15_d0,
        "D1_coarse_mlp_macro_mm": mets_d1["macro_target_mre"],
        "D1_coarse_mlp_micro_mm": mets_d1["micro_mre"],
        "D1_coarse_mlp_sdr10": mets_d1["sdr_10"],
        "D1_coarse_mlp_sdr15": sdr15_d1,
        "R_k_p95_mm": p95_k_mm.tolist(),
        "b_k_mm": b_k_mm.tolist()
    }

    out_file = repo_root / "experiments" / "phase5" / "step1_baseline_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nStep 1 Baseline Results saved to: {out_file}")

if __name__ == "__main__":
    main()
