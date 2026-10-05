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
from sharon.anatomical_prior import MaskedLowRankAnatomyModel

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

def main():
    print("=" * 80)
    print("PHASE 6 - PARTS E, F, G, H, I, J: MASKED LOW-RANK PRIOR & ORACLE (D0)")
    print("=" * 80)

    # 1. Load Data
    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
    inner_path = repo_root / "experiments" / "phase4" / "inner_splits.json"
    tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"

    data = torch.load(str(pt_path), weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)
    with open(inner_path) as f:
        inner_splits = json.load(f)

    with open(tiers_path) as f:
        primary_107_indices = [int(r["target_index"]) for r in csv.DictReader(f) if r["tier"] in ["TIER_A", "TIER_B"]]

    K_prim = len(primary_107_indices)
    print(f"Number of primary evaluation targets: {K_prim}")

    # Subselect primary targets only for the anatomical deformation vector
    # targets_centered_mm: (440, 121, 3) in mm!
    tgt_all_mm = data["targets_centered_mm"].numpy()[:, primary_107_indices, :] # (440, 107, 3)
    prim_mask_all = data["target_primary_mask"].numpy()[:, primary_107_indices]  # (440, 107)

    tr_idx = np.array(splits["train_indices"])      # 352
    val_idx = np.array(splits["val_indices"])        # 44
    in_tr_idx = np.array(inner_splits["inner_train_indices"]) # 282
    in_dev_idx = np.array(inner_splits["inner_dev_indices"])   # 70

    print(f"Cohorts: Train={len(tr_idx)}, Val={len(val_idx)}, Inner-Train={len(in_tr_idx)}, Inner-Dev={len(in_dev_idx)}")

    # 2. Latent Dimension Grid Search on INNER_DEV (Part G)
    print("\n--- Part G: Latent Dimension Search on INNER_DEV ---")
    candidate_dims = [4, 8, 16, 24, 32, 48, 64]
    dim_results = {}

    in_tr_tgt = tgt_all_mm[in_tr_idx]
    in_tr_mask = prim_mask_all[in_tr_idx]

    in_dev_tgt = tgt_all_mm[in_dev_idx]
    in_dev_mask = prim_mask_all[in_dev_idx]

    for d_val in candidate_dims:
        t0 = time.time()
        model_d = MaskedLowRankAnatomyModel(num_targets=K_prim, latent_dim=d_val)
        fit_info = model_d.fit(in_tr_tgt, in_tr_mask, epochs=600, lr=0.03, lambda_z=1e-4, lambda_u=1e-4)

        # Evaluate Ground-Truth Reconstruction Oracle on INNER_DEV
        _, dev_oracle_preds = model_d.project_ground_truth(in_dev_tgt, in_dev_mask)

        # Compute metrics
        mets = compute_all_metrics(dev_oracle_preds, in_dev_tgt, in_dev_mask)
        elapsed = time.time() - t0

        dim_results[str(d_val)] = {
            "latent_dim": d_val,
            "params_count": 3 * K_prim * d_val,
            "dev_oracle_macro_mre": float(mets["macro_target_mre"]),
            "dev_oracle_micro_mre": float(mets["micro_mre"]),
            "dev_oracle_sdr10": float(mets["sdr_10"]),
            "dev_oracle_sdr15": float(mets["sdr_15"]),
            "dev_oracle_p90": float(mets["p90"]),
            "fit_loss": fit_info["final_rec_loss"],
            "fit_time_s": float(elapsed)
        }
        print(f"d = {d_val:2d} | Dev Oracle Macro MRE: {mets['macro_target_mre']:.2f} mm | SDR@10: {mets['sdr_10']:.2f}% | SDR@15: {mets['sdr_15']:.2f}% | P90: {mets['p90']:.2f} mm ({elapsed:.1f}s)")

    # Select optimal d on INNER_DEV (lowest macro MRE with parsimony)
    # Check if diminishing returns occur beyond a certain dimension
    best_d = min(candidate_dims, key=lambda d: dim_results[str(d)]["dev_oracle_macro_mre"])
    print(f"\nOptimal Latent Dimension from Inner-Dev: d* = {best_d} ({dim_results[str(best_d)]['dev_oracle_macro_mre']:.2f} mm)")

    # 3. Fit Final Masked Low-Rank Anatomical Basis on FULL TRAIN (352 Patients)
    print(f"\n--- Fitting Final Low-Rank Anatomical Model on 352 TRAIN Patients (d*={best_d}) ---")
    full_tr_tgt = tgt_all_mm[tr_idx]
    full_tr_mask = prim_mask_all[tr_idx]

    val_tgt = tgt_all_mm[val_idx]
    val_mask = prim_mask_all[val_idx]

    final_model = MaskedLowRankAnatomyModel(num_targets=K_prim, latent_dim=best_d)
    t0 = time.time()
    final_fit = final_model.fit(full_tr_tgt, full_tr_mask, epochs=800, lr=0.03, lambda_z=1e-4, lambda_u=1e-4)
    print(f"Final Model Fitted in {time.time() - t0:.1f}s (Final Loss: {final_fit['final_rec_loss']:.4f})")

    # 4. Evaluate D0 Ground-Truth Oracle on VALIDATION (44 Patients)
    print("\n--- Part H & I: Anatomical Reconstruction Oracle (D0) on 44 VALIDATION Patients ---")
    val_z_star, val_oracle_preds = final_model.project_ground_truth(val_tgt, val_mask)
    d0_mets = compute_all_metrics(val_oracle_preds, val_tgt, val_mask)

    print(f"D0 Latent Oracle Macro MRE: {d0_mets['macro_target_mre']:.2f} mm")
    print(f"D0 Latent Oracle Micro MRE: {d0_mets['micro_mre']:.2f} mm")
    print(f"D0 Latent Oracle SDR@5:     {d0_mets['sdr_5']:.2f}%")
    print(f"D0 Latent Oracle SDR@10:    {d0_mets['sdr_10']:.2f}%")
    print(f"D0 Latent Oracle SDR@15:    {d0_mets['sdr_15']:.2f}%")
    print(f"D0 Latent Oracle P90:       {d0_mets['p90']:.2f} mm")

    # Oracle Acceptance Criterion Check
    oracle_macro = d0_mets['macro_target_mre']
    if oracle_macro < 5.0:
        criterion_status = "EXCELLENT_CAPACITY (<5 mm)"
    elif oracle_macro <= 8.0:
        criterion_status = "STRONGLY_PROMISING (5-8 mm)"
    else:
        criterion_status = "INSUFFICIENT_LINEAR_CAPACITY (>10 mm)"
    print(f"Oracle Acceptance Criterion: {criterion_status}")

    # 5. Target-Wise Oracle Error Analysis (Part J)
    print("\n--- Part J: Target-Wise Oracle Error Analysis ---")
    oracle_errs = np.linalg.norm(val_oracle_preds - val_tgt, axis=-1) # (44, 107)

    target_oracle_stats = {}
    strongly_modeled = 0
    moderately_modeled = 0
    poorly_modeled = 0

    for idx, k in enumerate(primary_107_indices):
        t_name = ORGAN_NAMES[k]
        val_k_mask = val_mask[:, idx] > 0
        if not np.any(val_k_mask):
            continue
        errs_k = oracle_errs[val_k_mask, idx]
        mre_k = float(np.mean(errs_k))
        p90_k = float(np.percentile(errs_k, 90))

        # Population variance on train
        tr_k_mask = full_tr_mask[:, idx] > 0
        tr_pts_k = full_tr_tgt[tr_k_mask, idx]
        pop_std = float(np.mean(np.std(tr_pts_k, axis=0)))

        if mre_k < 5.0:
            cat = "STRONGLY_MODELLED_BY_PRIOR"
            strongly_modeled += 1
        elif mre_k <= 10.0:
            cat = "MODERATELY_MODELLED"
            moderately_modeled += 1
        else:
            cat = "POORLY_MODELLED"
            poorly_modeled += 1

        target_oracle_stats[t_name] = {
            "target_index": k,
            "train_support_count": int(tr_k_mask.sum()),
            "population_std_mm": pop_std,
            "oracle_mre_mm": mre_k,
            "oracle_p90_mm": p90_k,
            "classification": cat
        }

    print(f"Target Classification Summary: Strongly Modeled (<5mm): {strongly_modeled} | Moderately Modeled (5-10mm): {moderately_modeled} | Poorly Modeled (>10mm): {poorly_modeled}")

    # 6. Anatomical Mode Audit (Parts K & L)
    print("\n--- Parts K & L: Anatomical Deformation Mode Audit ---")
    # Singular values / variance per mode
    mode_variances = final_model.mode_stds ** 2
    var_explained_ratio = mode_variances / np.sum(mode_variances)

    # Check correlation of latent codes with body height, width, and z-extent (scan coverage)
    body_dims_val = data["body_dimensions_mm"].numpy()[val_idx] # (44, 3): [x_dim, y_dim, z_dim]
    # val_z_star has shape (44, best_d)
    corr_z_height = []
    corr_z_width = []
    corr_z_depth = []

    for r in range(best_d):
        corr_z_width.append(float(np.corrcoef(val_z_star[:, r], body_dims_val[:, 0])[0, 1]))
        corr_z_depth.append(float(np.corrcoef(val_z_star[:, r], body_dims_val[:, 1])[0, 1]))
        corr_z_height.append(float(np.corrcoef(val_z_star[:, r], body_dims_val[:, 2])[0, 1]))

    print(f"Leading Mode 1 Variance Ratio: {var_explained_ratio[0]*100:.1f}% | Corr with Torso Height: {corr_z_height[0]:.2f} | Corr with Width: {corr_z_width[0]:.2f}")
    if best_d > 1:
        print(f"Leading Mode 2 Variance Ratio: {var_explained_ratio[1]*100:.1f}% | Corr with Torso Height: {corr_z_height[1]:.2f} | Corr with Width: {corr_z_width[1]:.2f}")

    # Save anatomical prior model state
    save_dict = {
        "A_mean_mm": final_model.A,
        "U_basis": final_model.U,
        "mode_stds": final_model.mode_stds,
        "latent_dim": best_d,
        "primary_107_indices": primary_107_indices,
        "var_explained_ratio": var_explained_ratio
    }
    prior_ckpt_path = repo_root / "experiments" / "phase6" / "anatomical_prior_basis.pt"
    torch.save(save_dict, str(prior_ckpt_path))
    print(f"\nSaved anatomical prior basis to {prior_ckpt_path}")

    # Save full results json
    results_dict = {
        "dim_grid_search_inner_dev": dim_results,
        "best_latent_dim": best_d,
        "d0_validation_oracle": clean_dict(d0_mets),
        "oracle_acceptance_criterion": criterion_status,
        "target_classification_summary": {
            "strongly_modeled_count": strongly_modeled,
            "moderately_modeled_count": moderately_modeled,
            "poorly_modeled_count": poorly_modeled
        },
        "leading_modes_correlation": {
            "mode_variances_ratio": var_explained_ratio.tolist(),
            "corr_with_height": corr_z_height,
            "corr_with_width": corr_z_width,
            "corr_with_depth": corr_z_depth
        },
        "target_oracle_stats": target_oracle_stats
    }

    out_json = repo_root / "experiments" / "phase6" / "latent_search_and_oracle_results.json"
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved latent search and oracle results to {out_json}")

if __name__ == "__main__":
    main()
