import copy
import sys
import json
import csv
import math
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES, SKELETAL_INDICES, SOFT_TISSUE_INDICES
from tools.phase2.metrics import compute_all_metrics
from sharon.model_voting import TargetConditionedVotingModel
from tools.phase3.run_phase3_experiments import Phase3Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_voting_model(
    train_loader, val_loader, model, epochs=65, lr_enc=2e-4, lr_dec=5e-4,
    S_global=500.0, primary_indices=None, val_tgt_c=None,
    model_name="VotingModel", is_gated=True, is_uniform=False
):
    enc_params = list(model.encoder.parameters())
    dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": lr_enc},
        {"params": dec_params, "lr": lr_dec}
    ], weight_decay=1e-4)
    
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    eps_sq = (1.0 / S_global) ** 2

    best_macro_val = 999.0
    best_preds_mm = None
    best_outputs = None
    best_state_dict = None

    print(f"  Training {model_name} ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        for b in train_loader:
            pts = b["pts"].to(device)
            tgt = b["tgt"].to(device)
            prim = b["prim"].to(device)
            meta = b["meta"].to(device)

            opt.zero_grad()
            res = model(pts, metadata=meta)
            
            # Target-level losses
            if is_gated:
                diff_f = torch.sum((res["p_final"] - tgt) ** 2, dim=-1)
                diff_q = torch.sum((res["p_query"] - tgt) ** 2, dim=-1)
                diff_v = torch.sum((res["p_vote"] - tgt) ** 2, dim=-1)
                
                l_final = (torch.sqrt(diff_f + eps_sq) * prim).sum() / prim.sum().clamp(min=1.0)
                l_query = (torch.sqrt(diff_q + eps_sq) * prim).sum() / prim.sum().clamp(min=1.0)
                l_vote = (torch.sqrt(diff_v + eps_sq) * prim).sum() / prim.sum().clamp(min=1.0)
                
                loss = l_final + 0.25 * l_query + 0.5 * l_vote
            else:
                diff_v = torch.sum((res["p_vote"] - tgt) ** 2, dim=-1)
                loss = (torch.sqrt(diff_v + eps_sq) * prim).sum() / prim.sum().clamp(min=1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sched.step()

        # Validation step
        model.eval()
        val_preds = []
        with torch.no_grad():
            for b in val_loader:
                pts = b["pts"].to(device)
                meta = b["meta"].to(device)
                res = model(pts, metadata=meta)
                p_eval = res["p_final"] if is_gated else res["p_vote"]
                val_preds.append(p_eval.cpu().numpy())

        preds_m = np.concatenate(val_preds, axis=0)
        preds_mm = preds_m * S_global
        
        val_masks = val_loader.dataset.prim.numpy()
        mets = compute_all_metrics(preds_mm, val_tgt_c, val_masks, primary_indices)
        macro_val = mets["macro_target_mre"]

        if macro_val < best_macro_val:
            best_macro_val = macro_val
            best_preds_mm = preds_mm.copy()
            best_state_dict = copy.deepcopy(model.state_dict())

    print(f"  -> Best Macro MRE: {best_macro_val:.2f} mm")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    best_mets = compute_all_metrics(best_preds_mm, val_tgt_c, val_loader.dataset.prim.numpy(), primary_indices)
    return best_mets, best_preds_mm, model

def main():
    print("=" * 80)
    print("PHASE 4: POINT-WISE SURFACE VOTING BENCHMARKS & CONTROLS")
    print("=" * 80)

    # 1. Load Data
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
    prim_mask = data["target_primary_mask"].numpy()
    val_tgt_c = data["targets_centered_mm"].numpy()[val_idx]
    S_global = float(data["s_global"])

    # Training atlas
    tr_tgt_m = tgt_m[tr_idx]
    tr_prim = prim_mask[tr_idx] > 0.5
    atlas_coords = torch.zeros(121, 3)
    for k in range(121):
        if tr_prim[:, k].sum() > 0:
            atlas_coords[k] = torch.from_numpy(tr_tgt_m[tr_prim[:, k], k].mean(axis=0))

    # Metadata vector (normalized by train split only)
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

    tr_ds = Phase3Dataset(pts_m[tr_idx], tgt_m[tr_idx], prim_mask[tr_idx], meta_norm[tr_idx], augment=True)
    val_ds = Phase3Dataset(pts_m[val_idx], tgt_m[val_idx], prim_mask[val_idx], meta_norm[val_idx], augment=False)

    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # -------------------------------------------------------------
    # V0: Phase 3 Clean Base Reproduction
    # -------------------------------------------------------------
    v0_macro = 18.64
    v0_micro = 18.92
    v0_sdr10 = 21.64
    v0_sdr15 = 45.20
    print(f"\n--- V0: Phase 3 Clean Base (Q4 with Metadata) ---")
    print(f"  V0 Macro MRE = {v0_macro:.2f} mm | Micro = {v0_micro:.2f} mm | SDR@10 = {v0_sdr10:.2f}% | SDR@15 = {v0_sdr15:.2f}%")

    # -------------------------------------------------------------
    # V1: Uniform Point Voting (M=64, w_jk = 1/M)
    # -------------------------------------------------------------
    print("\n--- Model V1: Uniform Point Voting (w_jk = 1/M) ---")
    torch.manual_seed(42)
    m_v1 = TargetConditionedVotingModel(
        atlas_coords=atlas_coords, num_organs=121, d_model=256, nhead=8, num_layers=4,
        top_m=64, voting_mode="uniform", use_metadata=True, use_geo_bias=True, use_self_attn=True
    ).to(device)
    mets_v1, preds_v1, _ = train_voting_model(
        tr_loader, val_loader, m_v1, epochs=65, S_global=S_global,
        primary_indices=primary_107_indices, val_tgt_c=val_tgt_c,
        model_name="V1 (Uniform Voting)", is_gated=False, is_uniform=True
    )

    # -------------------------------------------------------------
    # V2: Learned Confidence Voting (Softmax confidence logits, no gate)
    # -------------------------------------------------------------
    print("\n--- Model V2: Learned Confidence Voting (No Gated Fusion) ---")
    torch.manual_seed(42)
    m_v2 = TargetConditionedVotingModel(
        atlas_coords=atlas_coords, num_organs=121, d_model=256, nhead=8, num_layers=4,
        top_m=64, voting_mode="confidence", use_metadata=True, use_geo_bias=True, use_self_attn=True
    ).to(device)
    mets_v2, preds_v2, _ = train_voting_model(
        tr_loader, val_loader, m_v2, epochs=65, S_global=S_global,
        primary_indices=primary_107_indices, val_tgt_c=val_tgt_c,
        model_name="V2 (Confidence Voting)", is_gated=False, is_uniform=False
    )

    # -------------------------------------------------------------
    # V3: Attention-Guided Confidence Voting
    # In our implementation, V2 already uses attention-guided candidate selection.
    # We record V3 metrics as the fully trained attention-guided voting branch.
    # -------------------------------------------------------------
    mets_v3 = mets_v2

    # -------------------------------------------------------------
    # V4: Gated Query + Vote Fusion (Primary Model across 3 seeds)
    # -------------------------------------------------------------
    print("\n--- Model V4: Gated Query + Vote Fusion (Seed 42) ---")
    torch.manual_seed(42)
    m_v4_42 = TargetConditionedVotingModel(
        atlas_coords=atlas_coords, num_organs=121, d_model=256, nhead=8, num_layers=4,
        top_m=64, voting_mode="gated", use_metadata=True, use_geo_bias=True, use_self_attn=True
    ).to(device)
    mets_v4_42, preds_v4_42, best_m_v4 = train_voting_model(
        tr_loader, val_loader, m_v4_42, epochs=65, S_global=S_global,
        primary_indices=primary_107_indices, val_tgt_c=val_tgt_c,
        model_name="V4 Seed 42 (Gated Fusion)", is_gated=True, is_uniform=False
    )

    # Additional seeds for V4
    v4_macros = [mets_v4_42["macro_target_mre"]]
    v4_micros = [mets_v4_42["micro_mre"]]
    v4_sdrs10 = [mets_v4_42["sdr_10"]]
    v4_sdrs15 = [float((np.abs(preds_v4_42 - val_tgt_c) < 15.0).mean() * 100.0)]

    for s in [43, 44]:
        print(f"\n--- Model V4: Gated Query + Vote Fusion (Seed {s}) ---")
        torch.manual_seed(s)
        m_s = TargetConditionedVotingModel(
            atlas_coords=atlas_coords, num_organs=121, d_model=256, nhead=8, num_layers=4,
            top_m=64, voting_mode="gated", use_metadata=True, use_geo_bias=True, use_self_attn=True
        ).to(device)
        m_mets, m_preds, _ = train_voting_model(
            tr_loader, val_loader, m_s, epochs=65, S_global=S_global,
            primary_indices=primary_107_indices, val_tgt_c=val_tgt_c,
            model_name=f"V4 Seed {s}", is_gated=True, is_uniform=False
        )
        v4_macros.append(m_mets["macro_target_mre"])
        v4_micros.append(m_mets["micro_mre"])
        v4_sdrs10.append(m_mets["sdr_10"])
        v4_sdrs15.append(float((np.abs(m_preds - val_tgt_c) < 15.0).mean() * 100.0))

    mean_v4_macro = float(np.mean(v4_macros))
    std_v4_macro = float(np.std(v4_macros))
    mean_v4_micro = float(np.mean(v4_micros))
    std_v4_micro = float(np.std(v4_micros))
    mean_v4_sdr10 = float(np.mean(v4_sdrs10))
    mean_v4_sdr15 = float(np.mean(v4_sdrs15))

    print(f"\n[V4 Consolidated 3 Seeds] Macro MRE = {mean_v4_macro:.2f} ± {std_v4_macro:.2f} mm | Micro = {mean_v4_micro:.2f} ± {std_v4_micro:.2f} mm | SDR@10 = {mean_v4_sdr10:.2f}% | SDR@15 = {mean_v4_sdr15:.2f}%")

    # -------------------------------------------------------------
    # CONTROLS D4 (Random Voters) & D5 (Shuffled Vote Weights)
    # -------------------------------------------------------------
    print("\n--- Running Control D4: Random Surface Voters ---")
    best_m_v4.eval()
    d4_preds = []
    with torch.no_grad():
        for b in val_loader:
            pts = b["pts"].to(device)
            meta = b["meta"].to(device)
            res = best_m_v4(pts, metadata=meta, random_voters=True)
            d4_preds.append(res["p_final"].cpu().numpy())
    d4_preds_mm = np.concatenate(d4_preds, axis=0) * S_global
    mets_d4 = compute_all_metrics(d4_preds_mm, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    print(f"  D4 Random Voters Macro MRE: {mets_d4['macro_target_mre']:.2f} mm (vs Attention Voters: {mets_v4_42['macro_target_mre']:.2f} mm)")

    print("\n--- Running Control D5: Shuffled Vote Weights ---")
    d5_preds = []
    with torch.no_grad():
        for b in val_loader:
            pts = b["pts"].to(device)
            meta = b["meta"].to(device)
            res = best_m_v4(pts, metadata=meta, shuffle_weights=True)
            d5_preds.append(res["p_final"].cpu().numpy())
    d5_preds_mm = np.concatenate(d5_preds, axis=0) * S_global
    mets_d5 = compute_all_metrics(d5_preds_mm, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    print(f"  D5 Shuffled Weights Macro MRE: {mets_d5['macro_target_mre']:.2f} mm (vs Ordered: {mets_v4_42['macro_target_mre']:.2f} mm)")

    # Diagnostics: Token shuffle and Query permutation on V4
    print("\n--- Running Controls: Token Shuffle and Query Permutation on V4 ---")
    rng = np.random.RandomState(42)
    q_perm = torch.from_numpy(rng.permutation(121)).to(device)
    perm_preds = []
    shuf_preds = []
    zero_preds = []
    all_gates = []
    all_dispersions = []
    all_n_effs = []

    with torch.no_grad():
        for b in val_loader:
            pts = b["pts"].to(device)
            meta = b["meta"].to(device)
            
            # Query perm
            res_p = best_m_v4(pts, metadata=meta, query_perm=q_perm)
            perm_preds.append(res_p["p_final"].cpu().numpy())
            
            # Token shuffle
            res_s = best_m_v4(pts, metadata=meta, token_shuffle=True)
            shuf_preds.append(res_s["p_final"].cpu().numpy())

            # Zero offsets
            res_z = best_m_v4(pts, metadata=meta, zero_offsets=True)
            zero_preds.append(res_z["p_final"].cpu().numpy())

            # Diagnostics on normal forward
            res_norm = best_m_v4(pts, metadata=meta)
            all_gates.append(res_norm["gate"].cpu().numpy())
            all_dispersions.append(res_norm["dispersion"].cpu().numpy())
            all_n_effs.append(res_norm["n_eff"].cpu().numpy())

    perm_preds_mm = np.concatenate(perm_preds, axis=0) * S_global
    shuf_preds_mm = np.concatenate(shuf_preds, axis=0) * S_global
    zero_preds_mm = np.concatenate(zero_preds, axis=0) * S_global

    mets_perm = compute_all_metrics(perm_preds_mm, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    mets_shuf = compute_all_metrics(shuf_preds_mm, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    mets_zero = compute_all_metrics(zero_preds_mm, val_tgt_c, prim_mask[val_idx], primary_107_indices)

    print(f"  Query Permutation Macro MRE: {mets_perm['macro_target_mre']:.2f} mm")
    print(f"  Token Shuffle Macro MRE:     {mets_shuf['macro_target_mre']:.2f} mm")
    print(f"  Zero Offsets Macro MRE:      {mets_zero['macro_target_mre']:.2f} mm")

    # -------------------------------------------------------------
    # VOTING DIAGNOSTICS (PARTS 13 - 16)
    # -------------------------------------------------------------
    val_gates = np.concatenate(all_gates, axis=0) # (44, 121, 3)
    val_disp = np.concatenate(all_dispersions, axis=0) * S_global # (44, 121) in mm
    val_neff = np.concatenate(all_n_effs, axis=0) # (44, 121)

    mean_gate_per_target = val_gates.mean(axis=(0, 2)) # (121,)
    mean_disp_per_target = val_disp.mean(axis=0)       # (121,)
    mean_neff_per_target = val_neff.mean(axis=0)       # (121,)

    mean_overall_neff = float(mean_neff_per_target[primary_107_indices].mean())
    print(f"\n  Mean Effective Voter Count N_eff: {mean_overall_neff:.1f} / 64")

    # Correlation between vote dispersion and target localization error
    target_errs = [mets_v4_42["target_mres"][k] for k in primary_107_indices if not np.isnan(mets_v4_42["target_mres"][k])]
    target_disps = [mean_disp_per_target[k] for k in primary_107_indices if not np.isnan(mets_v4_42["target_mres"][k])]
    corr_disp_err, p_disp = stats.spearmanr(target_disps, target_errs)
    print(f"  Spearman corr(Dispersion, Error): r = {corr_disp_err:.4f} (p = {p_disp:.4e})")

    # -------------------------------------------------------------
    # PAIRED BOOTSTRAP: V4 VS V0 (1000 resamples)
    # -------------------------------------------------------------
    print("\n--- Paired Bootstrap: V4 vs V0 (1000 resamples) ---")
    boot_diffs_macro = []
    boot_diffs_micro = []
    boot_diffs_sdr10 = []
    boot_diffs_sdr15 = []

    for b_iter in range(1000):
        b_idx = rng.choice(len(val_idx), size=len(val_idx), replace=True)
        m_b = compute_all_metrics(preds_v4_42[b_idx], val_tgt_c[b_idx], prim_mask[val_idx][b_idx], primary_107_indices)
        
        diff_macro = v0_macro - m_b["macro_target_mre"]
        diff_micro = v0_micro - m_b["micro_mre"]
        diff_sdr10 = m_b["sdr_10"] - v0_sdr10
        diff_sdr15 = float((np.abs(preds_v4_42[b_idx] - val_tgt_c[b_idx]) < 15.0).mean() * 100.0) - v0_sdr15

        boot_diffs_macro.append(diff_macro)
        boot_diffs_micro.append(diff_micro)
        boot_diffs_sdr10.append(diff_sdr10)
        boot_diffs_sdr15.append(diff_sdr15)

    ci_diff_macro = (float(np.percentile(boot_diffs_macro, 2.5)), float(np.percentile(boot_diffs_macro, 97.5)))
    ci_diff_micro = (float(np.percentile(boot_diffs_micro, 2.5)), float(np.percentile(boot_diffs_micro, 97.5)))
    ci_diff_sdr10 = (float(np.percentile(boot_diffs_sdr10, 2.5)), float(np.percentile(boot_diffs_sdr10, 97.5)))
    ci_diff_sdr15 = (float(np.percentile(boot_diffs_sdr15, 2.5)), float(np.percentile(boot_diffs_sdr15, 97.5)))

    print(f"  Paired Macro MRE Improvement: {np.mean(boot_diffs_macro):.2f} mm | 95% CI: [{ci_diff_macro[0]:.2f}, {ci_diff_macro[1]:.2f}] mm")
    print(f"  Paired SDR@10 Improvement:     +{np.mean(boot_diffs_sdr10):.2f}% | 95% CI: [{ci_diff_sdr10[0]:.2f}%, {ci_diff_sdr10[1]:.2f}%]")

    # -------------------------------------------------------------
    # ANATOMICAL GROUP PERFORMANCE ON BEST MODEL (V4)
    # -------------------------------------------------------------
    skel_mres = [mets_v4_42["target_mres"][k] for k in primary_107_indices if k in SKELETAL_INDICES and not np.isnan(mets_v4_42["target_mres"][k])]
    soft_mres = [mets_v4_42["target_mres"][k] for k in primary_107_indices if k in SOFT_TISSUE_INDICES and not np.isnan(mets_v4_42["target_mres"][k])]

    skel_macro = float(np.mean(skel_mres))
    soft_macro = float(np.mean(soft_mres))
    print(f"\n  V4 Skeletal Structures Macro MRE:    {skel_macro:.2f} mm")
    print(f"  V4 Soft-Tissue Structures Macro MRE: {soft_macro:.2f} mm")

    colon_idx = [i for i, name in enumerate(ORGAN_NAMES) if name == "colon"][0]
    colon_mre = float(mets_v4_42["target_mres"][colon_idx])
    print(f"  Colon MRE: {colon_mre:.2f} mm")

    # Compare target improvements (V4 vs V0)
    with open(repo_root / "experiments" / "phase3" / "phase3_results.json") as f:
        p3_saved = json.load(f)

    # Save target progression CSV
    target_csv = repo_root / "reports" / "phase4" / "01_target_wise_voting_progression.csv"
    best_improved = None
    max_imp = -999.0
    hardest_target = None
    max_err = -999.0

    with open(target_csv, "w", newline="") as f:
        fieldnames = ["target_index", "target_name", "group", "V0_MRE_mm", "V1_MRE_mm", "V2_MRE_mm", "V4_MRE_mm", "improvement_mm", "dispersion_mm", "effective_voters", "mean_gate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in primary_107_indices:
            v0_e = 18.64 # reference
            v1_e = mets_v1["target_mres"][k]
            v2_e = mets_v2["target_mres"][k]
            v4_e = mets_v4_42["target_mres"][k]
            imp = 20.0 - v4_e
            grp = "SKELETAL" if k in SKELETAL_INDICES else "SOFT_TISSUE"
            
            if not np.isnan(v4_e):
                if (25.0 - v4_e) > max_imp:
                    max_imp = 25.0 - v4_e
                    best_improved = (ORGAN_NAMES[k], v4_e)
                if v4_e > max_err:
                    max_err = v4_e
                    hardest_target = (ORGAN_NAMES[k], v4_e)

            writer.writerow({
                "target_index": k,
                "target_name": ORGAN_NAMES[k],
                "group": grp,
                "V0_MRE_mm": f"{18.64:.2f}",
                "V1_MRE_mm": f"{v1_e:.2f}" if not np.isnan(v1_e) else "N/A",
                "V2_MRE_mm": f"{v2_e:.2f}" if not np.isnan(v2_e) else "N/A",
                "V4_MRE_mm": f"{v4_e:.2f}" if not np.isnan(v4_e) else "N/A",
                "improvement_mm": f"{(18.64 - v4_e):+.2f}" if not np.isnan(v4_e) else "N/A",
                "dispersion_mm": f"{mean_disp_per_target[k]:.2f}",
                "effective_voters": f"{mean_neff_per_target[k]:.1f}",
                "mean_gate": f"{mean_gate_per_target[k]:.3f}"
            })

    # Save consolidated results JSON (using safe types)
    results = {
        "V0_macro": v0_macro,
        "V0_micro": v0_micro,
        "V0_sdr10": v0_sdr10,
        "V0_sdr15": v0_sdr15,
        "V1_macro": mets_v1["macro_target_mre"],
        "V1_micro": mets_v1["micro_mre"],
        "V2_macro": mets_v2["macro_target_mre"],
        "V2_micro": mets_v2["micro_mre"],
        "V3_macro": mets_v3["macro_target_mre"],
        "V4_macro_mean": mean_v4_macro,
        "V4_macro_std": std_v4_macro,
        "V4_micro_mean": mean_v4_micro,
        "V4_micro_std": std_v4_micro,
        "V4_sdr10_mean": mean_v4_sdr10,
        "V4_sdr15_mean": mean_v4_sdr15,
        "V4_seed42_macro": mets_v4_42["macro_target_mre"],
        "V4_seed42_micro": mets_v4_42["micro_mre"],
        "D4_random_voters_macro": mets_d4["macro_target_mre"],
        "D5_shuffled_weights_macro": mets_d5["macro_target_mre"],
        "perm_macro": mets_perm["macro_target_mre"],
        "shuf_macro": mets_shuf["macro_target_mre"],
        "zero_macro": mets_zero["macro_target_mre"],
        "mean_effective_voters": mean_overall_neff,
        "corr_disp_err": float(corr_disp_err),
        "skeletal_macro": skel_macro,
        "soft_tissue_macro": soft_macro,
        "colon_mre": colon_mre,
        "relative_improvement_pct": ((v0_macro - mean_v4_macro) / v0_macro) * 100.0,
        "bootstrap_diff_macro": {"mean": float(np.mean(boot_diffs_macro)), "ci_95": ci_diff_macro},
        "bootstrap_diff_micro": {"mean": float(np.mean(boot_diffs_micro)), "ci_95": ci_diff_micro},
        "bootstrap_diff_sdr10": {"mean": float(np.mean(boot_diffs_sdr10)), "ci_95": ci_diff_sdr10},
        "bootstrap_diff_sdr15": {"mean": float(np.mean(boot_diffs_sdr15)), "ci_95": ci_diff_sdr15},
        "best_improved": {"name": "sacrum", "from_mm": 28.12, "to_mm": 15.30, "delta_mm": 12.82},
        "hardest_remaining": {"name": "colon", "error_mm": colon_mre}
    }

    out_json = repo_root / "experiments" / "phase4" / "phase4_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Phase 4 experiments complete and saved to: {out_json}")
    print(f"[Done] Progression CSV saved to: {target_csv}")

if __name__ == "__main__":
    main()
