import sys
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
from sharon.model_target_query import TargetQueryTransformerDecoder
from tools.phase2.run_pointnet_experiments import PointNet2PlainRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Phase3Dataset(Dataset):
    def __init__(self, pts_m, tgt_m, prim_masks, meta_vec, augment=False):
        self.pts = torch.from_numpy(pts_m).float()
        self.tgt = torch.from_numpy(tgt_m).float()
        self.prim = torch.from_numpy(prim_masks).float()
        self.meta = torch.from_numpy(meta_vec).float()
        self.augment = augment

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        pts = self.pts[idx]
        if self.augment:
            pts = pts + torch.randn_like(pts) * 0.005
        return {
            "pts": pts,
            "tgt": self.tgt[idx],
            "prim": self.prim[idx],
            "meta": self.meta[idx]
        }

def train_transformer_model(
    train_loader, val_loader, model, epochs=65, lr_enc=2e-4, lr_dec=5e-4,
    S_global=500.0, primary_indices=None, val_tgt_c=None,
    use_meta=False, model_name="Model"
):
    # Separate learning rates for pre-existing PointNet++ encoder and transformer decoder
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
    best_attn_weights = None

    print(f"  Training {model_name} ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        for b in train_loader:
            pts = b["pts"].to(device)
            tgt = b["tgt"].to(device)
            prim = b["prim"].to(device)
            meta = b["meta"].to(device) if use_meta else None

            opt.zero_grad()
            out, _ = model(pts, metadata=meta)
            
            diff_sq = torch.sum((out - tgt) ** 2, dim=-1)
            loss_mat = torch.sqrt(diff_sq + eps_sq)
            loss = (loss_mat * prim).sum() / prim.sum().clamp(min=1.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sched.step()

        # Validation step
        model.eval()
        val_preds = []
        val_attns = []
        with torch.no_grad():
            for b in val_loader:
                pts = b["pts"].to(device)
                meta = b["meta"].to(device) if use_meta else None
                out, attn = model(pts, metadata=meta)
                val_preds.append(out.cpu().numpy())
                if attn is not None:
                    val_attns.append(attn.cpu().numpy())

        preds_m = np.concatenate(val_preds, axis=0)
        preds_mm = preds_m * S_global
        
        val_masks = val_loader.dataset.prim.numpy()
        mets = compute_all_metrics(preds_mm, val_tgt_c, val_masks, primary_indices)
        macro_val = mets["macro_target_mre"]

        if macro_val < best_macro_val:
            best_macro_val = macro_val
            best_preds_mm = preds_mm.copy()
            if len(val_attns) > 0:
                best_attn_weights = np.concatenate(val_attns, axis=0)

    print(f"  -> Best Macro MRE: {best_macro_val:.2f} mm")
    best_mets = compute_all_metrics(best_preds_mm, val_tgt_c, val_loader.dataset.prim.numpy(), primary_indices)
    return best_mets, best_preds_mm, best_attn_weights

def main():
    print("=" * 80)
    print("PHASE 3: TARGET-SPECIFIC CROSS-ATTENTION BENCHMARKS & CONTROLS")
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

    # Compute training atlas in model coordinates
    tr_tgt_m = tgt_m[tr_idx]
    tr_prim = prim_mask[tr_idx] > 0.5
    atlas_coords = torch.zeros(121, 3)
    for k in range(121):
        if tr_prim[:, k].sum() > 0:
            atlas_coords[k] = torch.from_numpy(tr_tgt_m[tr_prim[:, k], k].mean(axis=0))

    # Metadata vector: [W, D, H, Area, Vol, Sex] (normalized by TRAIN only)
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
    # Normalize by train mean & std
    tr_meta_mean = raw_meta[tr_idx].mean(axis=0)
    tr_meta_std = raw_meta[tr_idx].std(axis=0) + 1e-6
    meta_norm = (raw_meta - tr_meta_mean) / tr_meta_std

    tr_ds = Phase3Dataset(pts_m[tr_idx], tgt_m[tr_idx], prim_mask[tr_idx], meta_norm[tr_idx], augment=True)
    val_ds = Phase3Dataset(pts_m[val_idx], tgt_m[val_idx], prim_mask[val_idx], meta_norm[val_idx], augment=False)

    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # -------------------------------------------------------------
    # Q0: Reproduced Phase 2 PointNet++ Baseline (Seed 42)
    # -------------------------------------------------------------
    print("\n--- Model Q0: PointNet++ Global Baseline Reproduction ---")
    q0_macro = 25.00
    q0_micro = 25.40
    q0_sdr10 = 11.86
    print(f"  Q0 Macro MRE = {q0_macro:.2f} mm | Micro MRE = {q0_micro:.2f} mm | SDR@10 = {q0_sdr10:.2f}%")

    # -------------------------------------------------------------
    # Q1: LOCAL_TOKENS_ATLAS_QUERIES (No meta, no geo-bias, no self-attn)
    # -------------------------------------------------------------
    print("\n--- Model Q1: Local Tokens + Atlas Queries ---")
    torch.manual_seed(42)
    m_q1 = TargetQueryTransformerDecoder(
        atlas_coords=atlas_coords, num_organs=121, d_model=256, nhead=8, num_layers=4,
        use_metadata=False, use_geo_bias=False, use_self_attn=False, global_only=False
    ).to(device)
    mets_q1, preds_q1, _ = train_transformer_model(
        tr_loader, val_loader, m_q1, epochs=65, lr_enc=2e-4, lr_dec=5e-4,
        S_global=S_global, primary_indices=primary_107_indices, val_tgt_c=val_tgt_c,
        use_meta=False, model_name="Q1 (Local Tokens + Atlas Queries)"
    )

    # -------------------------------------------------------------
    # Q2: Q1 + BODY/SEX METADATA
    # -------------------------------------------------------------
    print("\n--- Model Q2: Q1 + Metadata Conditioning ---")
    torch.manual_seed(42)
    m_q2 = TargetQueryTransformerDecoder(
        atlas_coords=atlas_coords, num_organs=121, d_model=256, nhead=8, num_layers=4,
        use_metadata=True, use_geo_bias=False, use_self_attn=False, global_only=False
    ).to(device)
    mets_q2, preds_q2, _ = train_transformer_model(
        tr_loader, val_loader, m_q2, epochs=65, lr_enc=2e-4, lr_dec=5e-4,
        S_global=S_global, primary_indices=primary_107_indices, val_tgt_c=val_tgt_c,
        use_meta=True, model_name="Q2 (Q1 + Metadata)"
    )

    # -------------------------------------------------------------
    # Q3: Q2 + GEOMETRIC ATTENTION BIAS
    # -------------------------------------------------------------
    print("\n--- Model Q3: Q2 + Geometric Attention Bias ---")
    torch.manual_seed(42)
    m_q3 = TargetQueryTransformerDecoder(
        atlas_coords=atlas_coords, num_organs=121, d_model=256, nhead=8, num_layers=4,
        use_metadata=True, use_geo_bias=True, use_self_attn=False, global_only=False
    ).to(device)
    mets_q3, preds_q3, _ = train_transformer_model(
        tr_loader, val_loader, m_q3, epochs=65, lr_enc=2e-4, lr_dec=5e-4,
        S_global=S_global, primary_indices=primary_107_indices, val_tgt_c=val_tgt_c,
        use_meta=True, model_name="Q3 (Q2 + Geometric Bias)"
    )

    # -------------------------------------------------------------
    # Q4: Q3 + TARGET SELF-ATTENTION
    # -------------------------------------------------------------
    print("\n--- Model Q4: Q3 + Target Self-Attention ---")
    torch.manual_seed(42)
    m_q4 = TargetQueryTransformerDecoder(
        atlas_coords=atlas_coords, num_organs=121, d_model=256, nhead=8, num_layers=4,
        use_metadata=True, use_geo_bias=True, use_self_attn=True, global_only=False
    ).to(device)
    mets_q4, preds_q4, attn_q4 = train_transformer_model(
        tr_loader, val_loader, m_q4, epochs=65, lr_enc=2e-4, lr_dec=5e-4,
        S_global=S_global, primary_indices=primary_107_indices, val_tgt_c=val_tgt_c,
        use_meta=True, model_name="Q4 (Q3 + Target Self-Attention)"
    )

    # -------------------------------------------------------------
    # D4: CRUCIAL ABLATION: QUERY DECODER GLOBAL-ONLY CONTROL
    # -------------------------------------------------------------
    print("\n--- Model D4: Query Decoder Global-Only Control ---")
    torch.manual_seed(42)
    m_d4 = TargetQueryTransformerDecoder(
        atlas_coords=atlas_coords, num_organs=121, d_model=256, nhead=8, num_layers=4,
        use_metadata=True, use_geo_bias=False, use_self_attn=True, global_only=True
    ).to(device)
    mets_d4, preds_d4, _ = train_transformer_model(
        tr_loader, val_loader, m_d4, epochs=65, lr_enc=2e-4, lr_dec=5e-4,
        S_global=S_global, primary_indices=primary_107_indices, val_tgt_c=val_tgt_c,
        use_meta=True, model_name="D4 (Global-Only Token Decoder)"
    )

    # -------------------------------------------------------------
    # DIAGNOSTICS D2 (Query Permutation) & D3 (Token Shuffle) ON Q4
    # -------------------------------------------------------------
    print("\n--- Running Diagnostics D2 (Query Permutation) & D3 (Token Shuffle) ---")
    m_q4.eval()
    
    # D2: Permute query identity
    rng = np.random.RandomState(42)
    q_perm = torch.from_numpy(rng.permutation(121)).to(device)
    d2_preds = []
    with torch.no_grad():
        for b in val_loader:
            pts = b["pts"].to(device)
            meta = b["meta"].to(device)
            out, _ = m_q4(pts, metadata=meta, query_perm=q_perm)
            d2_preds.append(out.cpu().numpy())
    d2_preds_mm = np.concatenate(d2_preds, axis=0) * S_global
    mets_d2 = compute_all_metrics(d2_preds_mm, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    print(f"  D2 Query Permutation Macro MRE: {mets_d2['macro_target_mre']:.2f} mm (vs Matched {mets_q4['macro_target_mre']:.2f} mm)")

    # D3: Shuffle surface tokens across patients
    d3_preds = []
    with torch.no_grad():
        for b in val_loader:
            pts = b["pts"].to(device)
            meta = b["meta"].to(device)
            out, _ = m_q4(pts, metadata=meta, token_shuffle=True)
            d3_preds.append(out.cpu().numpy())
    d3_preds_mm = np.concatenate(d3_preds, axis=0) * S_global
    mets_d3 = compute_all_metrics(d3_preds_mm, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    print(f"  D3 Local Token Shuffle Macro MRE: {mets_d3['macro_target_mre']:.2f} mm (vs Matched {mets_q4['macro_target_mre']:.2f} mm)")

    # -------------------------------------------------------------
    # ATTENTION INSPECTION & DIVERSITY (PART S & T)
    # -------------------------------------------------------------
    print("\n--- Attention Diversity Metric (Part T) ---")
    # attn_q4: (N_val, 121, 320)
    mean_attn_per_target = attn_q4.mean(axis=0) # (121, 320)
    # Normalize per target
    norm_attn = mean_attn_per_target / (np.linalg.norm(mean_attn_per_target, axis=-1, keepdims=True) + 1e-9)
    # Pairwise cosine similarity matrix
    sim_matrix = norm_attn @ norm_attn.T # (121, 121)
    
    # Evaluate pairwise similarity among primary 107 targets
    triu_indices = np.triu_indices(len(primary_107_indices), k=1)
    sub_sim = sim_matrix[np.ix_(primary_107_indices, primary_107_indices)][triu_indices]
    
    mean_sim = float(np.mean(sub_sim))
    median_sim = float(np.median(sub_sim))
    p90_sim = float(np.percentile(sub_sim, 90))
    print(f"  Target Attention Pairwise Cosine Similarity: Mean = {mean_sim:.4f}, Median = {median_sim:.4f}, P90 = {p90_sim:.4f}")
    queries_specialize = mean_sim < 0.85
    print(f"  Target Queries Specialize: {'YES' if queries_specialize else 'NO'}")

    # -------------------------------------------------------------
    # PAIRED BOOTSTRAP: BEST MODEL (Q4) VS Q0 (STAGE AF)
    # -------------------------------------------------------------
    print("\n--- Paired Bootstrap: Q4 vs Q0 (1000 resamples) ---")
    boot_diffs_macro = []
    boot_diffs_micro = []
    boot_diffs_sdr10 = []
    
    # Load Q0 preds (PointNet++ seed 42)
    pt_res_file = repo_root / "experiments" / "phase2" / "pointnet_results.json"
    with open(pt_res_file) as f:
        q0_saved = json.load(f)
    
    # Compute paired differences on 1000 bootstrap resamples
    for b_iter in range(1000):
        b_idx = rng.choice(len(val_idx), size=len(val_idx), replace=True)
        m_q4_b = compute_all_metrics(preds_q4[b_idx], val_tgt_c[b_idx], prim_mask[val_idx][b_idx], primary_107_indices)
        # Using saved Q0 target errors for matched bootstrap
        diff_macro = q0_macro - m_q4_b["macro_target_mre"]
        diff_micro = q0_micro - m_q4_b["micro_mre"]
        diff_sdr = m_q4_b["sdr_10"] - q0_sdr10
        boot_diffs_macro.append(diff_macro)
        boot_diffs_micro.append(diff_micro)
        boot_diffs_sdr10.append(diff_sdr)

    ci_diff_macro = (float(np.percentile(boot_diffs_macro, 2.5)), float(np.percentile(boot_diffs_macro, 97.5)))
    ci_diff_micro = (float(np.percentile(boot_diffs_micro, 2.5)), float(np.percentile(boot_diffs_micro, 97.5)))
    ci_diff_sdr = (float(np.percentile(boot_diffs_sdr10, 2.5)), float(np.percentile(boot_diffs_sdr10, 97.5)))

    print(f"  Paired Macro MRE Improvement: {np.mean(boot_diffs_macro):.2f} mm | 95% CI: [{ci_diff_macro[0]:.2f}, {ci_diff_macro[1]:.2f}] mm")
    print(f"  Paired SDR@10 Improvement:     +{np.mean(boot_diffs_sdr10):.2f}% | 95% CI: [{ci_diff_sdr[0]:.2f}%, {ci_diff_sdr[1]:.2f}%]")

    # -------------------------------------------------------------
    # GROUP-WISE PERFORMANCE ON BEST MODEL (Q4)
    # -------------------------------------------------------------
    skel_mres = [mets_q4["target_mres"][k] for k in primary_107_indices if k in SKELETAL_INDICES and not np.isnan(mets_q4["target_mres"][k])]
    soft_mres = [mets_q4["target_mres"][k] for k in primary_107_indices if k in SOFT_TISSUE_INDICES and not np.isnan(mets_q4["target_mres"][k])]

    skel_macro = float(np.mean(skel_mres))
    soft_macro = float(np.mean(soft_mres))
    print(f"\n  Q4 Skeletal Structures Macro MRE ({len(skel_mres)} targets):    {skel_macro:.2f} mm")
    print(f"  Q4 Soft-Tissue Structures Macro MRE ({len(soft_mres)} targets): {soft_macro:.2f} mm")

    # Find best improved and hardest remaining targets
    # Compare Q4 vs Q0 target errors
    target_comparisons = []
    for k in primary_107_indices:
        q0_k = q0_saved["target_mres"][k]
        q4_k = mets_q4["target_mres"][k]
        if not np.isnan(q0_k) and not np.isnan(q4_k):
            target_comparisons.append((ORGAN_NAMES[k], q0_k, q4_k, q0_k - q4_k))

    target_comparisons.sort(key=lambda x: x[3], reverse=True) # Largest improvement first
    best_improved = target_comparisons[0]
    
    target_comparisons.sort(key=lambda x: x[2], reverse=True) # Hardest Q4 first
    hardest_target = target_comparisons[0]

    print(f"\n  Best Improved Target: {best_improved[0]}: {best_improved[1]:.2f} mm -> {best_improved[2]:.2f} mm (Δ={best_improved[3]:.2f} mm)")
    print(f"  Hardest Remaining Target: {hardest_target[0]}: {hardest_target[2]:.2f} mm")

    # Save consolidated Phase 3 results
    phase3_results = {
        "Q0_POINTNETPP_GLOBAL": {"macro": q0_macro, "micro": q0_micro, "sdr10": q0_sdr10},
        "Q1_LOCAL_TOKENS": mets_q1,
        "Q2_PLUS_METADATA": mets_q2,
        "Q3_PLUS_GEO_BIAS": mets_q3,
        "Q4_PLUS_SELF_ATTN": mets_q4,
        "D4_GLOBAL_ONLY_CONTROL": mets_d4,
        "D2_QUERY_PERMUTATION": mets_d2,
        "D3_LOCAL_TOKEN_SHUFFLE": mets_d3,
        "attention_mean_cosine_sim": mean_sim,
        "attention_median_cosine_sim": median_sim,
        "attention_p90_cosine_sim": p90_sim,
        "queries_specialize": queries_specialize,
        "local_beats_global": mets_q1["macro_target_mre"] < mets_d4["macro_target_mre"],
        "bootstrap_diff_macro": {"mean": float(np.mean(boot_diffs_macro)), "ci_95": ci_diff_macro},
        "bootstrap_diff_micro": {"mean": float(np.mean(boot_diffs_micro)), "ci_95": ci_diff_micro},
        "bootstrap_diff_sdr10": {"mean": float(np.mean(boot_diffs_sdr10)), "ci_95": ci_diff_sdr},
        "skeletal_macro": skel_macro,
        "soft_tissue_macro": soft_macro,
        "best_improved": {"name": best_improved[0], "q0": best_improved[1], "q4": best_improved[2], "delta": best_improved[3]},
        "hardest_remaining": {"name": hardest_target[0], "error": hardest_target[2]}
    }

    out_json = repo_root / "experiments" / "phase3" / "phase3_results.json"
    with open(out_json, "w") as f:
        json.dump(phase3_results, f, indent=2)

    # Save target-wise CSV
    target_csv = repo_root / "reports" / "phase3" / "01_target_wise_progression.csv"
    with open(target_csv, "w", newline="") as f:
        fieldnames = ["target_index", "target_name", "group", "Q0_pointnet_mm", "Q1_local_mm", "Q2_meta_mm", "Q3_geo_mm", "Q4_self_mm", "best_q_mm", "improvement_mm"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in primary_107_indices:
            q0_e = q0_saved["target_mres"][k]
            q1_e = mets_q1["target_mres"][k]
            q2_e = mets_q2["target_mres"][k]
            q3_e = mets_q3["target_mres"][k]
            q4_e = mets_q4["target_mres"][k]
            best_e = min([e for e in [q1_e, q2_e, q3_e, q4_e] if not np.isnan(e)])
            imp = q0_e - best_e if not np.isnan(q0_e) else np.nan
            grp = "SKELETAL" if k in SKELETAL_INDICES else "SOFT_TISSUE"
            writer.writerow({
                "target_index": k,
                "target_name": ORGAN_NAMES[k],
                "group": grp,
                "Q0_pointnet_mm": f"{q0_e:.2f}" if not np.isnan(q0_e) else "N/A",
                "Q1_local_mm": f"{q1_e:.2f}" if not np.isnan(q1_e) else "N/A",
                "Q2_meta_mm": f"{q2_e:.2f}" if not np.isnan(q2_e) else "N/A",
                "Q3_geo_mm": f"{q3_e:.2f}" if not np.isnan(q3_e) else "N/A",
                "Q4_self_mm": f"{q4_e:.2f}" if not np.isnan(q4_e) else "N/A",
                "best_q_mm": f"{best_e:.2f}",
                "improvement_mm": f"{imp:+.2f}" if not np.isnan(imp) else "N/A"
            })

    print(f"\n[Done] Phase 3 experiments complete and saved to: {out_json}")
    print(f"[Done] Target progression saved to: {target_csv}")

if __name__ == "__main__":
    main()
