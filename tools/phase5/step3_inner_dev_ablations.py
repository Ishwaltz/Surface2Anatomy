import sys
import json
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES
from tools.phase2.metrics import compute_all_metrics
from sharon.model_voting import TargetConditionedVotingModel
from sharon.model_refiner import TargetSpecificSurfaceRefiner, SurfaceRefinementPipeline
from tools.phase3.run_phase3_experiments import Phase3Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_eval_refiner_inner(
    tr_loader, dev_loader, pipeline, dev_tgt_mm, dev_masks, primary_107_indices,
    epochs=40, lr=1e-3, S_global=500.0, name="Refiner"
):
    opt = torch.optim.AdamW(pipeline.refiner.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_macro = 999.0
    for ep in range(epochs):
        pipeline.refiner.train()
        for b in tr_loader:
            pts = b["pts"].to(device)
            tgt = b["tgt"].to(device)
            prim = b["prim"].to(device)
            meta = b["meta"].to(device)
            
            opt.zero_grad()
            out = pipeline(pts, metadata=meta, freeze_base=True)
            p1 = out["p1"]
            loss = (torch.norm(p1 - tgt, dim=-1) * prim).sum() / prim.sum().clamp(min=1.0)
            # Add weak L_delta regularization
            loss += 0.01 * (torch.norm(out["delta_p"], dim=-1) * prim).sum() / prim.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.refiner.parameters(), 1.0)
            opt.step()
        sched.step()
        
        # Dev evaluation
        pipeline.refiner.eval()
        preds_list = []
        with torch.no_grad():
            for b in dev_loader:
                pts = b["pts"].to(device)
                meta = b["meta"].to(device)
                out = pipeline(pts, metadata=meta, freeze_base=True)
                preds_list.append((out["p1"] * S_global).cpu().numpy())
        preds_mm = np.concatenate(preds_list, axis=0)
        mets = compute_all_metrics(preds_mm, dev_tgt_mm, dev_masks, primary_107_indices)
        macro = mets["macro_target_mre"]
        if macro < best_macro:
            best_macro = macro
            
    print(f"    [{name}] Inner-Dev Best Macro MRE: {best_macro:.2f} mm")
    return best_macro

def main():
    print("=" * 80)
    print("PHASE 5 - STAGE 23 & 24: SUPPORT STRATEGY & MULTI-ANCHOR ABLATIONS (INNER-DEV)")
    print("=" * 80)

    # 1. Load Data and Inner Splits
    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    inner_path = repo_root / "experiments" / "phase4" / "inner_splits.json"
    ckpt_path = repo_root / "experiments" / "phase5" / "r0_phase4_base_seed42.pt"
    base_res = json.load(open(repo_root / "experiments" / "phase5" / "step1_baseline_results.json"))
    tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"

    data = torch.load(str(pt_path), weights_only=False)
    with open(inner_path) as f:
        inner_splits = json.load(f)

    with open(tiers_path) as f:
        primary_107_indices = [int(r["target_index"]) for r in csv.DictReader(f) if r["tier"] in ["TIER_A", "TIER_B"]]

    in_tr_idx = np.array(inner_splits["inner_train_indices"]) # 282
    in_dev_idx = np.array(inner_splits["inner_dev_indices"])   # 70

    pts_m = data["points_model"].numpy()
    tgt_m = data["targets_model"].numpy()
    prim_mask = data["target_primary_mask"].numpy()
    tgt_c_mm = data["targets_centered_mm"].numpy()
    S_global = float(data["s_global"])

    # Load metadata
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
    tr_meta_mean = raw_meta[in_tr_idx].mean(axis=0)
    tr_meta_std = raw_meta[in_tr_idx].std(axis=0) + 1e-6
    meta_norm = (raw_meta - tr_meta_mean) / tr_meta_std

    tr_ds = Phase3Dataset(pts_m[in_tr_idx], tgt_m[in_tr_idx], prim_mask[in_tr_idx], meta_norm[in_tr_idx], augment=True)
    dev_ds = Phase3Dataset(pts_m[in_dev_idx], tgt_m[in_dev_idx], prim_mask[in_dev_idx], meta_norm[in_dev_idx], augment=False)

    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False)

    dev_tgt_mm = tgt_c_mm[in_dev_idx]
    dev_masks = prim_mask[in_dev_idx]

    # Load base model
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    base_model = TargetConditionedVotingModel(atlas_coords=ckpt["atlas_coords"]).to(device)
    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()

    # Initial coarse error on inner_dev
    coarse_dev_preds = []
    with torch.no_grad():
        for b in dev_loader:
            pts = b["pts"].to(device)
            meta = b["meta"].to(device)
            out = base_model(pts, metadata=meta)
            coarse_dev_preds.append((out["p_final"] * S_global).cpu().numpy())
    coarse_dev_mm = np.concatenate(coarse_dev_preds, axis=0)
    coarse_dev_mets = compute_all_metrics(coarse_dev_mm, dev_tgt_mm, dev_masks, primary_107_indices)
    print(f"\nInitial Inner-Dev Coarse Macro MRE (p0): {coarse_dev_mets['macro_target_mre']:.2f} mm")

    R_bounds_mm = torch.tensor(base_res["R_k_p95_mm"]).float().to(device)

    ablation_results = {}

    # 1. ABLATION 1: Support Strategy Comparison (M=64, K_anchors=1)
    print("\n--- 1. Support Strategy Comparison (M=64, K_anchors=1) ---")
    strategies = ["attention", "voting", "combined", "axial_band"]
    strat_scores = {}
    for strat in strategies:
        torch.manual_seed(42)
        refiner = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2).to(device)
        pipe = SurfaceRefinementPipeline(base_model, refiner, R_bounds_mm=R_bounds_mm, S_global=S_global, support_strategy=strat, M=64, k_anchors=1).to(device)
        m = train_eval_refiner_inner(tr_loader, dev_loader, pipe, dev_tgt_mm, dev_masks, primary_107_indices, epochs=35, name=f"Strategy-{strat}")
        strat_scores[strat] = m

    best_strat = min(strat_scores, key=strat_scores.get)
    print(f"  -> Best Support Strategy: {best_strat} ({strat_scores[best_strat]:.2f} mm)")
    ablation_results["support_strategy"] = strat_scores

    # 2. ABLATION 2: Multi-Anchor Count (K_anchors = 1, 2, 4) with best strategy
    print(f"\n--- 2. Multi-Anchor Count Ablation (Strategy={best_strat}, M=64) ---")
    anchor_scores = {}
    for ka in [1, 2, 4]:
        torch.manual_seed(42)
        refiner = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2).to(device)
        pipe = SurfaceRefinementPipeline(base_model, refiner, R_bounds_mm=R_bounds_mm, S_global=S_global, support_strategy=best_strat, M=64, k_anchors=ka).to(device)
        m = train_eval_refiner_inner(tr_loader, dev_loader, pipe, dev_tgt_mm, dev_masks, primary_107_indices, epochs=35, name=f"K_anchors={ka}")
        anchor_scores[str(ka)] = m

    best_ka = int(min(anchor_scores, key=anchor_scores.get))
    print(f"  -> Best Multi-Anchor Count: {best_ka} ({anchor_scores[str(best_ka)]:.2f} mm)")
    ablation_results["multi_anchor"] = anchor_scores

    # 3. ABLATION 3: Support Size M (64, 128)
    print(f"\n--- 3. Support Token Size M Ablation (Strategy={best_strat}, K_anchors={best_ka}) ---")
    m_scores = {}
    for m_val in [64, 128]:
        torch.manual_seed(42)
        refiner = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2).to(device)
        pipe = SurfaceRefinementPipeline(base_model, refiner, R_bounds_mm=R_bounds_mm, S_global=S_global, support_strategy=best_strat, M=m_val, k_anchors=best_ka).to(device)
        m = train_eval_refiner_inner(tr_loader, dev_loader, pipe, dev_tgt_mm, dev_masks, primary_107_indices, epochs=35, name=f"M={m_val}")
        m_scores[str(m_val)] = m

    best_m = int(min(m_scores, key=m_scores.get))
    print(f"  -> Best Support Size M: {best_m} ({m_scores[str(best_m)]:.2f} mm)")
    ablation_results["support_size_M"] = m_scores

    ablation_summary = {
        "initial_coarse_dev_mre": coarse_dev_mets['macro_target_mre'],
        "best_strategy": best_strat,
        "best_k_anchors": best_ka,
        "best_M": best_m,
        "ablation_results": ablation_results
    }

    out_file = repo_root / "experiments" / "phase5" / "inner_dev_ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(ablation_summary, f, indent=2)
    print(f"\nInner-Dev Ablation Results saved to: {out_file}")

if __name__ == "__main__":
    main()
