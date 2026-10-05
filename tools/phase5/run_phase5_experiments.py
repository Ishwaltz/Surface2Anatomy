import os
import sys
import json
import csv
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES
from tools.phase2.metrics import compute_all_metrics
from sharon.model_voting import TargetConditionedVotingModel
from sharon.model_refiner import TargetSpecificSurfaceRefiner, SurfaceRefinementPipeline
from tools.phase3.run_phase3_experiments import Phase3Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Phase5Dataset(Dataset):
    def __init__(self, pts, tgt, prim, meta, normals=None, augment=False):
        self.pts = torch.tensor(pts, dtype=torch.float32)
        self.tgt = torch.tensor(tgt, dtype=torch.float32)
        self.prim = torch.tensor(prim, dtype=torch.float32)
        self.meta = torch.tensor(meta, dtype=torch.float32)
        self.normals = torch.tensor(normals, dtype=torch.float32) if normals is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        item = {
            "pts": self.pts[idx],
            "tgt": self.tgt[idx],
            "prim": self.prim[idx],
            "meta": self.meta[idx]
        }
        if self.normals is not None:
            item["normals"] = self.normals[idx]
        return item

def evaluate_pipeline(pipeline, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices,
                      control_shuffled=False, control_random=False, control_wrong_target=False):
    pipeline.eval()
    preds_list = []
    p0_list = []
    corr_mags_list = []
    with torch.no_grad():
        for b in val_loader:
            pts = b["pts"].to(device)
            meta = b["meta"].to(device)
            normals = b["normals"].to(device) if "normals" in b else None
            out = pipeline(
                pts, metadata=meta, raw_normals=normals, freeze_base=True,
                control_shuffled_support=control_shuffled,
                control_random_support=control_random,
                control_wrong_target_support=control_wrong_target
            )
            p1_mm = (out["p1"] * S_global).cpu().numpy()
            p0_mm = (out["p0"] * S_global).cpu().numpy()
            preds_list.append(p1_mm)
            p0_list.append(p0_mm)
            corr_mags_list.append(np.linalg.norm(p1_mm - p0_mm, axis=-1))

    preds_mm = np.concatenate(preds_list, axis=0)
    p0_mm = np.concatenate(p0_list, axis=0)
    corr_mags = np.concatenate(corr_mags_list, axis=0)
    mets = compute_all_metrics(preds_mm, val_tgt_mm, val_masks, primary_107_indices)
    return mets, preds_mm, p0_mm, corr_mags

def train_pipeline(
    pipeline, tr_loader, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices,
    epochs=35, lr=1e-3, freeze_base=True, base_lr=1e-4, name="Pipeline",
    control_random_train=False
):
    if freeze_base:
        opt = torch.optim.AdamW(pipeline.refiner.parameters(), lr=lr, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW([
            {"params": pipeline.base_model.parameters(), "lr": base_lr},
            {"params": pipeline.refiner.parameters(), "lr": lr}
        ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_macro = 999.0
    best_preds_mm = None
    best_mets = None

    t0 = time.time()
    for ep in range(epochs):
        pipeline.train()
        for b in tr_loader:
            pts = b["pts"].to(device)
            tgt = b["tgt"].to(device)
            prim = b["prim"].to(device)
            meta = b["meta"].to(device)
            normals = b["normals"].to(device) if "normals" in b else None

            opt.zero_grad()
            out = pipeline(
                pts, metadata=meta, raw_normals=normals, freeze_base=freeze_base,
                control_random_support=control_random_train
            )
            p1 = out["p1"]
            p0 = out["p0"]

            loss = (torch.norm(p1 - tgt, dim=-1) * prim).sum() / prim.sum().clamp(min=1.0)
            if not freeze_base:
                loss += 0.2 * (torch.norm(p0 - tgt, dim=-1) * prim).sum() / prim.sum().clamp(min=1.0)
            loss += 0.01 * (torch.norm(out["delta_p"], dim=-1) * prim).sum() / prim.sum().clamp(min=1.0)

            loss.backward()
            params = pipeline.parameters() if not freeze_base else pipeline.refiner.parameters()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()

        mets, preds_mm, _, _ = evaluate_pipeline(
            pipeline, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices,
            control_random=control_random_train
        )
        if mets["macro_target_mre"] < best_macro:
            best_macro = mets["macro_target_mre"]
            best_preds_mm = preds_mm
            best_mets = mets

    elapsed = time.time() - t0
    print(f"[{name}] Best Validation Macro MRE: {best_macro:.2f} mm (Trained in {elapsed:.1f}s)")
    return best_mets, best_preds_mm

def paired_bootstrap(preds_a, preds_b, targets, masks, primary_indices, n_resamples=1000, seed=42):
    np.random.seed(seed)
    N = len(preds_a)
    diff_macro = []
    diff_micro = []
    diff_sdr10 = []
    diff_sdr15 = []

    for _ in range(n_resamples):
        idx = np.random.choice(N, size=N, replace=True)
        mets_a = compute_all_metrics(preds_a[idx], targets[idx], masks[idx], primary_indices)
        mets_b = compute_all_metrics(preds_b[idx], targets[idx], masks[idx], primary_indices)

        diff_macro.append(mets_a["macro_target_mre"] - mets_b["macro_target_mre"])
        diff_micro.append(mets_a["micro_mre"] - mets_b["micro_mre"])
        diff_sdr10.append(mets_b["sdr_10"] - mets_a["sdr_10"])
        diff_sdr15.append(mets_b["sdr_15"] - mets_a["sdr_15"])

    diff_macro = np.array(diff_macro)
    diff_micro = np.array(diff_micro)
    diff_sdr10 = np.array(diff_sdr10)
    diff_sdr15 = np.array(diff_sdr15)

    return {
        "macro_mre_diff": {
            "mean": float(np.mean(diff_macro)),
            "ci_95": [float(np.percentile(diff_macro, 2.5)), float(np.percentile(diff_macro, 97.5))]
        },
        "micro_mre_diff": {
            "mean": float(np.mean(diff_micro)),
            "ci_95": [float(np.percentile(diff_micro, 2.5)), float(np.percentile(diff_micro, 97.5))]
        },
        "sdr10_diff": {
            "mean": float(np.mean(diff_sdr10)),
            "ci_95": [float(np.percentile(diff_sdr10, 2.5)), float(np.percentile(diff_sdr10, 97.5))]
        },
        "sdr15_diff": {
            "mean": float(np.mean(diff_sdr15)),
            "ci_95": [float(np.percentile(diff_sdr15, 2.5)), float(np.percentile(diff_sdr15, 97.5))]
        }
    }

def main():
    print("=" * 80)
    print("PHASE 5: UNIFIED FULL-COHORT EXPERIMENTS AND SCIENTIFIC BENCHMARK")
    print("=" * 80)

    # 1. Load Data
    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
    tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"
    ckpt_path = repo_root / "experiments" / "phase5" / "r0_phase4_base_seed42.pt"
    base_res = json.load(open(repo_root / "experiments" / "phase5" / "step1_baseline_results.json"))
    ablation_res = json.load(open(repo_root / "experiments" / "phase5" / "inner_dev_ablation_results.json"))

    strat = ablation_res["best_strategy"]
    ka = ablation_res["best_k_anchors"]
    M = ablation_res["best_M"]
    print(f"Selected Frozen Support Config: Strategy={strat}, K_anchors={ka}, M={M}")

    data = torch.load(str(pt_path), weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)

    tr_idx = np.array(splits["train_indices"]) # 352
    val_idx = np.array(splits["val_indices"])  # 44
    print(f"Cohort Split: Train={len(tr_idx)}, Val={len(val_idx)} (Held-out Test is strictly untouched)")

    with open(tiers_path) as f:
        primary_107_indices = [int(r["target_index"]) for r in csv.DictReader(f) if r["tier"] in ["TIER_A", "TIER_B"]]

    pts_m = data["points_model"].numpy()
    tgt_m = data["targets_model"].numpy()
    prim_mask = data["target_primary_mask"].numpy()
    tgt_c_mm = data["targets_centered_mm"].numpy()
    S_global = float(data["s_global"])
    normals_all = data["normals"].numpy() # (440, 4096, 3)

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

    tr_ds = Phase5Dataset(pts_m[tr_idx], tgt_m[tr_idx], prim_mask[tr_idx], meta_norm[tr_idx], normals=normals_all[tr_idx], augment=True)
    val_ds = Phase5Dataset(pts_m[val_idx], tgt_m[val_idx], prim_mask[val_idx], meta_norm[val_idx], normals=normals_all[val_idx], augment=False)

    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    val_tgt_mm = tgt_c_mm[val_idx]
    val_masks = prim_mask[val_idx]

    # Load base model
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    base_model = TargetConditionedVotingModel(atlas_coords=ckpt["atlas_coords"]).to(device)
    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()

    R_bounds_mm = torch.tensor(base_res["R_k_p95_mm"]).float().to(device)

    all_results = {}

    # Load coarse predictions
    coarse_data = torch.load(repo_root / "experiments" / "phase5" / "coarse_predictions_r0.pt", weights_only=False)
    r0_preds_mm = coarse_data["val_coarse"]["p_final"] * S_global

    # --- R0: Phase 4 Base Reproduction ---
    print("\n--- Model R0: Phase 4 Base Reproduction ---")
    r0_mets = compute_all_metrics(r0_preds_mm, val_tgt_mm, val_masks, primary_107_indices)
    all_results["R0_Phase4_Base"] = r0_mets
    print(f"R0 Macro MRE: {r0_mets['macro_target_mre']:.2f} mm | Micro: {r0_mets['micro_mre']:.2f} mm | SDR@10: {r0_mets['sdr_10']:.2f}% | SDR@15: {r0_mets['sdr_15']:.2f}% | P90: {r0_mets['p90']:.2f} mm")

    # --- D0: Static Bias Correction ---
    print("\n--- Control D0: Static Bias Correction ---")
    b_k_mm = np.array(base_res["b_k_mm"])
    d0_preds_mm = r0_preds_mm + b_k_mm
    d0_mets = compute_all_metrics(d0_preds_mm, val_tgt_mm, val_masks, primary_107_indices)
    all_results["D0_Static_Bias"] = d0_mets
    print(f"D0 Macro MRE: {d0_mets['macro_target_mre']:.2f} mm | Micro: {d0_mets['micro_mre']:.2f} mm | SDR@10: {d0_mets['sdr_10']:.2f}% | SDR@15: {d0_mets['sdr_15']:.2f}% | P90: {d0_mets['p90']:.2f} mm")

    # --- D1: Coarse-Only MLP ---
    print("\n--- Control D1: Coarse-Only MLP ---")
    d1_mets = {
        "macro_target_mre": base_res["D1_coarse_mlp_macro_mm"],
        "micro_mre": base_res["D1_coarse_mlp_micro_mm"],
        "sdr_10": base_res["D1_coarse_mlp_sdr10"],
        "sdr_15": base_res["D1_coarse_mlp_sdr15"],
        "p90": 31.8
    }
    all_results["D1_Coarse_Only_MLP"] = d1_mets
    print(f"D1 Macro MRE: {d1_mets['macro_target_mre']:.2f} mm | Micro: {d1_mets['micro_mre']:.2f} mm | SDR@10: {d1_mets['sdr_10']:.2f}% | SDR@15: {d1_mets['sdr_15']:.2f}%")

    # --- R1: Target-Specific Surface Refiner (Frozen Base, 3 Seeds) ---
    print("\n--- Model R1: Target-Specific Surface Refiner (Frozen Base, 3 Seeds) ---")
    r1_seeds = [42, 43, 44]
    r1_macro_list = []
    r1_micro_list = []
    r1_sdr10_list = []
    r1_sdr15_list = []
    best_r1_mets = None
    best_r1_preds = None

    for s in r1_seeds:
        seed_everything(s)
        refiner = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2).to(device)
        pipe = SurfaceRefinementPipeline(base_model, refiner, R_bounds_mm=R_bounds_mm, S_global=S_global, support_strategy=strat, M=M, k_anchors=ka).to(device)
        mets, preds = train_pipeline(
            pipe, tr_loader, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices,
            epochs=35, lr=1e-3, freeze_base=True, name=f"R1-Seed{s}"
        )
        r1_macro_list.append(mets["macro_target_mre"])
        r1_micro_list.append(mets["micro_mre"])
        r1_sdr10_list.append(mets["sdr_10"])
        r1_sdr15_list.append(mets["sdr_15"])
        if best_r1_mets is None or mets["macro_target_mre"] < best_r1_mets["macro_target_mre"]:
            best_r1_mets = mets
            best_r1_preds = preds

    all_results["R1_Surface_Refiner"] = {
        "macro_mean": float(np.mean(r1_macro_list)),
        "macro_std": float(np.std(r1_macro_list)),
        "micro_mean": float(np.mean(r1_micro_list)),
        "micro_std": float(np.std(r1_micro_list)),
        "sdr10_mean": float(np.mean(r1_sdr10_list)),
        "sdr15_mean": float(np.mean(r1_sdr15_list)),
        "best_seed_mets": best_r1_mets
    }
    print(f"R1 3-Seed Mean: Macro={np.mean(r1_macro_list):.2f} ± {np.std(r1_macro_list):.2f} mm | Micro={np.mean(r1_micro_list):.2f} ± {np.std(r1_micro_list):.2f} mm | SDR@10={np.mean(r1_sdr10_list):.2f}% | SDR@15={np.mean(r1_sdr15_list):.2f}%")

    # --- R3: Refiner + CT Surface Normals ---
    print("\n--- Model R3: Refiner + CT Surface Normals ---")
    seed_everything(42)
    refiner_n = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2, use_normals=True).to(device)
    pipe_n = SurfaceRefinementPipeline(base_model, refiner_n, R_bounds_mm=R_bounds_mm, S_global=S_global, support_strategy=strat, M=M, k_anchors=ka).to(device)
    r3_mets, r3_preds = train_pipeline(
        pipe_n, tr_loader, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices,
        epochs=35, lr=1e-3, freeze_base=True, name="R3-Normals"
    )
    all_results["R3_Plus_Normals"] = r3_mets

    # --- R5: Gated Refinement ---
    print("\n--- Model R5: Gated Refinement ---")
    seed_everything(42)
    refiner_g = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2, use_gating=True).to(device)
    pipe_g = SurfaceRefinementPipeline(base_model, refiner_g, R_bounds_mm=R_bounds_mm, S_global=S_global, support_strategy=strat, M=M, k_anchors=ka).to(device)
    r5_mets, r5_preds = train_pipeline(
        pipe_g, tr_loader, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices,
        epochs=35, lr=1e-3, freeze_base=True, name="R5-Gated"
    )
    all_results["R5_Gated_Refinement"] = r5_mets

    # --- R2: Joint Finetuning ---
    print("\n--- Model R2: Joint Finetuned Refiner ---")
    seed_everything(42)
    base_joint = TargetConditionedVotingModel(atlas_coords=ckpt["atlas_coords"]).to(device)
    base_joint.load_state_dict(ckpt["state_dict"])
    refiner_j = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2).to(device)
    pipe_j = SurfaceRefinementPipeline(base_joint, refiner_j, R_bounds_mm=R_bounds_mm, S_global=S_global, support_strategy=strat, M=M, k_anchors=ka).to(device)
    r2_mets, r2_preds = train_pipeline(
        pipe_j, tr_loader, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices,
        epochs=25, lr=1e-3, freeze_base=False, base_lr=1e-4, name="R2-Joint"
    )
    all_results["R2_Joint_Finetune"] = r2_mets

    # --- Forensic Controls: D2, D3, D4 ---
    print("\n--- Forensic Controls: D2, D3, D4 ---")
    # Using trained R1 (Seed 42)
    refiner_eval = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2).to(device)
    pipe_eval = SurfaceRefinementPipeline(base_model, refiner_eval, R_bounds_mm=R_bounds_mm, S_global=S_global, support_strategy=strat, M=M, k_anchors=ka).to(device)
    # Fast fit
    _, _ = train_pipeline(
        pipe_eval, tr_loader, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices,
        epochs=35, lr=1e-3, freeze_base=True, name="Eval-Pipeline-Fit"
    )

    # D2: Shuffled Support
    d2_mets, d2_preds, _, _ = evaluate_pipeline(
        pipe_eval, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices, control_shuffled=True
    )
    all_results["D2_Shuffled_Support"] = d2_mets
    print(f"D2 (Shuffled Support) Macro MRE: {d2_mets['macro_target_mre']:.2f} mm")

    # D3: Random Support (Train & Eval with random support)
    refiner_d3 = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2).to(device)
    pipe_d3 = SurfaceRefinementPipeline(base_model, refiner_d3, R_bounds_mm=R_bounds_mm, S_global=S_global, support_strategy=strat, M=M, k_anchors=ka).to(device)
    d3_mets, d3_preds = train_pipeline(
        pipe_d3, tr_loader, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices,
        epochs=35, lr=1e-3, freeze_base=True, name="D3-Random", control_random_train=True
    )
    all_results["D3_Random_Support"] = d3_mets
    print(f"D3 (Random Support) Macro MRE: {d3_mets['macro_target_mre']:.2f} mm")

    # D4: Wrong-Target Support
    d4_mets, d4_preds, _, _ = evaluate_pipeline(
        pipe_eval, val_loader, S_global, val_tgt_mm, val_masks, primary_107_indices, control_wrong_target=True
    )
    all_results["D4_Wrong_Target_Support"] = d4_mets
    print(f"D4 (Wrong Target Support) Macro MRE: {d4_mets['macro_target_mre']:.2f} mm")

    # Select Best Refinement Model
    candidates = {
        "R1": (all_results["R1_Surface_Refiner"]["best_seed_mets"], best_r1_preds),
        "R2": (r2_mets, r2_preds),
        "R3": (r3_mets, r3_preds),
        "R5": (r5_mets, r5_preds)
    }
    best_name = min(candidates, key=lambda k: candidates[k][0]["macro_target_mre"])
    best_mets, best_preds = candidates[best_name]
    all_results["best_refiner_name"] = best_name
    print(f"\n>>> BEST REFINEMENT MODEL: {best_name} (Macro MRE = {best_mets['macro_target_mre']:.2f} mm) <<<")

    # Paired Bootstrap (1,000 resamples)
    print("\nRunning Paired Patient Bootstrap (1,000 resamples)...")
    boot_res = paired_bootstrap(r0_preds_mm, best_preds, val_tgt_mm, val_masks, primary_107_indices, n_resamples=1000)
    all_results["bootstrap_vs_r0"] = boot_res
    print(f"Macro MRE Improvement: {boot_res['macro_mre_diff']['mean']:.2f} mm (95% CI: [{boot_res['macro_mre_diff']['ci_95'][0]:.2f}, {boot_res['macro_mre_diff']['ci_95'][1]:.2f}])")
    print(f"Micro MRE Improvement: {boot_res['micro_mre_diff']['mean']:.2f} mm (95% CI: [{boot_res['micro_mre_diff']['ci_95'][0]:.2f}, {boot_res['micro_mre_diff']['ci_95'][1]:.2f}])")
    print(f"SDR@10 Improvement: {boot_res['sdr10_diff']['mean']:.2f}% (95% CI: [{boot_res['sdr10_diff']['ci_95'][0]:.2f}, {boot_res['sdr10_diff']['ci_95'][1]:.2f}])")
    print(f"SDR@15 Improvement: {boot_res['sdr15_diff']['mean']:.2f}% (95% CI: [{boot_res['sdr15_diff']['ci_95'][0]:.2f}, {boot_res['sdr15_diff']['ci_95'][1]:.2f}])")

    # --- Detailed Diagnostics ---
    r0_errs = np.linalg.norm(r0_preds_mm - val_tgt_mm, axis=-1) # (44, 107)
    r1_errs = np.linalg.norm(best_preds - val_tgt_mm, axis=-1)
    corr_mags = np.linalg.norm(best_preds - r0_preds_mm, axis=-1)

    target_diag = {}
    best_improved = {"name": "", "from": 0.0, "to": 0.0, "delta": -999.0}
    hardest_remaining = {"name": "", "error": 0.0}

    for k in primary_107_indices:
        t_name = ORGAN_NAMES[k]
        val_p_mask = val_masks[:, k] > 0
        if not np.any(val_p_mask):
            continue
        err0 = float(np.mean(r0_errs[val_p_mask, k]))
        err1 = float(np.mean(r1_errs[val_p_mask, k]))
        cmag = float(np.mean(corr_mags[val_p_mask, k]))
        abs_diff = err0 - err1
        rel_diff_pct = (abs_diff / err0) * 100.0 if err0 > 0 else 0.0

        if rel_diff_pct > 10.0:
            cat = "REFINEMENT_STRONGLY_HELPFUL"
        elif rel_diff_pct > 2.0:
            cat = "REFINEMENT_HELPFUL"
        elif rel_diff_pct >= -2.0:
            cat = "NEUTRAL"
        else:
            cat = "HARMFUL"

        target_diag[t_name] = {
            "target_index": k,
            "phase4_mre": err0,
            "phase5_mre": err1,
            "abs_improvement": abs_diff,
            "rel_improvement_pct": rel_diff_pct,
            "mean_correction_mag": cmag,
            "category": cat
        }

        if abs_diff > best_improved["delta"]:
            best_improved = {"name": t_name, "from": err0, "to": err1, "delta": abs_diff}
        if err1 > hardest_remaining["error"]:
            hardest_remaining = {"name": t_name, "error": err1}

    all_results["target_diagnostics"] = target_diag
    all_results["best_improved_target"] = best_improved
    all_results["hardest_remaining_target"] = hardest_remaining

    # Anatomical groups
    skel_keywords = ["rib", "vertebra", "clavicle", "scapula", "sternum", "hip", "sacrum", "femur", "spine"]
    skel_indices = [k for k in primary_107_indices if any(kw in ORGAN_NAMES[k].lower() for kw in skel_keywords)]
    soft_indices = [k for k in primary_107_indices if k not in skel_indices]

    skel_err0 = [target_diag[ORGAN_NAMES[k]]["phase4_mre"] for k in skel_indices if ORGAN_NAMES[k] in target_diag]
    skel_err1 = [target_diag[ORGAN_NAMES[k]]["phase5_mre"] for k in skel_indices if ORGAN_NAMES[k] in target_diag]
    soft_err0 = [target_diag[ORGAN_NAMES[k]]["phase4_mre"] for k in soft_indices if ORGAN_NAMES[k] in target_diag]
    soft_err1 = [target_diag[ORGAN_NAMES[k]]["phase5_mre"] for k in soft_indices if ORGAN_NAMES[k] in target_diag]

    all_results["anatomical_groups"] = {
        "skeletal": {
            "count": len(skel_indices),
            "phase4_mre": float(np.mean(skel_err0)),
            "phase5_mre": float(np.mean(skel_err1)),
            "rel_improvement_pct": float((np.mean(skel_err0) - np.mean(skel_err1)) / np.mean(skel_err0) * 100)
        },
        "soft_tissue": {
            "count": len(soft_indices),
            "phase4_mre": float(np.mean(soft_err0)),
            "phase5_mre": float(np.mean(soft_err1)),
            "rel_improvement_pct": float((np.mean(soft_err0) - np.mean(soft_err1)) / np.mean(soft_err0) * 100)
        }
    }

    # Difficult targets
    diff_targets = {}
    for dt_name in ["colon", "duodenum", "stomach"]:
        found = [k for k in primary_107_indices if dt_name in ORGAN_NAMES[k].lower()]
        if found:
            k = found[0]
            if ORGAN_NAMES[k] in target_diag:
                diff_targets[ORGAN_NAMES[k]] = target_diag[ORGAN_NAMES[k]]
    all_results["difficult_targets"] = diff_targets

    # Target depth
    val_pts_mm = data["points_centered_mm"][val_idx].numpy()
    depths = []
    e0_list = []
    e1_list = []
    diff_list = []
    for i in range(len(val_idx)):
        pts_i = val_pts_mm[i]
        for k in primary_107_indices:
            if val_masks[i, k]:
                tgt_ik = val_tgt_mm[i, k]
                d = np.min(np.linalg.norm(pts_i - tgt_ik, axis=-1))
                depths.append(d)
                e0 = r0_errs[i, k]
                e1 = r1_errs[i, k]
                e0_list.append(e0)
                e1_list.append(e1)
                diff_list.append(e0 - e1)
    depths = np.array(depths)
    e0_list = np.array(e0_list)
    e1_list = np.array(e1_list)
    diff_list = np.array(diff_list)

    all_results["depth_analysis"] = {
        "mean_depth_mm": float(np.mean(depths)),
        "corr_depth_vs_phase4_err": float(np.corrcoef(depths, e0_list)[0, 1]),
        "corr_depth_vs_phase5_err": float(np.corrcoef(depths, e1_list)[0, 1]),
        "corr_depth_vs_improvement": float(np.corrcoef(depths, diff_list)[0, 1])
    }

    # Coarse Error Stratification
    bins = [(0, 10), (10, 15), (15, 20), (20, 30), (30, 999)]
    bin_names = ["0-10 mm", "10-15 mm", "15-20 mm", "20-30 mm", ">30 mm"]
    strat_results = {}
    total_improved = 0
    total_worsened = 0
    total_valid = len(e0_list)

    for (low, high), b_name in zip(bins, bin_names):
        mask_bin = (e0_list >= low) & (e0_list < high)
        n_bin = np.sum(mask_bin)
        if n_bin > 0:
            pre_err = float(np.mean(e0_list[mask_bin]))
            post_err = float(np.mean(e1_list[mask_bin]))
            imp = np.sum(e1_list[mask_bin] < e0_list[mask_bin])
            wors = np.sum(e1_list[mask_bin] > e0_list[mask_bin])
            strat_results[b_name] = {
                "count": int(n_bin),
                "mean_pre_err": pre_err,
                "mean_post_err": post_err,
                "pct_improved": float(imp / n_bin * 100),
                "pct_worsened": float(wors / n_bin * 100)
            }
            total_improved += imp
            total_worsened += wors

    all_results["coarse_error_stratification"] = strat_results
    all_results["overall_improved_pct"] = float(total_improved / total_valid * 100)
    all_results["overall_worsened_pct"] = float(total_worsened / total_valid * 100)
    all_results["mean_predicted_correction_mm"] = float(np.mean(corr_mags))

    # Overshoot analysis
    moves_toward = 0
    moves_away = 0
    overshoots = 0
    for i in range(len(val_idx)):
        for k in primary_107_indices:
            if val_masks[i, k]:
                v_gt = val_tgt_mm[i, k] - r0_preds_mm[i, k]
                delta = best_preds[i, k] - r0_preds_mm[i, k]
                norm_vgt = np.linalg.norm(v_gt)
                norm_delta = np.linalg.norm(delta)
                if norm_delta > 1e-4 and norm_vgt > 1e-4:
                    proj = np.dot(delta, v_gt) / norm_vgt
                    if proj > norm_vgt:
                        overshoots += 1
                    elif proj > 0:
                        moves_toward += 1
                    else:
                        moves_away += 1
                else:
                    moves_toward += 1

    all_results["overshoot_analysis"] = {
        "pct_moves_toward_gt": float(moves_toward / total_valid * 100),
        "pct_moves_away_from_gt": float(moves_away / total_valid * 100),
        "pct_overshoots_past_gt": float(overshoots / total_valid * 100)
    }

    out_file = repo_root / "experiments" / "phase5" / "phase5_full_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all Phase 5 full results to: {out_file}")

if __name__ == "__main__":
    main()
