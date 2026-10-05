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
from sharon.support_selection import select_support_tokens
from tools.phase3.run_phase3_experiments import Phase3Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SupportCacheDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.n = len(data_dict["p0"])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: self.data_dict[k][idx] for k in self.data_dict}

def extract_base_support(base_model, loader, S_global, normals_all, split_indices, strat, M, ka):
    base_model.eval()
    p0_list = []
    h_queries_list = []
    voter_xyz_list = []
    voter_feat_list = []
    scores_list = []
    voter_normals_list = []
    rand_voter_xyz_list = []
    rand_voter_feat_list = []
    rand_scores_list = []
    d_qv_list = []
    disp_list = []
    tgt_list = []
    prim_list = []

    with torch.no_grad():
        for b_idx, b in enumerate(loader):
            pts = b["pts"].to(device)
            tgt = b["tgt"].to(device)
            prim = b["prim"].to(device)
            meta = b["meta"].to(device)

            out = base_model(pts, metadata=meta)
            B_cur = pts.shape[0]
            K = base_model.num_organs

            p0 = out["p_final"]
            h_queries = out["queries"]
            mem_xyz = out["mem_xyz"]
            mem_feat = out["mem_tokens"]
            attn_w = out["last_attn_weights"]
            disp = out["dispersion"].unsqueeze(-1)
            d_qv = torch.norm(out["p_query"] - out["p_vote"], dim=-1, keepdim=True)

            # Match normals
            diff = mem_xyz.unsqueeze(2) - pts.unsqueeze(1)
            nn_idx = torch.argmin(torch.norm(diff, dim=-1), dim=-1)
            start_i = b_idx * loader.batch_size
            end_i = start_i + B_cur
            cur_raw_normals = torch.tensor(normals_all[split_indices[start_i:end_i]], device=device, dtype=torch.float32)
            nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, 3)
            mem_normals = torch.gather(cur_raw_normals, 1, nn_idx_exp)

            # Support tokens
            sup_out = select_support_tokens(
                mem_xyz=mem_xyz,
                mem_feat=mem_feat,
                attn_weights=attn_w,
                p0=p0,
                strategy=strat,
                M=M,
                k_anchors=ka,
                normals=mem_normals
            )

            # Random tokens for D3
            N_tokens = mem_xyz.shape[1]
            rand_idx = torch.randint(0, N_tokens, (B_cur, K, M), device=device)
            rand_idx_xyz = rand_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            rand_idx_feat = rand_idx.unsqueeze(-1).expand(-1, -1, -1, mem_feat.shape[-1])
            rand_xyz = torch.gather(mem_xyz.unsqueeze(1).expand(-1, K, -1, -1), 2, rand_idx_xyz)
            rand_feat = torch.gather(mem_feat.unsqueeze(1).expand(-1, K, -1, -1), 2, rand_idx_feat)
            rand_sc = torch.full_like(sup_out["scores"], 1.0 / M)

            p0_list.append(p0.cpu())
            h_queries_list.append(h_queries.cpu())
            voter_xyz_list.append(sup_out["voter_xyz"].cpu())
            voter_feat_list.append(sup_out["voter_feat"].cpu())
            scores_list.append(sup_out["scores"].cpu())
            voter_normals_list.append(sup_out["voter_normals"].cpu())
            rand_voter_xyz_list.append(rand_xyz.cpu())
            rand_voter_feat_list.append(rand_feat.cpu())
            rand_scores_list.append(rand_sc.cpu())
            d_qv_list.append(d_qv.cpu())
            disp_list.append(disp.cpu())
            tgt_list.append(tgt.cpu())
            prim_list.append(prim.cpu())

    return {
        "p0": torch.cat(p0_list, dim=0),
        "h_queries": torch.cat(h_queries_list, dim=0),
        "voter_xyz": torch.cat(voter_xyz_list, dim=0),
        "voter_feat": torch.cat(voter_feat_list, dim=0),
        "scores": torch.cat(scores_list, dim=0),
        "voter_normals": torch.cat(voter_normals_list, dim=0),
        "rand_voter_xyz": torch.cat(rand_voter_xyz_list, dim=0),
        "rand_voter_feat": torch.cat(rand_voter_feat_list, dim=0),
        "rand_scores": torch.cat(rand_scores_list, dim=0),
        "d_qv": torch.cat(d_qv_list, dim=0),
        "disp": torch.cat(disp_list, dim=0),
        "tgt": torch.cat(tgt_list, dim=0),
        "prim": torch.cat(prim_list, dim=0)
    }

def train_refiner_fast(
    refiner, tr_loader, val_loader, atlas_coords, R_bounds_m, S_global, val_tgt_mm, val_masks, primary_107_indices,
    epochs=35, lr=1e-3, use_normals=False, use_random=False, name="Refiner"
):
    opt = torch.optim.AdamW(refiner.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_macro = 999.0
    best_preds_mm = None
    best_mets = None

    t0 = time.time()
    for ep in range(epochs):
        refiner.train()
        for b in tr_loader:
            p0 = b["p0"].to(device)
            h_queries = b["h_queries"].to(device)
            tgt = b["tgt"].to(device)
            prim = b["prim"].to(device)
            d_qv = b["d_qv"].to(device)
            disp = b["disp"].to(device)

            if use_random:
                v_xyz = b["rand_voter_xyz"].to(device)
                v_feat = b["rand_voter_feat"].to(device)
                scores = b["rand_scores"].to(device)
                v_norm = None
            else:
                v_xyz = b["voter_xyz"].to(device)
                v_feat = b["voter_feat"].to(device)
                scores = b["scores"].to(device)
                v_norm = b["voter_normals"].to(device) if use_normals else None

            opt.zero_grad()
            out = refiner(
                p0=p0,
                h_queries=h_queries,
                atlas_coords=atlas_coords,
                voter_xyz=v_xyz,
                voter_feat=v_feat,
                scores=scores,
                R_bounds=R_bounds_m,
                normals=v_norm,
                d_qv=d_qv,
                dispersion=disp
            )
            p1 = out["p1"]
            loss = (torch.norm(p1 - tgt, dim=-1) * prim).sum() / prim.sum().clamp(min=1.0)
            loss += 0.01 * (torch.norm(out["delta_p"], dim=-1) * prim).sum() / prim.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()
        sched.step()

        # Val eval
        refiner.eval()
        preds_list = []
        with torch.no_grad():
            for b in val_loader:
                p0 = b["p0"].to(device)
                h_queries = b["h_queries"].to(device)
                d_qv = b["d_qv"].to(device)
                disp = b["disp"].to(device)
                if use_random:
                    v_xyz = b["rand_voter_xyz"].to(device)
                    v_feat = b["rand_voter_feat"].to(device)
                    scores = b["rand_scores"].to(device)
                    v_norm = None
                else:
                    v_xyz = b["voter_xyz"].to(device)
                    v_feat = b["voter_feat"].to(device)
                    scores = b["scores"].to(device)
                    v_norm = b["voter_normals"].to(device) if use_normals else None

                out = refiner(
                    p0=p0,
                    h_queries=h_queries,
                    atlas_coords=atlas_coords,
                    voter_xyz=v_xyz,
                    voter_feat=v_feat,
                    scores=scores,
                    R_bounds=R_bounds_m,
                    normals=v_norm,
                    d_qv=d_qv,
                    dispersion=disp
                )
                preds_list.append((out["p1"] * S_global).cpu().numpy())

        preds_mm = np.concatenate(preds_list, axis=0)
        mets = compute_all_metrics(preds_mm, val_tgt_mm, val_masks, primary_107_indices)
        if mets["macro_target_mre"] < best_macro:
            best_macro = mets["macro_target_mre"]
            best_preds_mm = preds_mm
            best_mets = mets

    elapsed = time.time() - t0
    print(f"[{name}] Best Validation Macro MRE: {best_macro:.2f} mm (Trained in {elapsed:.1f}s)")
    return best_mets, best_preds_mm

def eval_refiner_control(refiner, val_loader, atlas_coords, R_bounds_m, S_global, val_tgt_mm, val_masks, primary_107_indices,
                         shuffle_patient=False, wrong_target=False, name="Control"):
    refiner.eval()
    preds_list = []
    with torch.no_grad():
        for b in val_loader:
            p0 = b["p0"].to(device)
            h_queries = b["h_queries"].to(device)
            v_xyz = b["voter_xyz"].to(device)
            v_feat = b["voter_feat"].to(device)
            scores = b["scores"].to(device)
            d_qv = b["d_qv"].to(device)
            disp = b["disp"].to(device)

            if shuffle_patient:
                perm_p = torch.randperm(p0.shape[0], device=device)
                v_xyz = v_xyz[perm_p]
                v_feat = v_feat[perm_p]
                scores = scores[perm_p]

            if wrong_target:
                v_xyz = torch.roll(v_xyz, shifts=1, dims=1)
                v_feat = torch.roll(v_feat, shifts=1, dims=1)
                scores = torch.roll(scores, shifts=1, dims=1)

            out = refiner(
                p0=p0,
                h_queries=h_queries,
                atlas_coords=atlas_coords,
                voter_xyz=v_xyz,
                voter_feat=v_feat,
                scores=scores,
                R_bounds=R_bounds_m,
                d_qv=d_qv,
                dispersion=disp
            )
            preds_list.append((out["p1"] * S_global).cpu().numpy())

    preds_mm = np.concatenate(preds_list, axis=0)
    mets = compute_all_metrics(preds_mm, val_tgt_mm, val_masks, primary_107_indices)
    print(f"[{name}] Macro MRE: {mets['macro_target_mre']:.2f} mm")
    return mets, preds_mm

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
    print("PHASE 5: ULTRA-FAST UNIFIED FULL-COHORT EXPERIMENTS & SCIENTIFIC BENCHMARK")
    print("=" * 80)

    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
    tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"
    ckpt_path = repo_root / "experiments" / "phase5" / "r0_phase4_base_seed42.pt"
    base_res = json.load(open(repo_root / "experiments" / "phase5" / "step1_baseline_results.json"))
    ablation_res = json.load(open(repo_root / "experiments" / "phase5" / "inner_dev_ablation_results.json"))

    strat = ablation_res["best_strategy"]
    ka = ablation_res["best_k_anchors"]
    M = ablation_res["best_M"]
    print(f"Frozen Support Config from Inner-Dev: Strategy={strat}, K_anchors={ka}, M={M}")

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
    normals_all = data["normals"].numpy()

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

    tr_ds_raw = Phase3Dataset(pts_m[tr_idx], tgt_m[tr_idx], prim_mask[tr_idx], meta_norm[tr_idx], augment=False)
    val_ds_raw = Phase3Dataset(pts_m[val_idx], tgt_m[val_idx], prim_mask[val_idx], meta_norm[val_idx], augment=False)

    tr_loader_raw = DataLoader(tr_ds_raw, batch_size=16, shuffle=False)
    val_loader_raw = DataLoader(val_ds_raw, batch_size=16, shuffle=False)

    val_tgt_mm = tgt_c_mm[val_idx]
    val_masks = prim_mask[val_idx]

    # Load base model
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    atlas_coords = ckpt["atlas_coords"].to(device)
    base_model = TargetConditionedVotingModel(atlas_coords=atlas_coords).to(device)
    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()

    R_bounds_mm = torch.tensor(base_res["R_k_p95_mm"]).float().to(device)
    R_bounds_m = (R_bounds_mm / S_global).view(-1, 1)

    print("\nPre-extracting support representations (One-Time Pass)...")
    t0 = time.time()
    tr_cache = extract_base_support(base_model, tr_loader_raw, S_global, normals_all, tr_idx, strat, M, ka)
    val_cache = extract_base_support(base_model, val_loader_raw, S_global, normals_all, val_idx, strat, M, ka)
    print(f"Precomputed support tensors in {time.time() - t0:.2f}s!")

    tr_loader = DataLoader(SupportCacheDataset(tr_cache), batch_size=16, shuffle=True)
    val_loader = DataLoader(SupportCacheDataset(val_cache), batch_size=16, shuffle=False)

    all_results = {}

    # --- R0: Phase 4 Base Reproduction ---
    print("\n--- Model R0: Phase 4 Base Reproduction ---")
    r0_preds_mm = (val_cache["p0"] * S_global).numpy()
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
        mets, preds = train_refiner_fast(
            refiner, tr_loader, val_loader, atlas_coords, R_bounds_m, S_global, val_tgt_mm, val_masks, primary_107_indices,
            epochs=35, lr=1e-3, name=f"R1-Seed{s}"
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
    r3_mets, r3_preds = train_refiner_fast(
        refiner_n, tr_loader, val_loader, atlas_coords, R_bounds_m, S_global, val_tgt_mm, val_masks, primary_107_indices,
        epochs=35, lr=1e-3, use_normals=True, name="R3-Normals"
    )
    all_results["R3_Plus_Normals"] = r3_mets

    # --- R5: Gated Refinement ---
    print("\n--- Model R5: Gated Refinement ---")
    seed_everything(42)
    refiner_g = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2, use_gating=True).to(device)
    r5_mets, r5_preds = train_refiner_fast(
        refiner_g, tr_loader, val_loader, atlas_coords, R_bounds_m, S_global, val_tgt_mm, val_masks, primary_107_indices,
        epochs=35, lr=1e-3, name="R5-Gated"
    )
    all_results["R5_Gated_Refinement"] = r5_mets

    # --- D2: Shuffled Support Control ---
    print("\n--- Control D2: Shuffled Support Control ---")
    d2_mets, d2_preds = eval_refiner_control(
        refiner, val_loader, atlas_coords, R_bounds_m, S_global, val_tgt_mm, val_masks, primary_107_indices,
        shuffle_patient=True, name="D2-Shuffled"
    )
    all_results["D2_Shuffled_Support"] = d2_mets

    # --- D3: Random Support Control ---
    print("\n--- Control D3: Random Support Control ---")
    seed_everything(42)
    refiner_d3 = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2).to(device)
    d3_mets, d3_preds = train_refiner_fast(
        refiner_d3, tr_loader, val_loader, atlas_coords, R_bounds_m, S_global, val_tgt_mm, val_masks, primary_107_indices,
        epochs=35, lr=1e-3, use_random=True, name="D3-Random"
    )
    all_results["D3_Random_Support"] = d3_mets

    # --- D4: Wrong-Target Support Control ---
    print("\n--- Control D4: Wrong-Target Support Control ---")
    d4_mets, d4_preds = eval_refiner_control(
        refiner, val_loader, atlas_coords, R_bounds_m, S_global, val_tgt_mm, val_masks, primary_107_indices,
        wrong_target=True, name="D4-Wrong-Target"
    )
    all_results["D4_Wrong_Target_Support"] = d4_mets

    # --- R2: Joint Finetuning ---
    print("\n--- Model R2: Joint Finetuned Refiner (20 epochs) ---")
    seed_everything(42)
    base_joint = TargetConditionedVotingModel(atlas_coords=atlas_coords).to(device)
    base_joint.load_state_dict(ckpt["state_dict"])
    refiner_j = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2).to(device)
    pipe_j = SurfaceRefinementPipeline(base_joint, refiner_j, R_bounds_mm=R_bounds_mm, S_global=S_global, support_strategy=strat, M=M, k_anchors=ka).to(device)

    opt_j = torch.optim.AdamW([
        {"params": pipe_j.base_model.parameters(), "lr": 1e-4},
        {"params": pipe_j.refiner.parameters(), "lr": 1e-3}
    ], weight_decay=1e-4)
    sched_j = torch.optim.lr_scheduler.CosineAnnealingLR(opt_j, T_max=20)

    best_r2_macro = 999.0
    best_r2_preds = None
    t0 = time.time()
    for ep in range(20):
        pipe_j.train()
        for b in tr_loader_raw:
            pts = b["pts"].to(device)
            tgt = b["tgt"].to(device)
            prim = b["prim"].to(device)
            meta = b["meta"].to(device)

            opt_j.zero_grad()
            out = pipe_j(pts, metadata=meta, freeze_base=False)
            p1 = out["p1"]
            p0 = out["p0"]
            loss_final = (torch.norm(p1 - tgt, dim=-1) * prim).sum() / prim.sum().clamp(min=1.0)
            loss_coarse = (torch.norm(p0 - tgt, dim=-1) * prim).sum() / prim.sum().clamp(min=1.0)
            loss_delta = 0.01 * (torch.norm(out["delta_p"], dim=-1) * prim).sum() / prim.sum().clamp(min=1.0)
            total_loss = loss_final + 0.2 * loss_coarse + loss_delta
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe_j.parameters(), 1.0)
            opt_j.step()
        sched_j.step()

        pipe_j.eval()
        preds_list = []
        with torch.no_grad():
            for b in val_loader_raw:
                pts = b["pts"].to(device)
                meta = b["meta"].to(device)
                out = pipe_j(pts, metadata=meta, freeze_base=False)
                preds_list.append((out["p1"] * S_global).cpu().numpy())
        preds_mm = np.concatenate(preds_list, axis=0)
        mets = compute_all_metrics(preds_mm, val_tgt_mm, val_masks, primary_107_indices)
        if mets["macro_target_mre"] < best_r2_macro:
            best_r2_macro = mets["macro_target_mre"]
            best_r2_preds = preds_mm

    r2_mets = compute_all_metrics(best_r2_preds, val_tgt_mm, val_masks, primary_107_indices)
    all_results["R2_Joint_Finetune"] = r2_mets
    print(f"[R2-Joint] Best Validation Macro MRE: {best_r2_macro:.2f} mm (Trained in {time.time() - t0:.1f}s)")

    # Select Best Refinement Model
    candidates = {
        "R1": (all_results["R1_Surface_Refiner"]["best_seed_mets"], best_r1_preds),
        "R2": (r2_mets, best_r2_preds),
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
    r0_errs = np.linalg.norm(r0_preds_mm - val_tgt_mm, axis=-1)
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
