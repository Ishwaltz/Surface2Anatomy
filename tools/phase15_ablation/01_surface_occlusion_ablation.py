#!/usr/bin/env python3
"""
tools/phase15_ablation/01_surface_occlusion_ablation.py
======================================================
Surface Occlusion and Coverage Ablation Study:
Covers/masks specific anatomical regions of the input 3D body surface point cloud
and measures the impact on organ localization across all 104 primary benchmark targets.

Evaluates 11 conditions:
1. Full 360 Reference (0% occlusion)
2. Anterior Covered (Y > 0 removed: front/belly covered)
3. Posterior Covered (Y < 0 removed: back/spine covered, e.g. lying supine)
4. Superior Covered (Z > 0 removed: chest/shoulders covered)
5. Inferior Covered (Z < 0 removed: pelvis/lower abdomen covered)
6. Right Flank Covered (X > 0 removed)
7. Left Flank Covered (X < 0 removed)
8. Central Abdomen Drape Covered (|X| < 80mm, Y > 0, |Z| < 80mm: surgical drape / ultrasound window)
9. Random Drop 25% (sensor sparsity)
10. Random Drop 50%
11. Random Drop 75%
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from sharon.model_target_query import TargetQueryTransformerDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = repo_root / "reports" / "phase15_ablation"
out_dir.mkdir(parents=True, exist_ok=True)

NUM_POINTS = 4096
S_GLOBAL_MM = 500.0

def apply_occlusion(pts_mm: np.ndarray, mode: str, seed: int = 42) -> np.ndarray:
    """
    pts_mm: (4096, 3) in physical millimeter centered space
    Axes: +X Right, +Y Anterior, +Z Superior
    """
    np.random.seed(seed)
    N = len(pts_mm)
    
    if mode == "Full_360_Reference":
        mask = np.ones(N, dtype=bool)
    elif mode == "Anterior_Covered":
        # Cover front (Y > 0)
        mask = pts_mm[:, 1] <= 0
    elif mode == "Posterior_Covered":
        # Cover back (Y < 0)
        mask = pts_mm[:, 1] >= 0
    elif mode == "Superior_Covered":
        # Cover upper thorax/shoulders (Z > 0)
        mask = pts_mm[:, 2] <= 0
    elif mode == "Inferior_Covered":
        # Cover pelvis/lower abdomen (Z < 0)
        mask = pts_mm[:, 2] >= 0
    elif mode == "Right_Flank_Covered":
        # Cover patient right side (X > 0)
        mask = pts_mm[:, 0] <= 0
    elif mode == "Left_Flank_Covered":
        # Cover patient left side (X < 0)
        mask = pts_mm[:, 0] >= 0
    elif mode == "Central_Drape_Covered":
        # Cover central anterior abdomen (|X| < 80mm, Y > 0, |Z| < 80mm)
        in_drape = (np.abs(pts_mm[:, 0]) < 80.0) & (pts_mm[:, 1] > 0.0) & (np.abs(pts_mm[:, 2]) < 80.0)
        mask = ~in_drape
    elif mode == "Random_Drop_25":
        keep_n = int(N * 0.75)
        keep_idx = np.random.choice(N, size=keep_n, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[keep_idx] = True
    elif mode == "Random_Drop_50":
        keep_n = int(N * 0.50)
        keep_idx = np.random.choice(N, size=keep_n, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[keep_idx] = True
    elif mode == "Random_Drop_75":
        keep_n = int(N * 0.25)
        keep_idx = np.random.choice(N, size=keep_n, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[keep_idx] = True
    else:
        raise ValueError(f"Unknown occlusion mode: {mode}")

    surviving = pts_mm[mask]
    if len(surviving) < 64:
        surviving = pts_mm # Fail-safe if too few points

    # Uniformly resample back to 4096 points (standard 3D point cloud protocol)
    resample_idx = np.random.choice(len(surviving), size=NUM_POINTS, replace=(len(surviving) < NUM_POINTS))
    return surviving[resample_idx].astype(np.float32)

class OcclusionDataset(Dataset):
    def __init__(self, pts_list, mode: str, S_global: float = 500.0):
        self.samples = []
        for i, p in enumerate(pts_list):
            p_occ = apply_occlusion(p, mode=mode, seed=42 + i)
            p_norm = p_occ / S_global
            self.samples.append(torch.from_numpy(p_norm).float())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"pts": self.samples[idx]}

def load_models():
    models = []
    for s in [42, 43, 44]:
        ckpt_path = repo_root / f"experiments/phase10R/checkpoints/C4_Proposed_seed{s}.pt"
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = TargetQueryTransformerDecoder(atlas_coords=torch.zeros(117, 3).to(device), num_organs=117).to(device)
        m.load_state_dict(saved["model_state_dict"])
        m.eval()
        models.append(m)
    return models

def evaluate_mode(models, loader, tgts_test, masks_test, bench_idx):
    # Predict with each seed
    seed_preds = []
    with torch.no_grad():
        for m in models:
            preds_m = []
            for batch in loader:
                p, _ = m(batch["pts"].to(device))
                preds_m.append(p.cpu().numpy() * S_GLOBAL_MM)
            seed_preds.append(np.concatenate(preds_m, axis=0))
    
    # Ensemble average
    ens_preds = np.mean(seed_preds, axis=0) # (B, 117, 3)
    
    # Evaluate across primary 104 benchmark targets
    pred_sub = ens_preds[:, bench_idx, :]
    tgt_sub = tgts_test[:, bench_idx, :]
    mask_sub = masks_test[:, bench_idx]
    
    errors = np.linalg.norm(pred_sub - tgt_sub, axis=-1) # (B, 104)
    valid = (mask_sub == 1)
    
    target_mres = {}
    for i, t_slot in enumerate(bench_idx):
        v = (mask_sub[:, i] == 1)
        if np.sum(v) > 0:
            target_mres[t_slot] = float(np.mean(errors[v, i]))
        else:
            target_mres[t_slot] = np.nan
            
    macro_mre = float(np.nanmean(list(target_mres.values())))
    micro_mre = float(np.mean(errors[valid]))
    median_err = float(np.median(errors[valid]))
    p90_err = float(np.percentile(errors[valid], 90))
    sdr10 = float(np.mean(errors[valid] <= 10.0) * 100.0)
    sdr20 = float(np.mean(errors[valid] <= 20.0) * 100.0)
    
    return {
        "macro_mre": macro_mre,
        "micro_mre": micro_mre,
        "median": median_err,
        "p90": p90_err,
        "sdr10": sdr10,
        "sdr20": sdr20,
        "target_mres": target_mres
    }

def main():
    print("=" * 80)
    print("PHASE 15: SURFACE OCCLUSION & POINT CLOUD COVERAGE ABLATION STUDY")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)
    
    # Load dataset
    data_path = repo_root / "sharon/dataset_v3/pointclouds_v3.pt"
    splits_path = repo_root / "sharon/dataset_v3/splits_v3_iid.json"
    
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)
        
    test_idx = splits["test_indices"]
    pts_test = [p.numpy() for p in data["points_centered_4096"][test_idx]]
    tgts_test = data["targets_centered"][test_idx].numpy()
    masks_test = data["target_masks"][test_idx].numpy()
    bench_idx = data["primary_104_indices"].tolist()
    target_names = data["canonical_target_names"]
    
    print(f"Loaded held-out test split: {len(pts_test)} cases.")
    print("Loading frozen Phase 10R ensemble checkpoints (Seeds 42, 43, 44)...")
    models = load_models()
    print("Models loaded successfully.\n")
    
    MODES = [
        ("Full_360_Reference", "Full 360° Body Surface (Reference)", 0.0),
        ("Anterior_Covered", "Anterior Covered (Chest & Belly hidden; Y > 0 removed)", 50.0),
        ("Posterior_Covered", "Posterior Covered (Back & Spine hidden; Y < 0 removed)", 50.0),
        ("Superior_Covered", "Superior Covered (Upper Thorax hidden; Z > 0 removed)", 50.0),
        ("Inferior_Covered", "Inferior Covered (Pelvis & Lower Torso hidden; Z < 0 removed)", 50.0),
        ("Right_Flank_Covered", "Right Flank Covered (Right side hidden; X > 0 removed)", 50.0),
        ("Left_Flank_Covered", "Left Flank Covered (Left side hidden; X < 0 removed)", 50.0),
        ("Central_Drape_Covered", "Central Drape Covered (80mm anterior patch hidden)", 8.5),
        ("Random_Drop_25", "Sparse Sensor Decimation (25% points randomly dropped)", 25.0),
        ("Random_Drop_50", "Sparse Sensor Decimation (50% points randomly dropped)", 50.0),
        ("Random_Drop_75", "Sparse Sensor Decimation (75% points randomly dropped)", 75.0),
    ]
    
    summary_rows = []
    target_records = []
    ref_target_mres = {}
    
    for mode_id, desc, pct_hidden in MODES:
        t0 = time.time()
        print(f"Running condition: {mode_id} ({desc})...")
        ds = OcclusionDataset(pts_test, mode=mode_id, S_global=S_GLOBAL_MM)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        
        metrics = evaluate_mode(models, loader, tgts_test, masks_test, bench_idx)
        elapsed = time.time() - t0
        
        if mode_id == "Full_360_Reference":
            ref_macro = metrics["macro_mre"]
            ref_target_mres = metrics["target_mres"].copy()
            delta_macro = 0.0
        else:
            delta_macro = metrics["macro_mre"] - ref_macro
            
        print(f"  -> Macro MRE: {metrics['macro_mre']:.2f} mm (Δ = {delta_macro:+.2f} mm) | Median: {metrics['median']:.2f} mm | SDR@10: {metrics['sdr10']:.1f}% [{elapsed:.1f}s]")
        
        summary_rows.append({
            "mode_id": mode_id,
            "description": desc,
            "surface_hidden_percent": pct_hidden,
            "macro_mre_mm": round(metrics["macro_mre"], 2),
            "micro_mre_mm": round(metrics["micro_mre"], 2),
            "median_mm": round(metrics["median"], 2),
            "p90_mm": round(metrics["p90"], 2),
            "delta_macro_mm": round(delta_macro, 2),
            "sdr10_percent": round(metrics["sdr10"], 1),
            "sdr20_percent": round(metrics["sdr20"], 1)
        })
        
        # Record target-wise errors
        for slot in bench_idx:
            t_name = target_names[slot]
            t_mre = metrics["target_mres"].get(slot, np.nan)
            t_ref = ref_target_mres.get(slot, np.nan)
            t_delta = (t_mre - t_ref) if not np.isnan(t_mre) and not np.isnan(t_ref) else np.nan
            target_records.append({
                "mode_id": mode_id,
                "target_slot": slot,
                "target_name": t_name,
                "mre_mm": round(t_mre, 2) if not np.isnan(t_mre) else None,
                "reference_mre_mm": round(t_ref, 2) if not np.isnan(t_ref) else None,
                "delta_mre_mm": round(t_delta, 2) if not np.isnan(t_delta) else None
            })
            
    # Save results
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "surface_occlusion_summary.csv", index=False)
    
    df_targets = pd.DataFrame(target_records)
    df_targets.to_csv(out_dir / "surface_occlusion_per_target.csv", index=False)
    
    print("\n" + "=" * 80)
    print("SURFACE OCCLUSION ABLATION SUMMARY TABLE:")
    print("=" * 80)
    print(df_summary[["mode_id", "surface_hidden_percent", "macro_mre_mm", "delta_macro_mm", "median_mm", "sdr10_percent"]].to_string(index=False))
    print("\nSaved summary to:", out_dir / "surface_occlusion_summary.csv")
    print("Saved target records to:", out_dir / "surface_occlusion_per_target.csv")

if __name__ == "__main__":
    main()
