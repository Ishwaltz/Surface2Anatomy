import sys
import json
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES
from tools.phase2.metrics import compute_all_metrics
from sharon.model_voting import TargetConditionedVotingModel
from tools.phase3.run_phase3_experiments import Phase3Dataset
from tools.phase4.run_phase4_voting_experiments import train_voting_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 80)
    print("PHASE 5 - STAGE 0: REPRODUCE PHASE 4 BASE MODEL (R0)")
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

    # Train Phase 4 Base (Seed 42)
    print("\nTraining Phase 4 V4 Base (Seed 42)...")
    torch.manual_seed(42)
    m_base = TargetConditionedVotingModel(
        atlas_coords=atlas_coords, num_organs=121, d_model=256, nhead=8, num_layers=4,
        top_m=64, voting_mode="gated", use_metadata=True, use_geo_bias=True, use_self_attn=True
    ).to(device)

    mets, preds_mm, trained_model = train_voting_model(
        tr_loader, val_loader, m_base, epochs=65, S_global=S_global,
        primary_indices=primary_107_indices, val_tgt_c=val_tgt_c,
        model_name="R0_PHASE4_BASE", is_gated=True, is_uniform=False
    )

    macro_mre = mets["macro_target_mre"]
    micro_mre = mets["micro_mre"]
    sdr10 = mets["sdr_10"]
    sdr15 = float((np.abs(preds_mm - val_tgt_c) < 15.0).mean() * 100.0)

    print(f"\n[R0 Result] Macro MRE = {macro_mre:.2f} mm | Micro = {micro_mre:.2f} mm | SDR@10 = {sdr10:.2f}% | SDR@15 = {sdr15:.2f}%")

    if macro_mre > 18.5:
        print("ERROR: Reproduction failed! Macro MRE > 18.5 mm.")
        sys.exit(1)

    # Save model checkpoint
    ckpt_path = repo_root / "experiments" / "phase5" / "r0_phase4_base_seed42.pt"
    torch.save({
        "state_dict": trained_model.state_dict(),
        "atlas_coords": atlas_coords,
        "tr_meta_mean": tr_meta_mean,
        "tr_meta_std": tr_meta_std,
        "S_global": S_global,
        "mets": mets
    }, str(ckpt_path))
    print(f"Checkpoint successfully saved to: {ckpt_path}")

    # Generate and save precomputed coarse predictions p_k^0 and features for train and val
    print("\nPrecomputing coarse predictions and features for train and validation...")
    trained_model.eval()
    
    # Val evaluation dataloader (without augmentation)
    val_loader_eval = DataLoader(Phase3Dataset(pts_m[val_idx], tgt_m[val_idx], prim_mask[val_idx], meta_norm[val_idx], augment=False), batch_size=16, shuffle=False)
    # Train evaluation dataloader (without augmentation)
    tr_loader_eval = DataLoader(Phase3Dataset(pts_m[tr_idx], tgt_m[tr_idx], prim_mask[tr_idx], meta_norm[tr_idx], augment=False), batch_size=16, shuffle=False)

    def extract_outputs(loader):
        p_final_list = []
        p_query_list = []
        p_vote_list = []
        disp_list = []
        neff_list = []
        with torch.no_grad():
            for b in loader:
                pts = b["pts"].to(device)
                meta = b["meta"].to(device)
                res = trained_model(pts, metadata=meta)
                p_final_list.append(res["p_final"].cpu().numpy())
                p_query_list.append(res["p_query"].cpu().numpy())
                p_vote_list.append(res["p_vote"].cpu().numpy())
                disp_list.append(res["dispersion"].cpu().numpy())
                neff_list.append(res["n_eff"].cpu().numpy())
        return {
            "p_final": np.concatenate(p_final_list, axis=0),
            "p_query": np.concatenate(p_query_list, axis=0),
            "p_vote": np.concatenate(p_vote_list, axis=0),
            "dispersion": np.concatenate(disp_list, axis=0),
            "n_eff": np.concatenate(neff_list, axis=0)
        }

    val_outs = extract_outputs(val_loader_eval)
    tr_outs = extract_outputs(tr_loader_eval)

    coarse_path = repo_root / "experiments" / "phase5" / "coarse_predictions_r0.pt"
    torch.save({
        "val_coarse": val_outs,
        "tr_coarse": tr_outs,
        "val_tgt_c_mm": val_tgt_c,
        "tr_tgt_m": tr_tgt_m,
        "tr_prim_mask": prim_mask[tr_idx],
        "val_prim_mask": prim_mask[val_idx],
        "S_global": S_global
    }, str(coarse_path))
    print(f"Coarse outputs saved to: {coarse_path}")

if __name__ == "__main__":
    main()
