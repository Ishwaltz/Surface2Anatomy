import sys
import json
import csv
from pathlib import Path
import numpy as np
from scipy import stats
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES

def compute_train_statistics():
    print("=" * 80)
    print("AUDIT: COMPUTING TRAINING-SPLIT ONLY POPULATION STATISTICS (RULE 4)")
    print("=" * 80)

    pt_path = repo_root / "sharon" / "dataset" / "pointclouds_450.pt"
    data = torch.load(str(pt_path), weights_only=False)

    case_ids = data["case_ids"]
    raw_points = data["raw_points"].numpy()       # (450, 4096, 3)
    raw_centroids = data["raw_centroids"].numpy() # (450, 121, 3)
    masks = data["masks"].numpy()                 # (450, 121)
    scales = data["scales"].numpy()               # (450, 3)
    centers = data["centers"].numpy()             # (450, 3)

    splits_file = repo_root / "sharon" / "outputs" / "splits_pointcloud.json"
    with open(splits_file) as f:
        sp = json.load(f)
    
    tr_indices = sp["train_indices"]
    val_indices = sp["val_indices"]
    te_indices = sp["test_indices"]

    print(f"Train samples: {len(tr_indices)}, Val samples: {len(val_indices)}, Test samples: {len(te_indices)}")

    # Compute bounding extents
    p_min = raw_points.min(axis=1) # (450, 3)
    p_max = raw_points.max(axis=1) # (450, 3)
    extents = p_max - p_min
    widths, depths, heights = extents[:, 0], extents[:, 1], extents[:, 2]

    # Training-only dimension statistics
    tr_w = widths[tr_indices]
    tr_d = depths[tr_indices]
    tr_h = heights[tr_indices]

    print("\n--- [DATA-VERIFIED] TRAINING SPLIT ONLY Physical Bounding Box Distributions (N=360) ---")
    print(f"  Width (X) [mm]:  mean={tr_w.mean():.2f} ± {tr_w.std():.2f}, median={np.median(tr_w):.2f}, range=[{tr_w.min():.2f}, {tr_w.max():.2f}], P5={np.percentile(tr_w, 5):.2f}, P95={np.percentile(tr_w, 95):.2f}")
    print(f"  Depth (Y) [mm]:  mean={tr_d.mean():.2f} ± {tr_d.std():.2f}, median={np.median(tr_d):.2f}, range=[{tr_d.min():.2f}, {tr_d.max():.2f}], P5={np.percentile(tr_d, 5):.2f}, P95={np.percentile(tr_d, 95):.2f}")
    print(f"  Height (Z) [mm]: mean={tr_h.mean():.2f} ± {tr_h.std():.2f}, median={np.median(tr_h):.2f}, range=[{tr_h.min():.2f}, {tr_h.max():.2f}], P5={np.percentile(tr_h, 5):.2f}, P95={np.percentile(tr_h, 95):.2f}")

    val_h = heights[val_indices]
    te_h = heights[te_indices]
    print(f"  Val Height (Z) [mm]:  mean={val_h.mean():.2f} ± {val_h.std():.2f}, median={np.median(val_h):.2f}")
    print(f"  Test Height (Z) [mm]: mean={te_h.mean():.2f} ± {te_h.std():.2f}, median={np.median(te_h):.2f}")

    # Training-only Correlations
    l1_idx = ORGAN_NAMES.index("vertebrae_L1")
    tr_l1_mask = masks[tr_indices, l1_idx] > 0.5
    h_tr_l1 = tr_h[tr_l1_mask]
    z_tr_l1 = raw_centroids[tr_indices][tr_l1_mask, l1_idx, 2]
    p_corr_h, _ = stats.pearsonr(h_tr_l1, z_tr_l1)
    s_corr_h, _ = stats.spearmanr(h_tr_l1, z_tr_l1)
    print(f"\n--- [DATA-VERIFIED] TRAINING SPLIT Anatomical Correlations (N={tr_l1_mask.sum()}) ---")
    print(f"  corr(Height H, L1 Vertebra Z-coord): Pearson r = {p_corr_h:.4f}, Spearman rho = {s_corr_h:.4f}")

    kl_idx = ORGAN_NAMES.index("kidney_left")
    kr_idx = ORGAN_NAMES.index("kidney_right")
    tr_k_mask = (masks[tr_indices, kl_idx] > 0.5) & (masks[tr_indices, kr_idx] > 0.5)
    w_tr_k = tr_w[tr_k_mask]
    sep_tr_x = np.abs(raw_centroids[tr_indices][tr_k_mask, kl_idx, 0] - raw_centroids[tr_indices][tr_k_mask, kr_idx, 0])
    p_corr_w, _ = stats.pearsonr(w_tr_k, sep_tr_x)
    s_corr_w, _ = stats.spearmanr(w_tr_k, sep_tr_x)
    print(f"  corr(Width W, Kidney Left-Right separation in X): Pearson r = {p_corr_w:.4f}, Spearman rho = {s_corr_w:.4f}")

    liv_idx = ORGAN_NAMES.index("liver")
    tr_liv_mask = masks[tr_indices, liv_idx] > 0.5
    d_tr_liv = tr_d[tr_liv_mask]
    y_tr_liv = raw_centroids[tr_indices][tr_liv_mask, liv_idx, 1]
    p_corr_d, _ = stats.pearsonr(d_tr_liv, y_tr_liv)
    s_corr_d, _ = stats.spearmanr(d_tr_liv, y_tr_liv)
    print(f"  corr(Depth D, Liver Y-coord): Pearson r = {p_corr_d:.4f}, Spearman rho = {s_corr_d:.4f}")

    # Training-only Target Statistics Table
    target_stats_train = []
    for j, name in enumerate(ORGAN_NAMES):
        tr_val_mask = masks[tr_indices, j] > 0.5
        v_count = int(tr_val_mask.sum())
        m_count = len(tr_indices) - v_count
        
        if v_count > 0:
            coords = raw_centroids[tr_indices][tr_val_mask, j]
            mx, my, mz = coords.mean(axis=0)
            sx, sy, sz = coords.std(axis=0)
        else:
            mx = my = mz = sx = sy = sz = 0.0

        target_stats_train.append({
            "target_index": j,
            "target_name": name,
            "train_valid_count": v_count,
            "train_missing_count": m_count,
            "train_mean_x": float(mx),
            "train_mean_y": float(my),
            "train_mean_z": float(mz),
            "train_std_x": float(sx),
            "train_std_y": float(sy),
            "train_std_z": float(sz),
        })

    out_csv = repo_root / "reports" / "phase1" / "target_statistics_train_split.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "target_index", "target_name", "train_valid_count", "train_missing_count",
            "train_mean_x", "train_mean_y", "train_mean_z", "train_std_x", "train_std_y", "train_std_z"
        ])
        writer.writeheader()
        writer.writerows(target_stats_train)
    print(f"✓ Saved training-only target statistics table: {out_csv} ({len(target_stats_train)} rows)")

if __name__ == "__main__":
    compute_train_statistics()
