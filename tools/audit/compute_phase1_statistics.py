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

def compute_phase1_statistics():
    print("=" * 80)
    print("AUDIT: COMPUTING COMPREHENSIVE PHASE 1 STATISTICAL TABLES")
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
    tr_cases = set(sp["train_cases"])
    val_cases = set(sp["val_cases"])
    te_cases = set(sp["test_cases"])

    N = len(case_ids)

    # 1. Patient Geometry Table
    patient_geo_rows = []
    widths = []
    depths = []
    heights = []

    for i in range(N):
        cid = case_ids[i]
        split = "train" if cid in tr_cases else ("val" if cid in val_cases else "test")
        pts = raw_points[i]
        
        p_min = pts.min(axis=0)
        p_max = pts.max(axis=0)
        extent = p_max - p_min
        
        w, d, h = extent[0], extent[1], extent[2]
        widths.append(w)
        depths.append(d)
        heights.append(h)

        # Unique point count
        u_pts = len(np.unique(np.round(pts, 4), axis=0))
        n_valid = int(masks[i].sum())

        patient_geo_rows.append({
            "patient_id": cid,
            "split": split,
            "surface_point_count": len(pts),
            "unique_point_count": u_pts,
            "bbox_width_mm": float(w),
            "bbox_depth_mm": float(d),
            "bbox_height_mm": float(h),
            "surface_center_x": float(centers[i, 0]),
            "surface_center_y": float(centers[i, 1]),
            "surface_center_z": float(centers[i, 2]),
            "num_valid_targets": n_valid,
        })

    geo_csv = repo_root / "reports" / "phase1" / "patient_geometry.csv"
    with open(geo_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patient_id", "split", "surface_point_count", "unique_point_count",
            "bbox_width_mm", "bbox_depth_mm", "bbox_height_mm",
            "surface_center_x", "surface_center_y", "surface_center_z", "num_valid_targets"
        ])
        writer.writeheader()
        writer.writerows(patient_geo_rows)
    print(f"✓ Saved patient geometry table: {geo_csv} ({len(patient_geo_rows)} rows)")

    # 2. Target Statistics Table
    target_stats_rows = []
    for j, name in enumerate(ORGAN_NAMES):
        val_mask = masks[:, j] > 0.5
        v_count = int(val_mask.sum())
        m_count = N - v_count
        
        if v_count > 0:
            coords = raw_centroids[val_mask, j]
            mx, my, mz = coords.mean(axis=0)
            sx, sy, sz = coords.std(axis=0)
        else:
            mx = my = mz = sx = sy = sz = 0.0

        target_stats_rows.append({
            "target_index": j,
            "target_name": name,
            "valid_count": v_count,
            "missing_count": m_count,
            "mean_x": float(mx),
            "mean_y": float(my),
            "mean_z": float(mz),
            "std_x": float(sx),
            "std_y": float(sy),
            "std_z": float(sz),
        })

    target_csv = repo_root / "reports" / "phase1" / "target_statistics.csv"
    with open(target_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "target_index", "target_name", "valid_count", "missing_count",
            "mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"
        ])
        writer.writeheader()
        writer.writerows(target_stats_rows)
    print(f"✓ Saved target statistics table: {target_csv} ({len(target_stats_rows)} rows)")

    # 3. Body Scale Variability & Correlations
    widths = np.array(widths)
    depths = np.array(depths)
    heights = np.array(heights)

    print("\nPhysical Bounding Box Summary Statistics (Corrupted Space):")
    print(f"  Width (X):  mean={widths.mean():.2f}, std={widths.std():.2f}, min={widths.min():.2f}, max={widths.max():.2f}, P5={np.percentile(widths, 5):.2f}, P95={np.percentile(widths, 95):.2f} mm")
    print(f"  Depth (Y):  mean={depths.mean():.2f}, std={depths.std():.2f}, min={depths.min():.2f}, max={depths.max():.2f}, P5={np.percentile(depths, 5):.2f}, P95={np.percentile(depths, 95):.2f} mm")
    print(f"  Height (Z): mean={heights.mean():.2f}, std={heights.std():.2f}, min={heights.min():.2f}, max={heights.max():.2f}, P5={np.percentile(heights, 5):.2f}, P95={np.percentile(heights, 95):.2f} mm")

    # Correlations between body dimensions and internal anatomy
    # corr(H, p_{k, z}) for L1 vertebrae
    l1_idx = ORGAN_NAMES.index("vertebrae_L1")
    l1_val = masks[:, l1_idx] > 0.5
    h_l1 = heights[l1_val]
    z_l1 = raw_centroids[l1_val, l1_idx, 2]
    p_corr_h, _ = stats.pearsonr(h_l1, z_l1)
    s_corr_h, _ = stats.spearmanr(h_l1, z_l1)
    print(f"\nCorrelation between Height (H) and L1 Vertebra Z-coord:")
    print(f"  Pearson: {p_corr_h:.4f}, Spearman: {s_corr_h:.4f}")

    # corr(W, |kidney_left - kidney_right|)
    kl_idx = ORGAN_NAMES.index("kidney_left")
    kr_idx = ORGAN_NAMES.index("kidney_right")
    k_val = (masks[:, kl_idx] > 0.5) & (masks[:, kr_idx] > 0.5)
    w_k = widths[k_val]
    # In swapped coordinates, Left-Right separation is along Z:
    k_sep_z = np.abs(raw_centroids[k_val, kl_idx, 2] - raw_centroids[k_val, kr_idx, 2])
    k_sep_x = np.abs(raw_centroids[k_val, kl_idx, 0] - raw_centroids[k_val, kr_idx, 0])
    p_corr_w, _ = stats.pearsonr(w_k, k_sep_x)
    s_corr_w, _ = stats.spearmanr(w_k, k_sep_x)
    print(f"Correlation between Width (W) and Kidney Left-Right separation in X:")
    print(f"  Pearson: {p_corr_w:.4f}, Spearman: {s_corr_w:.4f}")

    # corr(D, liver Y-coord)
    liv_idx = ORGAN_NAMES.index("liver")
    liv_val = masks[:, liv_idx] > 0.5
    d_liv = depths[liv_val]
    y_liv = raw_centroids[liv_val, liv_idx, 1]
    p_corr_d, _ = stats.pearsonr(d_liv, y_liv)
    s_corr_d, _ = stats.spearmanr(d_liv, y_liv)
    print(f"Correlation between Depth (D) and Liver Y-coord:")
    print(f"  Pearson: {p_corr_d:.4f}, Spearman: {s_corr_d:.4f}")

if __name__ == "__main__":
    compute_phase1_statistics()
