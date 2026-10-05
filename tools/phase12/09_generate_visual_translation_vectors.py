#!/usr/bin/env python3
"""
Section 17: Visual 3D Error Vector Panels for 30 AMOS Patients
=============================================================
Renders 30 multi-view panels showing:
- External body surface points
- Ground truth organ centroids
- Raw Phase-10R predicted centroids
- Translation-oracle / Corrected predicted centroids
- Color-coded 3D error displacement vectors
Saved to reports/phase12/figures/translation_vectors/
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9
})

def main():
    print("=" * 80)
    print("PHASE 12 — SECTION 17: VISUAL TRANSLATION VECTOR PANELS (30 CASES)")
    print("=" * 80)

    out_dir = repo_root / "reports" / "phase12" / "figures" / "translation_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load AMOS Predictions & Ground Truth
    preds_file = repo_root / "reports" / "phase11" / "predictions" / "AMOS_ensemble.npz"
    gt_csv = repo_root / "data_external" / "AMOS22" / "processed" / "AMOS_GT_centroids.csv"
    mapping_csv = repo_root / "reports" / "phase11" / "mappings" / "AMOS_target_mapping.csv"
    pt_dir = repo_root / "data_external" / "AMOS22" / "processed" / "pointclouds"

    data_preds = np.load(preds_file)
    case_ids = [str(c) for c in data_preds["case_ids"]]
    preds = data_preds["predictions"] # (N, 117, 3)

    df_gt = pd.read_csv(gt_csv)
    df_map = pd.read_csv(mapping_csv)
    target_slots = df_map["Phase10R_target_index"].tolist()
    target_names = df_map["Phase10R_target_name"].tolist()

    # Load patient signed vectors to pick representative diverse cases
    vec_csv = repo_root / "reports" / "phase12" / "01_axis_bias" / "patient_signed_vectors.csv"
    df_vec = pd.read_csv(vec_csv)
    
    # Pick 20 CT cases and 10 MRI cases
    ct_cases = df_vec[df_vec["modality"] == "CT"]["case_id"].tolist()[:20]
    mri_cases = df_vec[df_vec["modality"] == "MRI"]["case_id"].tolist()[:10]
    selected_cases = ct_cases + mri_cases
    print(f"Selected {len(selected_cases)} cases ({len(ct_cases)} CT, {len(mri_cases)} MRI) for 3D visualization.")

    cohort_shift = np.array([-0.02, -46.22, +15.40])

    for rank, cid in enumerate(selected_cases):
        if cid not in case_ids:
            continue
        c_idx = case_ids.index(cid)
        npz_file = pt_dir / f"{cid}.npz"
        if not npz_file.exists():
            continue
        
        pt_data = np.load(npz_file)
        pts_surf = pt_data["points_unnormalized"] # (4096, 3)
        
        # Subsample 1000 surface points for clean rendering
        pts_sub = pts_surf[::4]

        # Extract GT, Raw Pred, and Oracle Pred
        gt_sub = df_gt[df_gt["case_id"] == cid]
        gt_pts, raw_pts, oracle_pts, names = [], [], [], []
        for slot, tname in zip(target_slots, target_names):
            r = gt_sub[gt_sub["target_name"] == tname]
            if len(r) > 0:
                c_true = np.array([r["x_centered_mm"].values[0], r["y_centered_mm"].values[0], r["z_centered_mm"].values[0]])
                c_raw = preds[c_idx, slot]
                c_oracle = c_raw + cohort_shift
                gt_pts.append(c_true)
                raw_pts.append(c_raw)
                oracle_pts.append(c_oracle)
                names.append(tname)

        if len(gt_pts) == 0:
            continue
        gt_pts = np.array(gt_pts)
        raw_pts = np.array(raw_pts)
        oracle_pts = np.array(oracle_pts)

        # Compute errors
        raw_errs = np.linalg.norm(raw_pts - gt_pts, axis=1)
        oracle_errs = np.linalg.norm(oracle_pts - gt_pts, axis=1)

        # Plot 2-view 3D panel (Frontal Coronal X-Z and Lateral Sagittal Y-Z)
        fig = plt.figure(figsize=(12, 6))
        
        # View 1: 3D Isometric View
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2], color="gray", alpha=0.08, s=2, label="Body Surface")
        ax1.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], color="green", s=50, marker="o", edgecolors="black", label="Ground Truth Centroid")
        ax1.scatter(raw_pts[:, 0], raw_pts[:, 1], raw_pts[:, 2], color="red", s=50, marker="^", edgecolors="black", label=f"Raw Phase-10R (MRE: {np.mean(raw_errs):.1f} mm)")
        ax1.scatter(oracle_pts[:, 0], oracle_pts[:, 1], oracle_pts[:, 2], color="blue", s=50, marker="s", edgecolors="black", label=f"Oracle Shift (MRE: {np.mean(oracle_errs):.1f} mm)")

        # Draw error vectors
        for k in range(len(gt_pts)):
            # Raw error vector (Red)
            ax1.plot([raw_pts[k, 0], gt_pts[k, 0]], [raw_pts[k, 1], gt_pts[k, 1]], [raw_pts[k, 2], gt_pts[k, 2]], color="red", lw=1.5, alpha=0.7)
            # Oracle error vector (Blue dotted)
            ax1.plot([oracle_pts[k, 0], gt_pts[k, 0]], [oracle_pts[k, 1], gt_pts[k, 1]], [oracle_pts[k, 2], gt_pts[k, 2]], color="blue", lw=1.0, linestyle="--", alpha=0.7)

        ax1.set_xlabel("X (Right) mm")
        ax1.set_ylabel("Y (Anterior) mm")
        ax1.set_zlabel("Z (Superior) mm")
        ax1.set_title(f"A: 3D Error Vectors — {cid} ({df_vec.loc[df_vec['case_id']==cid, 'modality'].values[0]})", fontweight="bold")
        ax1.view_init(elev=20, azim=-60)
        ax1.legend(loc="upper left", fontsize=8)

        # View 2: Sagittal Plane (Y-Z) showing dominant Anterior-Superior Shift
        ax2 = fig.add_subplot(122)
        ax2.scatter(pts_sub[:, 1], pts_sub[:, 2], color="lightgray", alpha=0.3, s=5, label="Surface (Sagittal)")
        ax2.scatter(gt_pts[:, 1], gt_pts[:, 2], color="green", s=60, marker="o", edgecolors="black", zorder=5, label="Ground Truth")
        ax2.scatter(raw_pts[:, 1], raw_pts[:, 2], color="red", s=60, marker="^", edgecolors="black", zorder=5, label="Raw Predicted")
        ax2.scatter(oracle_pts[:, 1], oracle_pts[:, 2], color="blue", s=60, marker="s", edgecolors="black", zorder=5, label="Oracle Predicted")

        # Draw 2D projection arrows
        for k in range(len(gt_pts)):
            ax2.annotate("", xy=(gt_pts[k, 1], gt_pts[k, 2]), xytext=(raw_pts[k, 1], raw_pts[k, 2]),
                         arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.8))
            ax2.annotate("", xy=(gt_pts[k, 1], gt_pts[k, 2]), xytext=(oracle_pts[k, 1], oracle_pts[k, 2]),
                         arrowprops=dict(arrowstyle="->", color="blue", lw=1.0, linestyle="--", alpha=0.8))

        ax2.set_xlabel("Y (Anterior-Posterior) mm", fontweight="bold")
        ax2.set_ylabel("Z (Superior-Inferior) mm", fontweight="bold")
        ax2.set_title(f"B: Sagittal Projection (Dominant +Y Shift = +{df_vec.loc[df_vec['case_id']==cid, 'mean_dy_mm'].values[0]:.1f} mm)", fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="lower left", fontsize=8)

        plt.tight_layout()
        save_path = out_dir / f"panel_{rank+1:02d}_{cid}.png"
        fig.savefig(save_path, dpi=180)
        plt.close(fig)

    print(f"Successfully generated {len(selected_cases)} 3D visual panels in {out_dir}.")

if __name__ == "__main__":
    main()
