import sys
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import nibabel as nib

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES, SKELETAL_INDICES, SOFT_TISSUE_INDICES
from tools.reconstruct.surface_extractor import extract_patient_body_surface

def compute_chamfer(p1, p2):
    # p1, p2: (N, 3)
    p1_t = torch.from_numpy(p1).float()
    p2_t = torch.from_numpy(p2).float()
    dist_mat = torch.cdist(p1_t, p2_t)
    d1 = dist_mat.min(dim=1).values.mean().item()
    d2 = dist_mat.min(dim=0).values.mean().item()
    return (d1 + d2) / 2.0

def main():
    print("=" * 80)
    print("STAGE 24-30: V2 ANALYTICS, VARIABILITY, TARGET SUPPORT & VISUAL QC")
    print("=" * 80)

    fig_dir = repo_root / "reports" / "phase1r" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir = repo_root / "reports" / "phase1r"

    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    data = torch.load(str(pt_path), weights_only=False)

    splits_file = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
    with open(splits_file) as f:
        sp = json.load(f)

    case_ids = data["case_ids"]
    pts_world = data["points_world_mm"].numpy() # (440, 4096, 3)
    tgt_world = data["targets_world_mm"].numpy() # (440, 121, 3)
    tgt_mask = data["target_mask"].numpy()       # (440, 121)
    prim_mask = data["target_primary_mask"].numpy() # (440, 121)
    body_dims = data["body_dimensions_mm"].numpy() # (440, 3)

    tr_idx = sp["train_indices"]
    val_idx = sp["val_indices"]
    te_idx = sp["test_indices"]

    N = len(case_ids)

    # 1. Physical Torso Dimension Distributions (Stage 24)
    widths = body_dims[:, 0]
    depths = body_dims[:, 1]
    heights = body_dims[:, 2]

    print("\n--- Physical Body Dimensions V2 (Training Split N=352) ---")
    tr_w, tr_d, tr_h = widths[tr_idx], depths[tr_idx], heights[tr_idx]
    print(f"  Width (X):  mean={tr_w.mean():.2f} ± {tr_w.std():.2f}, median={np.median(tr_w):.2f}, range=[{tr_w.min():.2f}, {tr_w.max():.2f}] mm")
    print(f"  Depth (Y):  mean={tr_d.mean():.2f} ± {tr_d.std():.2f}, median={np.median(tr_d):.2f}, range=[{tr_d.min():.2f}, {tr_d.max():.2f}] mm")
    print(f"  Height (Z): mean={tr_h.mean():.2f} ± {tr_h.std():.2f}, median={np.median(tr_h):.2f}, range=[{tr_h.min():.2f}, {tr_h.max():.2f}] mm")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(tr_w, bins=25, color="teal", edgecolor="black", alpha=0.7)
    axes[0].set_title("V2 Corrected Patient Width (X) [mm]")
    axes[0].set_xlabel("Width (mm)")
    axes[0].set_ylabel("Count")

    axes[1].hist(tr_d, bins=25, color="coral", edgecolor="black", alpha=0.7)
    axes[1].set_title("V2 Corrected Patient Depth (Y) [mm]")
    axes[1].set_xlabel("Depth (mm)")

    axes[2].hist(tr_h, bins=25, color="purple", edgecolor="black", alpha=0.7)
    axes[2].set_title("V2 Corrected Patient Height (Z) [mm]")
    axes[2].set_xlabel("Height (mm)")

    plt.tight_layout()
    plt.savefig(fig_dir / "v2_physical_dimension_distributions.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {fig_dir / 'v2_physical_dimension_distributions.png'}")

    # 2. Real-Surface Variability Test (Stage 26)
    print("\n--- Real Surface Variability Test (Stage 26) ---")
    # Between-patient Chamfer distance across 30 random patients (in physical mm)
    rng = np.random.RandomState(42)
    sample_sub = rng.choice(N, 30, replace=False)
    
    d_between = []
    for a in range(len(sample_sub)):
        for b in range(a + 1, len(sample_sub)):
            cd = compute_chamfer(pts_world[sample_sub[a]], pts_world[sample_sub[b]])
            d_between.append(cd)
    d_between = np.array(d_between)

    # Within-patient resampling distance (resample case_000, case_001, case_002 with different seeds)
    d_within = []
    for test_cid in ["case_000", "case_001", "case_002"]:
        ct_p = repo_root / "sharon" / "dataset" / test_cid / "ct.nii.gz"
        r1 = extract_patient_body_surface(ct_p, num_points=4096, seed=100)
        r2 = extract_patient_body_surface(ct_p, num_points=4096, seed=200)
        cd_w = compute_chamfer(r1["points_world_mm"], r2["points_world_mm"])
        d_within.append(cd_w)
    d_within = np.array(d_within)

    print(f"  Between-Patient Distance (mean ± std): {d_between.mean():.2f} ± {d_between.std():.2f} mm")
    print(f"  Within-Patient Resampling Distance:     {d_within.mean():.2f} ± {d_within.std():.2f} mm")
    print(f"  Ratio (D_between / D_within):          {d_between.mean() / d_within.mean():.2f}x")

    # 3. Target Support Matrix & Scan Coverage (Stage 29 & 30)
    print("\n--- Target Support Matrix (Stage 30) ---")
    support_rows = []
    
    sufficient_count = 0
    low_count = 0
    very_low_count = 0
    synthetic_count = 0
    absent_count = 0

    for idx, name in enumerate(ORGAN_NAMES):
        organ_id = idx + 1
        tr_v = int(tgt_mask[tr_idx, idx].sum())
        val_v = int(tgt_mask[val_idx, idx].sum())
        te_v = int(tgt_mask[te_idx, idx].sum())
        tot_v = int(tgt_mask[:, idx].sum())
        pct_v = (tot_v / N) * 100.0

        if organ_id in [118, 119, 120, 121]:
            cat = "SYNTHETIC_ONLY"
            synthetic_count += 1
            src = "SYNTHETIC"
        elif tot_v == 0:
            cat = "ABSENT"
            absent_count += 1
            src = "REAL_SEGMENTATION"
        elif tot_v < 15:
            cat = "VERY_LOW_SUPPORT"
            very_low_count += 1
            src = "REAL_SEGMENTATION"
        elif tot_v < 100:
            cat = "LOW_SUPPORT"
            low_count += 1
            src = "REAL_SEGMENTATION"
        else:
            cat = "SUFFICIENT_SUPPORT"
            sufficient_count += 1
            src = "REAL_SEGMENTATION"

        support_rows.append({
            "target_index": idx,
            "target_name": name,
            "category": cat,
            "source_type": src,
            "train_valid_count": tr_v,
            "val_valid_count": val_v,
            "test_valid_count": te_v,
            "total_valid_count": tot_v,
            "percentage_valid": f"{pct_v:.1f}%"
        })

    support_csv = report_dir / "target_support_matrix.csv"
    with open(support_csv, "w", newline="") as f:
        fieldnames = list(support_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(support_rows)
    print(f"✓ Saved target support matrix to: {support_csv}")

    print(f"  Total target schema:                   121")
    print(f"  SUFFICIENT_SUPPORT (>=100 cases):      {sufficient_count}")
    print(f"  LOW_SUPPORT (15..99 cases):            {low_count}")
    print(f"  VERY_LOW_SUPPORT (1..14 cases):        {very_low_count}")
    print(f"  SYNTHETIC_ONLY (uterus,ovary,vagina):  {synthetic_count}")
    print(f"  ABSENT (0 cases):                      {absent_count}")

    # 4. Visual QC Overlays (20 Train, 10 Val, 10 Test) (Stage 28)
    print("\n--- Visual QC Overlays (Stage 28) ---")
    qc_cases = (
        [(c, "train") for c in sp["train_cases"][:20]] +
        [(c, "val") for c in sp["val_cases"][:10]] +
        [(c, "test") for c in sp["test_cases"][:10]]
    )

    for cid, s_name in qc_cases:
        b = case_ids.index(cid)
        p_sub = pts_world[b][np.random.choice(4096, 1000, replace=False)]
        
        m = tgt_mask[b] > 0.5
        v_idx = np.where(m)[0]
        skel_v = [i for i in v_idx if i in SKELETAL_INDICES]
        soft_v = [i for i in v_idx if i in SOFT_TISSUE_INDICES]

        skel_pts = tgt_world[b, skel_v]
        soft_pts = tgt_world[b, soft_v]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        
        # Real patient skin points
        ax.scatter(p_sub[:, 0], p_sub[:, 1], p_sub[:, 2], s=1, color="lightgray", alpha=0.4, label="CT Skin Surface")
        if len(skel_pts) > 0:
            ax.scatter(skel_pts[:, 0], skel_pts[:, 1], skel_pts[:, 2], s=20, color="red", label=f"Skeletal ({len(skel_pts)})")
        if len(soft_pts) > 0:
            ax.scatter(soft_pts[:, 0], soft_pts[:, 1], soft_pts[:, 2], s=20, color="dodgerblue", label=f"Soft-Tissue ({len(soft_pts)})")

        ax.set_title(f"Patient {cid} ({s_name.upper()})\nReal CT Surface (W={widths[b]:.0f}, D={depths[b]:.0f}, H={heights[b]:.0f} mm)")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.legend(loc="upper right")
        plt.tight_layout()

        out_f = fig_dir / f"overlay_v2_{s_name}_{cid}.png"
        plt.savefig(out_f, dpi=100)
        plt.close()

    print(f"✓ Generated {len(qc_cases)} 3D visual QC overlays in: {fig_dir}")

if __name__ == "__main__":
    main()
