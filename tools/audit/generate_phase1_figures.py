import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES, SKELETAL_INDICES, SOFT_TISSUE_INDICES

def generate_all_figures():
    print("=" * 80)
    print("GENERATING PHASE 1 VISUAL DIAGNOSTICS (FIGURES 1-8)")
    print("=" * 80)

    fig_dir = repo_root / "reports" / "phase1" / "figures"
    geo_dir = fig_dir / "geometry"
    fig_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

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

    # Compute bounding extents
    p_min = raw_points.min(axis=1) # (450, 3)
    p_max = raw_points.max(axis=1) # (450, 3)
    extents = p_max - p_min
    widths, depths, heights = extents[:, 0], extents[:, 1], extents[:, 2]

    # FIGURE 1: Bounding-box width/depth/height distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(widths, bins=30, color="teal", edgecolor="black", alpha=0.7)
    axes[0].set_title("Surface Width (X) Distribution")
    axes[0].set_xlabel("Width (mm)")
    axes[0].set_ylabel("Count")

    axes[1].hist(depths, bins=30, color="coral", edgecolor="black", alpha=0.7)
    axes[1].set_title("Surface Depth (Y) Distribution")
    axes[1].set_xlabel("Depth (mm)")

    axes[2].hist(heights, bins=30, color="purple", edgecolor="black", alpha=0.7)
    axes[2].set_title("Surface Height (Z) Distribution (Swapped: up to 3.5m!)")
    axes[2].set_xlabel("Height (mm)")

    plt.tight_layout()
    f1_path = fig_dir / "fig1_bbox_distributions.png"
    plt.savefig(f1_path, dpi=150)
    plt.close()
    print(f"✓ Figure 1 saved: {f1_path}")

    # FIGURE 2: Scatter: body height vs target coordinate
    l1_idx = ORGAN_NAMES.index("vertebrae_L1")
    l1_val = masks[:, l1_idx] > 0.5
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(heights[l1_val], raw_centroids[l1_val, l1_idx, 2], alpha=0.6, color="navy")
    ax.set_title("Body Height (Z) vs L1 Vertebra Coordinate")
    ax.set_xlabel("Body Bounding Box Height (mm)")
    ax.set_ylabel("L1 Vertebra Coordinate (mm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    f2_path = fig_dir / "fig2_height_vs_target_coord.png"
    plt.savefig(f2_path, dpi=150)
    plt.close()
    print(f"✓ Figure 2 saved: {f2_path}")

    # FIGURE 3: Body width vs Left-Right separation
    kl_idx = ORGAN_NAMES.index("kidney_left")
    kr_idx = ORGAN_NAMES.index("kidney_right")
    k_val = (masks[:, kl_idx] > 0.5) & (masks[:, kr_idx] > 0.5)
    sep_x = np.abs(raw_centroids[k_val, kl_idx, 0] - raw_centroids[k_val, kr_idx, 0])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(widths[k_val], sep_x, alpha=0.6, color="crimson")
    ax.set_title("Body Width vs Bilateral Kidney Separation (X)")
    ax.set_xlabel("Body Width (mm)")
    ax.set_ylabel("Kidney Left-Right Separation (mm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    f3_path = fig_dir / "fig3_width_vs_kidney_separation.png"
    plt.savefig(f3_path, dpi=150)
    plt.close()
    print(f"✓ Figure 3 saved: {f3_path}")

    # FIGURE 4: Body depth vs Anterior-Posterior organ coordinate
    liv_idx = ORGAN_NAMES.index("liver")
    liv_val = masks[:, liv_idx] > 0.5
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(depths[liv_val], raw_centroids[liv_val, liv_idx, 1], alpha=0.6, color="forestgreen")
    ax.set_title("Body Depth (Y) vs Liver Y-Coordinate")
    ax.set_xlabel("Body Depth (mm)")
    ax.set_ylabel("Liver AP Coordinate (mm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    f4_path = fig_dir / "fig4_depth_vs_liver_ap.png"
    plt.savefig(f4_path, dpi=150)
    plt.close()
    print(f"✓ Figure 4 saved: {f4_path}")

    # FIGURE 6: Per-target coordinate standard deviation
    stds = []
    names = []
    for j, name in enumerate(ORGAN_NAMES):
        val = masks[:, j] > 0.5
        if val.sum() > 5:
            s_vec = raw_centroids[val, j].std(axis=0)
            stds.append(np.linalg.norm(s_vec))
            names.append(name)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(stds)), stds, color="steelblue", edgecolor="black", width=0.8)
    ax.set_title("Per-Target 3D Coordinate Standard Deviation Across Patients (mm)")
    ax.set_xlabel("Organ Index")
    ax.set_ylabel("Total Std Dev (mm)")
    plt.tight_layout()
    f6_path = fig_dir / "fig6_target_coord_std.png"
    plt.savefig(f6_path, dpi=150)
    plt.close()
    print(f"✓ Figure 6 saved: {f6_path}")

    # FIGURE 7: Missing-target percentage by structure
    miss_pct = (1.0 - masks.mean(axis=0)) * 100.0
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(121), miss_pct, color="darkorange", edgecolor="black", width=0.8)
    ax.set_title("Missing-Target Percentage Across All 450 Cases (%)")
    ax.set_xlabel("Organ Index (0..120)")
    ax.set_ylabel("Percentage Missing (%)")
    plt.tight_layout()
    f7_path = fig_dir / "fig7_missing_target_percentage.png"
    plt.savefig(f7_path, dpi=150)
    plt.close()
    print(f"✓ Figure 7 saved: {f7_path}")

    # FIGURE 8: Round-trip transformation error distribution
    # Subsample 20 cases for fast exact round-trip histogram
    all_rt_errs = []
    for b in range(45):
        m = masks[b] > 0.5
        rec = data["centroids"][b] * data["scales"][b] + data["centers"][b]
        errs = torch.norm((data["raw_centroids"][b] - rec)[m], dim=-1).numpy()
        all_rt_errs.extend(errs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(all_rt_errs, bins=30, color="orchid", edgecolor="black", alpha=0.7)
    ax.set_title("Normalization Round-Trip Numerical Error Distribution")
    ax.set_xlabel("Error (mm)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    f8_path = fig_dir / "fig8_roundtrip_error_distribution.png"
    plt.savefig(f8_path, dpi=150)
    plt.close()
    print(f"✓ Figure 8 saved: {f8_path}")

    # FIGURE 5: 3D Surface + Target Overlays for Sample Patients (10 Train, 5 Val, 5 Test)
    sample_train = sp["train_cases"][:10]
    sample_val = sp["val_cases"][:5]
    sample_test = sp["test_cases"][:5]
    all_vis_cases = [(c, "train") for c in sample_train] + [(c, "val") for c in sample_val] + [(c, "test") for c in sample_test]

    print(f"\nGenerating 3D patient overlays for {len(all_vis_cases)} cases in {geo_dir}...")
    for cid, split_name in all_vis_cases:
        b = case_ids.index(cid)
        pts = raw_points[b]
        # Subsample points for clear visualization
        sub_pts = pts[np.random.choice(len(pts), 800, replace=False)]
        
        m = masks[b] > 0.5
        v_indices = np.where(m)[0]
        
        skel_v = [idx for idx in v_indices if idx in SKELETAL_INDICES]
        soft_v = [idx for idx in v_indices if idx in SOFT_TISSUE_INDICES]

        skel_coords = raw_centroids[b, skel_v]
        soft_coords = raw_centroids[b, soft_v]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot skin surface points
        ax.scatter(sub_pts[:, 0], sub_pts[:, 1], sub_pts[:, 2], s=1, color="lightgray", alpha=0.3, label="Skin Surface")

        # Plot skeletal landmarks
        if len(skel_coords) > 0:
            ax.scatter(skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2], s=25, color="red", alpha=0.9, label=f"Skeletal ({len(skel_coords)})")

        # Plot soft-tissue landmarks
        if len(soft_coords) > 0:
            ax.scatter(soft_coords[:, 0], soft_coords[:, 1], soft_coords[:, 2], s=25, color="dodgerblue", alpha=0.9, label=f"Soft-Tissue ({len(soft_coords)})")

        ax.set_title(f"Patient {cid} ({split_name.upper()} split)\nAnatomical Overlay (Swapped Space: Z ~ {heights[b]:.0f} mm)")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.legend(loc="upper right")
        plt.tight_layout()

        out_plot = geo_dir / f"overlay_{split_name}_{cid}.png"
        plt.savefig(out_plot, dpi=120)
        plt.close()

    print(f"✓ All 20 3D patient overlays generated in: {geo_dir}")

if __name__ == "__main__":
    generate_all_figures()
