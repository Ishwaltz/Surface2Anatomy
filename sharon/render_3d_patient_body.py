import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SHARON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARON_DIR))

from config import Config
from dataset import CTLocalizationDataset
from ensemble_tta_predictor import EnsembleTTAPredictor
from labels import ORGAN_NAMES, NUM_ORGANS


def build_3d_patient_pointer(case_dir: str, output_html: str, output_png: str, presence_thresh: float = 0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    ckpt_paths = {
        "resnet50": str(SHARON_DIR / "outputs" / "checkpoints" / "resnet50" / "best_model.pth"),
        "densenet121": str(SHARON_DIR / "outputs" / "checkpoints" / "densenet121" / "best_model.pth"),
        "swin_unetr": str(SHARON_DIR / "outputs" / "checkpoints" / "swin_unetr" / "best_model.pth"),
    }
    predictor = EnsembleTTAPredictor(ckpt_paths, device=device)

    # Load dataset sample
    ds = CTLocalizationDataset(str(Path(case_dir).parent), cfg.target_shape, cfg.heatmap_size, augment=False)
    case_name = Path(case_dir).name
    idx = [i for i, c in enumerate(ds.cases) if c.name == case_name][0]
    sample = ds[idx]

    img = sample["image"].unsqueeze(0).to(device)
    spacing = sample["spacing"].numpy()
    gt_centroids = sample["gt_centroids"].numpy()
    found_mask = sample["found_mask"].numpy()

    with torch.inference_mode():
        pred_centroids_norm, heatmaps, confidences = predictor.predict(img, use_tta=True)

    pred_norm = pred_centroids_norm[0].cpu().numpy()
    conf_scores = confidences[0].cpu().numpy()
    ct_vol = img[0, 0].cpu().numpy()

    D, H, W = ct_vol.shape
    coords_mm = {}
    detected_count = 0

    print(f"\n[Case {case_name}] Organ Detection Analysis (Threshold = {presence_thresh:.2f}):")
    for i, name in enumerate(ORGAN_NAMES):
        pz, py, px = pred_norm[i]
        x = float(px * W * spacing[0])
        y = float(py * H * spacing[1])
        z = float(pz * D * spacing[2])
        is_detected = bool(conf_scores[i] >= presence_thresh)
        if is_detected:
            detected_count += 1
        coords_mm[name] = {
            "x": x, "y": y, "z": z,
            "confidence": float(conf_scores[i]),
            "is_detected": is_detected,
            "gt_found": bool(found_mask[i]),
        }

    print(f"  Total Anatomical Organs Detected in Field of View: {detected_count} / {NUM_ORGANS}")
    print(f"  Organs Outside Scan Field of View (Filtered Out): {NUM_ORGANS - detected_count} / {NUM_ORGANS}")

    # Extract skeleton point cloud
    bone_mask = (ct_vol > 0.55)[::2, ::2, ::2]
    bz, by, bx = np.where(bone_mask)
    bx_mm = bx * 2.0 * spacing[0]
    by_mm = by * 2.0 * spacing[1]
    bz_mm = bz * 2.0 * spacing[2]

    if len(bx_mm) > 8000:
        sample_idx = np.random.choice(len(bx_mm), 8000, replace=False)
        bx_mm, by_mm, bz_mm = bx_mm[sample_idx], by_mm[sample_idx], bz_mm[sample_idx]

    # Create Plotly 3D Figure
    fig_3d = go.Figure()

    fig_3d.add_trace(go.Scatter3d(
        x=bx_mm,
        y=by_mm,
        z=bz_mm,
        mode="markers",
        marker=dict(size=1.8, color="#556b8d", opacity=0.25),
        name="Patient Skeleton & Anatomy",
        hoverinfo="none",
    ))

    colors = {
        "visceral": "#ff3366",
        "urinary": "#33ccff",
        "endocrine": "#ffcc00",
        "respiratory": "#33ff99",
        "vertebrae": "#bb88ff",
    }

    key_organs = [
        ("spleen", "Spleen", "visceral", 12),
        ("liver", "Liver", "visceral", 14),
        ("heart", "Heart", "visceral", 12),
        ("stomach", "Stomach", "visceral", 12),
        ("kidney_left", "Left Kidney", "urinary", 10),
        ("kidney_right", "Right Kidney", "urinary", 10),
        ("urinary_bladder", "Urinary Bladder", "urinary", 11),
        ("prostate", "Prostate", "urinary", 9),
        ("pancreas", "Pancreas", "endocrine", 10),
        ("adrenal_gland_left", "Left Adrenal", "endocrine", 8),
        ("adrenal_gland_right", "Right Adrenal", "endocrine", 8),
        ("thyroid_gland", "Thyroid", "endocrine", 9),
        ("lung_upper_lobe_left", "Left Lung (Upper)", "respiratory", 10),
        ("lung_upper_lobe_right", "Right Lung (Upper)", "respiratory", 10),
        ("vertebrae_L5", "L5 Vertebra", "vertebrae", 8),
        ("vertebrae_L1", "L1 Vertebra", "vertebrae", 8),
        ("vertebrae_T12", "T12 Vertebra", "vertebrae", 8),
        ("vertebrae_T1", "T1 Vertebra", "vertebrae", 8),
    ]

    plotted_in_html = 0
    for org_key, display_name, cat, size in key_organs:
        pos = coords_mm[org_key]
        if not pos["is_detected"]:
            print(f"  [Filtered Out] {display_name}: Confidence {pos['confidence']:.3f} < {presence_thresh} (Outside Scan FOV)")
            continue

        plotted_in_html += 1
        fig_3d.add_trace(go.Scatter3d(
            x=[pos["x"]],
            y=[pos["y"]],
            z=[pos["z"]],
            mode="markers+text",
            marker=dict(size=size, color=colors[cat], opacity=0.95, symbol="circle"),
            text=[f"📍 {display_name}"],
            textposition="top center",
            textfont=dict(size=11, color="white"),
            name=f"📍 {display_name}",
            hovertemplate=f"<b>{display_name}</b><br>Confidence: {pos['confidence']:.1%}<br>X: %{{x:.1f}} mm<br>Y: %{{y:.1f}} mm<br>Z: %{{z:.1f}} mm<extra></extra>",
        ))

    fig_3d.update_layout(
        title=dict(
            text=f"<b>3D Anatomical Pointer</b> (Scan: {case_name})<br><span style='font-size:13px; color:#888;'>Detected {detected_count} Organs in FOV | Missing Organs Automatically Filtered Out</span>",
            x=0.05,
            y=0.95,
            font=dict(color="#ffffff", size=18)
        ),
        paper_bgcolor="#111216",
        plot_bgcolor="#111216",
        scene=dict(
            xaxis=dict(title="Lateral X (mm)", color="#888", gridcolor="#222", backgroundcolor="#111216"),
            yaxis=dict(title="Anterior-Posterior Y (mm)", color="#888", gridcolor="#222", backgroundcolor="#111216"),
            zaxis=dict(title="Cranial-Caudal Z (mm)", color="#888", gridcolor="#222", backgroundcolor="#111216"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        legend=dict(font=dict(color="#ccc"), bgcolor="#1a1c23"),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    fig_3d.write_html(output_html)
    print(f"[SUCCESS] Interactive 3D WebGL Anatomical Pointer saved to: {output_html}")

    # Generate Cinematic 3D Matplotlib Render PNG
    fig = plt.figure(figsize=(12, 12), facecolor="#111216")
    ax = fig.add_subplot(111, projection="3d", facecolor="#111216")

    ax.scatter(bx_mm, by_mm, bz_mm, c="#445577", s=1.0, alpha=0.15, depthshade=True)

    callout_organs = [
        ("liver", "Liver", "#ff3366", (60, 40, 20)),
        ("spleen", "Spleen", "#ff9933", (-60, -40, 20)),
        ("kidney_right", "Right Kidney", "#33ccff", (60, -30, -10)),
        ("kidney_left", "Left Kidney", "#33ccff", (-60, -30, -10)),
        ("heart", "Heart", "#ff1144", (30, 50, 40)),
        ("urinary_bladder", "Bladder", "#33ff99", (0, 60, -30)),
        ("prostate", "Prostate", "#00d4ff", (0, -60, -20)),
        ("vertebrae_L1", "L1 Spine", "#bb88ff", (-50, -50, -10)),
        ("thyroid_gland", "Thyroid", "#ffcc00", (0, 50, 30)),
    ]

    plotted_cinematic = 0
    for key, name, col, offset in callout_organs:
        p = coords_mm[key]
        if not p["is_detected"]:
            continue

        plotted_cinematic += 1
        x, y, z = p["x"], p["y"], p["z"]
        ax.scatter([x], [y], [z], c=col, s=140, edgecolors="white", linewidths=1.5, depthshade=False)
        tx, ty, tz = x + offset[0], y + offset[1], z + offset[2]
        ax.plot([x, tx], [y, ty], [z, tz], c=col, linestyle="--", linewidth=1.5)
        ax.text(tx, ty, tz, f"  📍 {name} ({p['confidence']:.0%})\n  ({x:.0f}, {y:.0f}, {z:.0f}) mm", color="white", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#1c1f26", ec=col, lw=1.2, alpha=0.9))

    ax.set_title(
        f"3D Continuous Anatomical Organ Localization & Pinpoint Navigation\nScan: {case_name} | {detected_count} Organs In FOV (Out-of-FOV Organs Filtered)",
        color="white",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("X (mm)", color="#888", labelpad=10)
    ax.set_ylabel("Y (mm)", color="#888", labelpad=10)
    ax.set_zlabel("Z (mm)", color="#888", labelpad=10)
    ax.tick_params(colors="#888")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=18, azim=-45)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SUCCESS] Cinematic 3D Patient Pointer Poster saved to: {output_png}")
    return coords_mm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", type=str, default=str(SHARON_DIR / "dataset" / "case_003"))
    parser.add_argument("--output_html", type=str, default="/home/sharon/Desktop/3D_Interactive_Organ_Pointer.html")
    parser.add_argument("--output_png", type=str, default="/home/sharon/Desktop/3D_Patient_Organ_Pointer_Cinematic.png")
    parser.add_argument("--thresh", type=float, default=0.25)
    args = parser.parse_args()

    build_3d_patient_pointer(args.case_dir, args.output_html, args.output_png, presence_thresh=args.thresh)
