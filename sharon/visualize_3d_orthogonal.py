import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import scipy.ndimage
import matplotlib.pyplot as plt

SHARON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARON_DIR))

from config import Config
from dataset import CTLocalizationDataset
from ensemble_tta_predictor import EnsembleTTAPredictor
from labels import ORGAN_NAMES, NUM_ORGANS


def render_orthogonal_card(
    ct_vol: np.ndarray,
    hm_vol: np.ndarray,
    pred_c_norm: np.ndarray,
    gt_c_norm: np.ndarray,
    organ_found: bool,
    spacing_mm: np.ndarray,
    organ_name: str,
    save_path: str,
):
    D, H, W = ct_vol.shape
    pz = int(np.clip(round(pred_c_norm[0] * D), 0, D - 1))
    py = int(np.clip(round(pred_c_norm[1] * H), 0, H - 1))
    px = int(np.clip(round(pred_c_norm[2] * W), 0, W - 1))

    pred_xyz_mm = (
        pred_c_norm[2] * W * spacing_mm[0],
        pred_c_norm[1] * H * spacing_mm[1],
        pred_c_norm[0] * D * spacing_mm[2],
    )

    if organ_found:
        gt_xyz_mm = (
            gt_c_norm[2] * W * spacing_mm[0],
            gt_c_norm[1] * H * spacing_mm[1],
            gt_c_norm[0] * D * spacing_mm[2],
        )
        dist_err_mm = float(np.linalg.norm(np.array(pred_xyz_mm) - np.array(gt_xyz_mm)))
    else:
        dist_err_mm = None

    hm_resampled = scipy.ndimage.zoom(hm_vol, (D / hm_vol.shape[0], H / hm_vol.shape[1], W / hm_vol.shape[2]), order=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#1a1a1a")

    # 1. Axial
    ax_axial = axes[0]
    ax_axial.imshow(ct_vol[pz, :, :], cmap="gray", origin="lower")
    ax_axial.imshow(hm_resampled[pz, :, :], cmap="hot", alpha=0.45, origin="lower")
    ax_axial.plot(px, py, 'g+', markersize=16, markeredgewidth=2.5, label="Model Prediction")
    if organ_found:
        ax_axial.plot(gt_c_norm[2] * W, gt_c_norm[1] * H, 'rx', markersize=12, markeredgewidth=2, label="Expert Annotation")
    ax_axial.set_title(f"Axial View (Z = {pz})", color="white", fontsize=13, fontweight="bold")
    ax_axial.axis("off")
    ax_axial.legend(loc="upper right", facecolor="#2a2a2a", edgecolor="none", labelcolor="white", fontsize=10)

    # 2. Coronal
    ax_coronal = axes[1]
    ax_coronal.imshow(ct_vol[:, py, :], cmap="gray", origin="lower", aspect=spacing_mm[2]/spacing_mm[0])
    ax_coronal.imshow(hm_resampled[:, py, :], cmap="hot", alpha=0.45, origin="lower", aspect=spacing_mm[2]/spacing_mm[0])
    ax_coronal.plot(px, pz, 'g+', markersize=16, markeredgewidth=2.5)
    if organ_found:
        ax_coronal.plot(gt_c_norm[2] * W, gt_c_norm[0] * D, 'rx', markersize=12, markeredgewidth=2)
    ax_coronal.set_title(f"Coronal View (Y = {py})", color="white", fontsize=13, fontweight="bold")
    ax_coronal.axis("off")

    # 3. Sagittal
    ax_sagittal = axes[2]
    ax_sagittal.imshow(ct_vol[:, :, px], cmap="gray", origin="lower", aspect=spacing_mm[2]/spacing_mm[1])
    ax_sagittal.imshow(hm_resampled[:, :, px], cmap="hot", alpha=0.45, origin="lower", aspect=spacing_mm[2]/spacing_mm[1])
    ax_sagittal.plot(py, pz, 'g+', markersize=16, markeredgewidth=2.5)
    if organ_found:
        ax_sagittal.plot(gt_c_norm[1] * H, gt_c_norm[0] * D, 'rx', markersize=12, markeredgewidth=2)
    ax_sagittal.set_title(f"Sagittal View (X = {px})", color="white", fontsize=13, fontweight="bold")
    ax_sagittal.axis("off")

    err_text = f" | Physical Error: {dist_err_mm:.2f} mm" if dist_err_mm is not None else ""
    fig.suptitle(
        f"3D Continuous Anatomical Localization: {organ_name.replace('_', ' ').upper()}{err_text}\n"
        f"Predicted Physical Coordinates: (X={pred_xyz_mm[0]:.1f}, Y={pred_xyz_mm[1]:.1f}, Z={pred_xyz_mm[2]:.1f}) mm",
        color="white",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SUCCESS] Rendered 3-plane diagnostic card to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", type=str, default=str(SHARON_DIR / "dataset" / "case_003"))
    parser.add_argument("--organ", type=str, default="liver")
    parser.add_argument("--out", type=str, default=str(SHARON_DIR / "outputs" / "liver_3plane_audit.png"))
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CTLocalizationDataset(str(Path(args.case_dir).parent), cfg.target_shape, cfg.heatmap_size, augment=False)
    case_name = Path(args.case_dir).name
    idx = [i for i, c in enumerate(ds.cases) if c.name == case_name][0]
    sample = ds[idx]

    img = sample["image"].unsqueeze(0).to(device)
    spacing = sample["spacing"].numpy()
    gt_c = sample["gt_centroids"].numpy()
    found = sample["found_mask"].numpy()

    ckpt_paths = {
        "resnet50": str(SHARON_DIR / "outputs" / "checkpoints" / "resnet50" / "best_model.pth"),
        "densenet121": str(SHARON_DIR / "outputs" / "checkpoints" / "densenet121" / "best_model.pth"),
        "swin_unetr": str(SHARON_DIR / "outputs" / "checkpoints" / "swin_unetr" / "best_model.pth"),
    }
    predictor = EnsembleTTAPredictor(ckpt_paths, device=device)

    with torch.inference_mode():
        pred_c_norm, heatmaps = predictor.predict(img, use_tta=True)

    org_idx = ORGAN_NAMES.index(args.organ)
    render_orthogonal_card(
        ct_vol=img[0, 0].cpu().numpy(),
        hm_vol=heatmaps[0, org_idx].cpu().numpy(),
        pred_c_norm=pred_c_norm[0, org_idx].cpu().numpy(),
        gt_c_norm=gt_c[org_idx],
        organ_found=found[org_idx],
        spacing_mm=spacing,
        organ_name=args.organ,
        save_path=args.out,
    )


if __name__ == "__main__":
    main()
