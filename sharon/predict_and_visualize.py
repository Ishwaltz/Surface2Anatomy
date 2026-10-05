import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

SHARON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARON_DIR))

from config import Config
from dataset import CTLocalizationDataset
from ensemble_tta_predictor import EnsembleTTAPredictor
from labels import ORGAN_NAMES, NUM_ORGANS


def predict_scan(case_dir: str, organs: list = None, save_fig: str = "prediction_overlay.png", use_tta: bool = True):
    """
    Given a CT volume directory (containing ct.nii.gz), runs model inference,
    extracts physical 3D (X, Y, Z) coordinates in millimeters, and plots visual overlays.
    """
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
    sample = None
    for i, c in enumerate(ds.cases):
        if c.name == case_name:
            sample = ds[i]
            break

    if sample is None:
        raise FileNotFoundError(f"Case {case_name} not found in {Path(case_dir).parent}")

    img = sample["image"].unsqueeze(0).to(device)
    spacing = sample["spacing"].numpy() # (dx, dy, dz)
    gt_centroids = sample["gt_centroids"].numpy()
    found_mask = sample["found_mask"].numpy()

    with torch.inference_mode():
        pred_centroids_norm, heatmaps = predictor.predict(img, use_tta=use_tta)

    pred_norm = pred_centroids_norm[0].cpu().numpy() # (117, 3) (z, y, x)
    hm = heatmaps[0].cpu().numpy()                  # (117, 64, 64, 64)

    # Convert normalized (z, y, x) to physical (x, y, z) mm
    pred_xyz_mm = {}
    for i, name in enumerate(ORGAN_NAMES):
        pz, py, px = pred_norm[i]
        x_mm = float(px * 128.0 * spacing[0])
        y_mm = float(py * 128.0 * spacing[1])
        z_mm = float(pz * 128.0 * spacing[2])
        pred_xyz_mm[name] = {"x_mm": round(x_mm, 2), "y_mm": round(y_mm, 2), "z_mm": round(z_mm, 2)}

    # If organs requested for plotting, plot them
    if organs is None:
        organs = ["spleen", "liver", "kidney_right", "urinary_bladder"]

    fig, axes = plt.subplots(1, len(organs), figsize=(5 * len(organs), 5))
    if len(organs) == 1:
        axes = [axes]

    ct_vol = img[0, 0].cpu().numpy()

    for ax_idx, org_name in enumerate(organs):
        if org_name not in ORGAN_NAMES:
            continue
        idx = ORGAN_NAMES.index(org_name)
        z_slice = int(np.clip(round(pred_norm[idx, 0] * 128), 0, 127))
        hm_z = int(np.clip(round(pred_norm[idx, 0] * 64), 0, 63))

        ct_slice = ct_vol[z_slice, :, :]
        hm_slice = hm[idx, hm_z, :, :]

        px = pred_norm[idx, 2] * 128
        py = pred_norm[idx, 1] * 128

        ax = axes[ax_idx]
        ax.imshow(ct_slice, cmap="gray", origin="lower")
        ax.imshow(hm_slice, cmap="hot", alpha=0.45, extent=[0, 128, 0, 128], origin="lower")
        ax.plot(px, py, 'g+', markersize=14, markeredgewidth=2.5, label=f"Pred ({pred_xyz_mm[org_name]['x_mm']:.0f}, {pred_xyz_mm[org_name]['y_mm']:.0f}, {pred_xyz_mm[org_name]['z_mm']:.0f}) mm")

        if found_mask[idx]:
            gx = gt_centroids[idx, 2] * 128
            gy = gt_centroids[idx, 1] * 128
            ax.plot(gx, gy, 'bx', markersize=10, markeredgewidth=2, label="GT")

        ax.set_title(f"{org_name.replace('_',' ').title()} (Axial Slice Z={z_slice})", fontsize=12, fontweight='bold')
        ax.axis("off")
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Visual overlay saved to: {save_fig}")
    return pred_xyz_mm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", type=str, default=str(SHARON_DIR / "dataset" / "case_003"))
    parser.add_argument("--save_fig", type=str, default=str(SHARON_DIR / "outputs" / "sample_prediction.png"))
    parser.add_argument("--organs", nargs="+", default=["spleen", "liver", "kidney_right", "urinary_bladder"])
    args = parser.parse_args()

    results = predict_scan(args.case_dir, organs=args.organs, save_fig=args.save_fig)
    print("\n--- Predicted 3D Coordinates (Sample) ---")
    for org in args.organs:
        print(f"  {org:20s}: {results[org]}")
