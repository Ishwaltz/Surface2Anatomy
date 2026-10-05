import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

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


ORGAN_ALIASES = {
    "lung": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
    "lungs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
    "left lung": ["lung_upper_lobe_left", "lung_lower_lobe_left"],
    "right lung": ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
    "liver": ["liver"],
    "spleen": ["spleen"],
    "kidney": ["kidney_left", "kidney_right"],
    "kidneys": ["kidney_left", "kidney_right"],
    "left kidney": ["kidney_left"],
    "right kidney": ["kidney_right"],
    "heart": ["heart"],
    "bladder": ["urinary_bladder"],
    "urinary bladder": ["urinary_bladder"],
    "prostate": ["prostate"],
    "stomach": ["stomach"],
    "pancreas": ["pancreas"],
    "gallbladder": ["gallbladder"],
    "thyroid": ["thyroid_gland"],
    "thyroid gland": ["thyroid_gland"],
    "esophagus": ["esophagus"],
    "trachea": ["trachea"],
    "adrenal": ["adrenal_gland_left", "adrenal_gland_right"],
    "adrenals": ["adrenal_gland_left", "adrenal_gland_right"],
    "spine": ["sacrum", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1", "vertebrae_T12", "vertebrae_T1"],
    "vertebrae": ["vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1", "vertebrae_T12"],
}


def resolve_query(query: str) -> List[str]:
    q = query.strip().lower()
    if q in ORGAN_ALIASES:
        return ORGAN_ALIASES[q]
    matches = [name for name in ORGAN_NAMES if q in name or name in q]
    return matches if matches else [query]


class OrganLocator:
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = Config()

        ckpt_paths = {
            "resnet50": str(SHARON_DIR / "outputs" / "checkpoints" / "resnet50" / "best_model.pth"),
            "densenet121": str(SHARON_DIR / "outputs" / "checkpoints" / "densenet121" / "best_model.pth"),
            "swin_unetr": str(SHARON_DIR / "outputs" / "checkpoints" / "swin_unetr" / "best_model.pth"),
        }
        self.predictor = EnsembleTTAPredictor(ckpt_paths, device=self.device)

    def locate(self, case_name: str, query: str, output_image_path: str = None, presence_thresh: float = 0.25):
        ds = CTLocalizationDataset(str(SHARON_DIR / "dataset"), self.cfg.target_shape, self.cfg.heatmap_size, augment=False)
        case_idx = [i for i, c in enumerate(ds.cases) if c.name == case_name][0]
        sample = ds[case_idx]

        img = sample["image"].unsqueeze(0).to(self.device)
        spacing = sample["spacing"].numpy()
        gt_c = sample["gt_centroids"].numpy()
        found = sample["found_mask"].numpy()

        target_organs = resolve_query(query)
        print(f"\n=========================================================================")
        print(f"  ORGAN LOCATOR: Searching for '{query.upper()}' in Scan: {case_name}")
        print(f"  Resolved Target Structures: {target_organs}")
        print(f"=========================================================================")

        with torch.inference_mode():
            centroids_norm, heatmaps, confidences = self.predictor.predict(img, use_tta=True)

        ct_vol = img[0, 0].cpu().numpy()
        D, H, W = ct_vol.shape
        results = []

        for org_name in target_organs:
            if org_name not in ORGAN_NAMES:
                print(f"  ❌ Unknown organ '{org_name}'")
                continue

            oidx = ORGAN_NAMES.index(org_name)
            conf = float(confidences[0, oidx].item())
            norm_c = centroids_norm[0, oidx].cpu().numpy()

            if conf < presence_thresh:
                print(f"\n  ⚠️  {org_name.upper()}: OUTSIDE SCAN FIELD OF VIEW (Confidence: {conf:.1%} < {presence_thresh:.0%})")
                results.append({"name": org_name, "detected": False, "conf": conf})
                continue

            # Exact physical millimeter coordinates
            xyz_mm = (
                float(norm_c[2] * W * spacing[0]),
                float(norm_c[1] * H * spacing[1]),
                float(norm_c[0] * D * spacing[2]),
            )

            gt_present = bool(found[oidx])
            if gt_present:
                gt_xyz = (
                    float(gt_c[oidx, 2] * W * spacing[0]),
                    float(gt_c[oidx, 1] * H * spacing[1]),
                    float(gt_c[oidx, 0] * D * spacing[2]),
                )
                err_mm = float(np.linalg.norm(np.array(xyz_mm) - np.array(gt_xyz)))
            else:
                gt_xyz, err_mm = None, None

            print(f"\n  ✅ FOUND: {org_name.upper()} (Confidence: {conf:.1%})")
            print(f"     • Coordinates: (X={xyz_mm[0]:.1f}, Y={xyz_mm[1]:.1f}, Z={xyz_mm[2]:.1f}) mm" + (f" | Physical Error: {err_mm:.2f} mm" if err_mm is not None else ""))

            results.append({
                "name": org_name,
                "detected": True,
                "confidence": conf,
                "xyz_mm": xyz_mm,
                "norm_c": norm_c,
                "index": oidx,
                "err_mm": err_mm,
            })

        if output_image_path and any(r["detected"] for r in results):
            detected_res = [r for r in results if r["detected"]]
            n_plots = len(detected_res)
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6), facecolor="#111216")
            if n_plots == 1:
                axes = [axes]

            for ax_i, r in enumerate(detected_res):
                oidx = r["index"]
                r_norm = r["norm_c"]
                pz = int(np.clip(round(r_norm[0] * D), 0, D - 1))
                hm_z = int(np.clip(round(r_norm[0] * 64), 0, 63))

                ct_slice = ct_vol[pz, :, :]
                hm_slice = heatmaps[0, oidx, hm_z].cpu().numpy()

                px = r_norm[2] * W
                py = r_norm[1] * H

                ax = axes[ax_i]
                ax.imshow(ct_slice, cmap="gray", origin="lower")
                ax.imshow(hm_slice, cmap="hot", alpha=0.45, extent=[0, W, 0, H], origin="lower")
                ax.plot(px, py, 'g+', markersize=18, markeredgewidth=3.0, label="Predicted Centroid")

                if found[oidx]:
                    gx = gt_c[oidx, 2] * W
                    gy = gt_c[oidx, 1] * H
                    ax.plot(gx, gy, 'bx', markersize=12, markeredgewidth=2.0, label="GT Centroid")

                err_str = f" | Err: {r['err_mm']:.1f}mm" if r['err_mm'] is not None else ""
                ax.set_title(
                    f"Target: {r['name'].replace('_', ' ').title()}\n"
                    f"Conf: {r['confidence']:.1%}{err_str}\n"
                    f"Coords: ({r['xyz_mm'][0]:.0f}, {r['xyz_mm'][1]:.0f}, {r['xyz_mm'][2]:.0f}) mm",
                    color="white", fontsize=11, fontweight="bold"
                )
                ax.axis("off")
                ax.legend(loc="upper right", facecolor="#1e222d", labelcolor="white", fontsize=9)

            fig.suptitle(f"Organ Location Finder: '{query.upper()}' | Scan: {case_name}", color="white", fontsize=14, fontweight="bold", y=0.98)
            plt.tight_layout()
            plt.savefig(output_image_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()
            print(f"\n[SUCCESS] Visual diagnostic panel saved to: {output_image_path}")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default="case_022")
    parser.add_argument("--query", type=str, default="lungs")
    parser.add_argument("--output", type=str, default="/home/sharon/Desktop/locate_lungs_output.png")
    args = parser.parse_args()

    locator = OrganLocator()
    locator.locate(args.case_name, args.query, output_image_path=args.output)
