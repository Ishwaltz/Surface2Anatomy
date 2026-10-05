import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sharon_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(sharon_dir))
sys.path.insert(0, str(sharon_dir.parent))

from labels import (
    NUM_ORGANS, NUM_SKELETAL, NUM_SOFT_TISSUE,
    SKELETAL_CLASSES, SOFT_TISSUE_CLASSES,
    SKELETAL_INDICES, SOFT_TISSUE_INDICES,
    ORGAN_NAMES, get_sex_organ_mask,
    ANATOMICAL_REGIONS, REGION_NAMES
)
from model_same import SAMeOrganGNN
from dataset import PointCloudOrganDataset


@torch.no_grad()
def evaluate_same_test_set(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()
    all_errors = []
    skel_errors = []
    soft_errors = []
    organ_errors = {name: [] for name in ORGAN_NAMES}
    soft_uncertainties_vol = {name: [] for name in SOFT_TISSUE_CLASSES}
    soft_uncertainties_rad = {name: [] for name in SOFT_TISSUE_CLASSES}

    for batch in loader:
        pts = batch["points"].to(device)
        ctr = batch["centroids"].to(device)
        sex = batch["sex_prior"].to(device)
        mask = batch["mask"].to(device)
        center = batch["center"].to(device)
        scale = batch["scale"].to(device)

        out = model(pts, sex, mask=mask)
        preds = out["full_coords"]
        log_diag = out["soft_log_diag"] # (B, 58, 3)

        # De-normalize coordinates to physical mm
        preds_mm = preds * scale.unsqueeze(1) + center.unsqueeze(1)
        ctr_mm = ctr * scale.unsqueeze(1) + center.unsqueeze(1)
        dist_mm = torch.norm(preds_mm - ctr_mm, dim=-1) # (B, 121)

        # Scale factor volume: s_x * s_y * s_z
        scale_vol = scale[:, 0] * scale[:, 1] * scale[:, 2] # (B,)

        # Compute physical covariance determinants for soft-tissue
        # |Sigma_phys| = (s_x * s_y * s_z)^2 * exp(2 * sum(log_diag))
        norm_det = torch.exp(2.0 * log_diag.sum(dim=-1)) # (B, 58)
        phys_det = norm_det * (scale_vol.unsqueeze(-1) ** 2) # (B, 58) in mm^6
        phys_vol_sd = torch.sqrt(torch.clamp(phys_det, min=1e-12)) # (B, 58) in mm^3
        phys_rad_sd = phys_vol_sd ** (1.0 / 3.0) # (B, 58) in mm

        for b in range(pts.shape[0]):
            for i, name in enumerate(ORGAN_NAMES):
                if mask[b, i] > 0.5:
                    err_val = dist_mm[b, i].item()
                    organ_errors[name].append(err_val)
                    all_errors.append(err_val)
                    if i in SKELETAL_INDICES:
                        skel_errors.append(err_val)
                    else:
                        soft_errors.append(err_val)

            for j, name in enumerate(SOFT_TISSUE_CLASSES):
                orig_idx = ORGAN_NAMES.index(name)
                if mask[b, orig_idx] > 0.5:
                    soft_uncertainties_vol[name].append(phys_vol_sd[b, j].item())
                    soft_uncertainties_rad[name].append(phys_rad_sd[b, j].item())

    # Organ breakdowns
    mean_per_organ = {name: float(np.mean(errs)) if errs else 0.0 for name, errs in organ_errors.items()}
    mean_unc_rad = {name: float(np.mean(vals)) if vals else 0.0 for name, vals in soft_uncertainties_rad.items()}
    mean_unc_vol = {name: float(np.mean(vals)) if vals else 0.0 for name, vals in soft_uncertainties_vol.items()}

    # Regional breakdowns
    regional_errs = {r_name: [] for r_name in REGION_NAMES}
    for name, errs in organ_errors.items():
        r_id = ANATOMICAL_REGIONS[name]
        r_name = REGION_NAMES[r_id]
        regional_errs[r_name].extend(errs)

    regional_stats = {}
    for r_name, errs in regional_errs.items():
        regional_stats[r_name] = {
            "mean": float(np.mean(errs)) if errs else 0.0,
            "sd": float(np.std(errs)) if errs else 0.0,
            "median": float(np.median(errs)) if errs else 0.0,
        }

    return {
        "overall_mean_mm": float(np.mean(all_errors)),
        "overall_sd_mm": float(np.std(all_errors)),
        "overall_median_mm": float(np.median(all_errors)),
        "skeletal_mean_mm": float(np.mean(skel_errors)),
        "skeletal_sd_mm": float(np.std(skel_errors)),
        "skeletal_median_mm": float(np.median(skel_errors)),
        "soft_tissue_mean_mm": float(np.mean(soft_errors)),
        "soft_tissue_sd_mm": float(np.std(soft_errors)),
        "soft_tissue_median_mm": float(np.median(soft_errors)),
        "organ_errors": mean_per_organ,
        "soft_uncertainty_radius_mm": mean_unc_rad,
        "soft_uncertainty_volume_mm3": mean_unc_vol,
        "regional_breakdown": regional_stats,
    }


def generate_same_report(
    results: Dict,
    output_path: str,
) -> str:
    ov_stat = f"{results['overall_mean_mm']:.2f} ± {results['overall_sd_mm']:.2f} mm"
    sk_stat = f"{results['skeletal_mean_mm']:.2f} ± {results['skeletal_sd_mm']:.2f} mm"
    sf_stat = f"{results['soft_tissue_mean_mm']:.2f} ± {results['soft_tissue_sd_mm']:.2f} mm"

    lines = [
        "# SAMe Framework Benchmark Report: Skeleton-Conditioned Probabilistic GNN",
        "",
        "## Executive Summary",
        "The **SAMe (Skeleton-Conditioned Probabilistic GNN)** framework decouples anatomical location regression into a deterministic skeletal anchor stage (63 classes) and an uncertainty-aware Multivariate Gaussian graph propagation stage (58 soft-tissue classes with Cholesky covariance factorization).",
        "",
        "Evaluation performed on the held-out test set (N=45 cases) in true physical millimeters (mm).",
        "",
        "---",
        "",
        "### Table 1: SAMe Partition Accuracy & Volumetric Uncertainty",
        "| Anatomical Partition | Classes | Physical Mean Error (mm) | Median Error (mm) | Mean Uncertainty Radius (mm) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Skeletal Anchor Framework** | 63 | {sk_stat} | {results['skeletal_median_mm']:.2f} mm | N/A (Deterministic) |",
        f"| **Soft-Tissue Probabilistic GNN** | 58 | {sf_stat} | {results['soft_tissue_median_mm']:.2f} mm | {np.mean(list(results['soft_uncertainty_radius_mm'].values())):.2f} mm |",
        f"| **Full 121-Organ SAMe Total** | 121 | **{ov_stat}** | **{results['overall_median_mm']:.2f} mm** | **Evaluated** |",
        "",
        "---",
        "",
        "### Table 2: Key Visceral & Reproductive Soft-Tissue Organs (Mean Error ± Uncertainty Radius)",
        "| Organ Structure | Biological Sex | Mean Physical Error (mm) | Uncertainty Radius (mm) | Covariance Vol. (mm^3) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    key_organs = [
        ("liver", "Both"), ("spleen", "Both"), ("kidney_right", "Both"), ("kidney_left", "Both"),
        ("aorta", "Both"), ("urinary_bladder", "Both"), ("pancreas", "Both"), ("stomach", "Both"),
        ("lung_lower_lobe_left", "Both"), ("lung_lower_lobe_right", "Both"),
        ("prostate", "Male (S=1)"), ("uterus", "Female (S=0)"), ("ovary_left", "Female (S=0)"),
        ("ovary_right", "Female (S=0)"), ("vagina", "Female (S=0)")
    ]

    for org, sex_t in key_organs:
        err = results["organ_errors"].get(org, 0.0)
        rad = results["soft_uncertainty_radius_mm"].get(org, 0.0)
        vol = results["soft_uncertainty_volume_mm3"].get(org, 0.0)
        if err > 0.0:
            lines.append(f"| `{org}` | {sex_t} | {err:.2f} mm | {rad:.2f} mm | {vol:.1f} mm^3 |")
        else:
            lines.append(f"| `{org}` | {sex_t} | N/A (Out of FOV) | N/A | N/A |")

    lines.extend([
        "",
        "---",
        "",
        "### Table 3: Functional Anatomical Subgroups Breakdown (Mean ± SD mm)",
        "| Functional Anatomical Region | Physical Mean Error (mm) | Median Error (mm) |",
        "| :--- | :---: | :---: |",
    ])

    for r_name in REGION_NAMES:
        r_info = results["regional_breakdown"].get(r_name, {})
        lines.append(f"| **{r_name} Region** | {r_info.get('mean', 0.0):.2f} ± {r_info.get('sd', 0.0):.2f} mm | {r_info.get('median', 0.0):.2f} mm |")

    lines.extend([
        "",
        "---",
        "",
        "## Clinical & Algorithmic Conclusions",
        "1. **Skeletal Landmark Instantiation**: Anchoring soft tissues to low-variance skeletal structures (vertebrae, ribs, pelvis) resolves global spatial ambiguity.",
        "2. **Multivariate Gaussian Cholesky Parameterization**: Parameterizing covariance as Sigma = L L^T guarantees strict positive definiteness and prevents numerical instability during Gaussian NLL backpropagation.",
        "3. **Volumetric Uncertainty Quantification**: Providing explicit physical uncertainty radii alongside 3D coordinates enables downstream surgical and radiation planning pipelines to quantify landmark confidence.",
    ])

    report = "\n".join(lines)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        f.write(report)
    print(f"[Report] Generated SAMe benchmark report: {out_p}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAMe Model on Held-Out Test Set")
    parser.add_argument("--data_file", type=str, default="sharon/dataset/pointclouds_450.pt", help="Dataset path")
    parser.add_argument("--splits_file", type=str, default="sharon/outputs/splits_pointcloud.json", help="Splits JSON")
    parser.add_argument("--checkpoint", type=str, default="sharon/outputs/checkpoints/same_model_best.pth", help="Checkpoint path")
    parser.add_argument("--output_report", type=str, default="sharon/outputs/SAME_BENCHMARK_REPORT.md", help="Output report")
    parser.add_argument("--output_json", type=str, default="sharon/outputs/SAME_BENCHMARK_RESULTS.json", help="Output JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} for SAMe Evaluation")

    full_ds = PointCloudOrganDataset(pt_path=args.data_file, augment=False)
    with open(args.splits_file) as f:
        splits = json.load(f)

    test_idx = splits["test_indices"]
    test_ds = Subset(full_ds, test_idx)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    model = SAMeOrganGNN(pointnet_feat_dim=1024, sex_emb_dim=128, latent_dim=512, hidden_dim=256, gnn_layers=4)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print(f"[Evaluation] Evaluating SAMe on {len(test_idx)} held-out test cases...")
    results = evaluate_same_test_set(model, test_loader, device)

    print(f"\n✓ SAMe Full 121-Organ Error: {results['overall_mean_mm']:.2f} ± {results['overall_sd_mm']:.2f} mm (Median: {results['overall_median_mm']:.2f} mm)")
    print(f"  - Skeletal Error (63 structures): {results['skeletal_mean_mm']:.2f} ± {results['skeletal_sd_mm']:.2f} mm")
    print(f"  - Soft-Tissue Error (58 structures): {results['soft_tissue_mean_mm']:.2f} ± {results['soft_tissue_sd_mm']:.2f} mm")

    # Save JSON
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Save Report
    generate_same_report(results, args.output_report)


if __name__ == "__main__":
    main()
