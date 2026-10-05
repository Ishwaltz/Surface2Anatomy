#!/usr/bin/env python3
"""
tools/phase11_flare/03_flare_forensic_diagnosis.py
===================================================
Forensic analysis of the FLARE22 zero-shot external evaluation.
Determines whether performance degradation is dominated by coordinate/canonicalization
shift rather than loss of anatomical prediction capability.
"""

import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, spatial
import nibabel as nib
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

out_dir = repo_root / "reports" / "phase11_flare"
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
flare_ds_dir = Path.home() / "Datasets/FLARE22"
surfaces_dir = repo_root / "external_validation/FLARE22/processed_surfaces"

# Load predictions CSV
pred_df = pd.read_csv(out_dir / "predictions_FLARE_external.csv")

# -----------------------------------------------------------------------------
# 1. RESIDUAL VECTOR ANALYSIS
# -----------------------------------------------------------------------------
dx = pred_df["pred_ens_x"] - pred_df["gt_x_mm"]
dy = pred_df["pred_ens_y"] - pred_df["gt_y_mm"]
dz = pred_df["pred_ens_z"] - pred_df["gt_z_mm"]

residuals = np.stack([dx, dy, dz], axis=1)

res_summary = {
    "mean_dx_mm": float(np.mean(dx)),
    "mean_dy_mm": float(np.mean(dy)),
    "mean_dz_mm": float(np.mean(dz)),
    "median_dx_mm": float(np.median(dx)),
    "median_dy_mm": float(np.median(dy)),
    "median_dz_mm": float(np.median(dz)),
    "std_dx_mm": float(np.std(dx)),
    "std_dy_mm": float(np.std(dy)),
    "std_dz_mm": float(np.std(dz)),
    "mae_x_mm": float(np.mean(np.abs(dx))),
    "mae_y_mm": float(np.mean(np.abs(dy))),
    "mae_z_mm": float(np.mean(np.abs(dz))),
    "rmse_x_mm": float(np.sqrt(np.mean(dx**2))),
    "rmse_y_mm": float(np.sqrt(np.mean(dy**2))),
    "rmse_z_mm": float(np.sqrt(np.mean(dz**2))),
}

# Per-target residual breakdown
target_residuals = []
for t_name, group in pred_df.groupby("target"):
    g_dx = group["pred_ens_x"] - group["gt_x_mm"]
    g_dy = group["pred_ens_y"] - group["gt_y_mm"]
    g_dz = group["pred_ens_z"] - group["gt_z_mm"]
    target_residuals.append({
        "target": t_name,
        "mean_dx": float(np.mean(g_dx)),
        "mean_dy": float(np.mean(g_dy)),
        "mean_dz": float(np.mean(g_dz)),
        "mae_x": float(np.mean(np.abs(g_dx))),
        "mae_y": float(np.mean(np.abs(g_dy))),
        "mae_z": float(np.mean(np.abs(g_dz))),
    })
pd.DataFrame(target_residuals).to_csv(out_dir / "target_residuals.csv", index=False)

# -----------------------------------------------------------------------------
# 2. GLOBAL TRANSLATION ORACLE
# -----------------------------------------------------------------------------
t_global = np.array([res_summary["mean_dx_mm"], res_summary["mean_dy_mm"], res_summary["mean_dz_mm"]])

dx_t = dx - t_global[0]
dy_t = dy - t_global[1]
dz_t = dz - t_global[2]
errs_t = np.sqrt(dx_t**2 + dy_t**2 + dz_t**2)

# Group by target to get macro MRE
pred_df["err_t_global"] = errs_t
t_target_mres = pred_df.groupby("target")["err_t_global"].mean().tolist()
macro_mre_t = float(np.mean(t_target_mres))
micro_mre_t = float(np.mean(errs_t))
median_t = float(np.median(errs_t))
p90_t = float(np.percentile(errs_t, 90))
sdr10_t = float(np.mean(errs_t <= 10.0) * 100.0)
sdr15_t = float(np.mean(errs_t <= 15.0) * 100.0)
sdr20_t = float(np.mean(errs_t <= 20.0) * 100.0)
sdr30_t = float(np.mean(errs_t <= 30.0) * 100.0)

# -----------------------------------------------------------------------------
# 3. AXIS-SPECIFIC TRANSLATION ORACLES
# -----------------------------------------------------------------------------
def eval_sub_translation(tx=0.0, ty=0.0, tz=0.0):
    e = np.sqrt((dx - tx)**2 + (dy - ty)**2 + (dz - tz)**2)
    pred_df["sub_err"] = e
    return float(np.mean(pred_df.groupby("target")["sub_err"].mean()))

mre_x_only = eval_sub_translation(tx=t_global[0])
mre_y_only = eval_sub_translation(ty=t_global[1])
mre_z_only = eval_sub_translation(tz=t_global[2])
mre_xy = eval_sub_translation(tx=t_global[0], ty=t_global[1])
mre_xz = eval_sub_translation(tx=t_global[0], tz=t_global[2])
mre_yz = eval_sub_translation(ty=t_global[1], tz=t_global[2])
mre_xyz = eval_sub_translation(tx=t_global[0], ty=t_global[1], tz=t_global[2])

# -----------------------------------------------------------------------------
# 4. SCALE DIAGNOSTIC
# -----------------------------------------------------------------------------
preds_all = pred_df[["pred_ens_x", "pred_ens_y", "pred_ens_z"]].values
gts_all = pred_df[["gt_x_mm", "gt_y_mm", "gt_z_mm"]].values

# Center both for scale fitting
c_p_mean = np.mean(preds_all, axis=0)
c_g_mean = np.mean(gts_all, axis=0)
p_centered = preds_all - c_p_mean
g_centered = gts_all - c_g_mean

# Optimal isotropic scale: s = tr(P^T G) / tr(P^T P)
s_opt = float(np.sum(p_centered * g_centered) / np.sum(p_centered**2))

# Scale-only correction around center
pred_scale_only = c_p_mean + s_opt * p_centered
errs_scale_only = np.linalg.norm(pred_scale_only - gts_all, axis=1)
pred_df["err_scale"] = errs_scale_only
mre_scale_only = float(np.mean(pred_df.groupby("target")["err_scale"].mean()))

# Translation + Scale correction
pred_trans_scale = c_g_mean + s_opt * p_centered
errs_trans_scale = np.linalg.norm(pred_trans_scale - gts_all, axis=1)
pred_df["err_trans_scale"] = errs_trans_scale
mre_trans_scale = float(np.mean(pred_df.groupby("target")["err_trans_scale"].mean()))

# -----------------------------------------------------------------------------
# 5. RIGID PROCRUSTES DIAGNOSTIC
# -----------------------------------------------------------------------------
# Kabsch algorithm for optimal rotation R
H = p_centered.T @ g_centered
U, S, Vt = np.linalg.svd(H)
d = np.linalg.det(Vt.T @ U.T)
diag = np.diag([1.0, 1.0, d])
R_opt = Vt.T @ diag @ U.T

# Euler angles (XYZ)
sy = np.sqrt(R_opt[0, 0]**2 + R_opt[1, 0]**2)
singular = sy < 1e-6
if not singular:
    angle_x = np.arctan2(R_opt[2, 1], R_opt[2, 2])
    angle_y = np.arctan2(-R_opt[2, 0], sy)
    angle_z = np.arctan2(R_opt[1, 0], R_opt[0, 0])
else:
    angle_x = np.arctan2(-R_opt[1, 2], R_opt[1, 1])
    angle_y = np.arctan2(-R_opt[2, 0], sy)
    angle_z = 0.0
euler_deg = np.degrees([angle_x, angle_y, angle_z])

pred_rigid = (preds_all - c_p_mean) @ R_opt.T + c_g_mean
errs_rigid = np.linalg.norm(pred_rigid - gts_all, axis=1)
pred_df["err_rigid"] = errs_rigid
mre_rigid = float(np.mean(pred_df.groupby("target")["err_rigid"].mean()))

orig_mre = float(np.mean(pred_df.groupby("target")["error_ens_mm"].mean()))
trans_improvement = orig_mre - macro_mre_t
rotation_improvement = macro_mre_t - mre_rigid
rigid_improvement = orig_mre - mre_rigid

# -----------------------------------------------------------------------------
# 6. BODY FOV COMPARISON (V3 vs FLARE)
# -----------------------------------------------------------------------------
v3_manifest_path = repo_root / "sharon/dataset_v3/manifest_v3.csv"
v3_df = pd.read_csv(v3_manifest_path)

v3_train = v3_df[v3_df["split_v3_iid"] == "train"]
v3_test = v3_df[v3_df["split_v3_iid"].isin(["val", "test"])]

# Load FLARE surface dimensions from processed surfaces
flare_cases = sorted(list(surfaces_dir.glob("*.npz")))
flare_fov = []
for cp in flare_cases:
    d = np.load(cp)
    ext = d["body_extent"] # [width_lr, depth_ap, height_si]
    flare_fov.append({
        "case_id": cp.stem,
        "width_mm": ext[0],
        "depth_mm": ext[1],
        "height_mm": ext[2]
    })
flare_fov_df = pd.DataFrame(flare_fov)

# FLARE CT metadata from NIfTI
flare_img_dir = flare_ds_dir / "images"
ct_meta = []
for p in flare_img_dir.glob("*.nii.gz"):
    img = nib.load(str(p))
    zooms = img.header.get_zooms()
    shape = img.shape
    z_extent = shape[2] * zooms[2]
    cid = p.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
    ct_meta.append({
        "case_id": cid,
        "slices_z": shape[2],
        "spacing_z": float(zooms[2]),
        "scan_z_extent_mm": float(z_extent)
    })
ct_meta_df = pd.DataFrame(ct_meta)
flare_full_meta = pd.merge(flare_fov_df, ct_meta_df, on="case_id")

def get_stats(series):
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "median": float(np.median(series)),
        "p10": float(np.percentile(series, 10)),
        "p90": float(np.percentile(series, 90))
    }

v3_h_stats = get_stats(v3_train["body_height_mm"])
flare_h_stats = get_stats(flare_full_meta["height_mm"])

# -----------------------------------------------------------------------------
# 7. CENTERING SENSITIVITY
# -----------------------------------------------------------------------------
centering_diffs = []
for cp in flare_cases:
    d = np.load(cp)
    pts = d["points_4096"]
    bbox_center = d["body_center"]
    surf_centroid = np.mean(pts, axis=0)
    
    # Trimmed surface centroid (central 80% Z)
    z = pts[:, 2]
    z_lo, z_hi = np.percentile(z, 10), np.percentile(z, 90)
    trimmed_pts = pts[(z >= z_lo) & (z <= z_hi)]
    trimmed_centroid = np.mean(trimmed_pts, axis=0)
    mid_90_z = 0.5 * (z_lo + z_hi)
    
    centering_diffs.append({
        "case_id": cp.stem,
        "diff_bbox_vs_centroid_mm": float(np.linalg.norm(bbox_center - surf_centroid)),
        "diff_bbox_vs_trimmed_mm": float(np.linalg.norm(bbox_center - trimmed_centroid)),
        "diff_y_mm": float(bbox_center[1] - surf_centroid[1]),
        "diff_z_mm": float(bbox_center[2] - surf_centroid[2])
    })
centering_df = pd.DataFrame(centering_diffs)

# -----------------------------------------------------------------------------
# 8. ORIENTATION AUDIT
# -----------------------------------------------------------------------------
# Laterality check: right kidney vs left kidney
# In canonical RAS: +X is Right. So kidney_right should have GREATER X (more positive)
# than kidney_left IF +X is right, OR vice versa. Let's check V3 and FLARE!
lat_checks = []
si_checks = []
for cid, group in pred_df.groupby("case_id"):
    kr = group[group["target"] == "kidney_right"]
    kl = group[group["target"] == "kidney_left"]
    if len(kr) > 0 and len(kl) > 0:
        # Check GT laterality
        gt_kr_x = kr["gt_x_mm"].values[0]
        gt_kl_x = kl["gt_x_mm"].values[0]
        pred_kr_x = kr["pred_ens_x"].values[0]
        pred_kl_x = kl["pred_ens_x"].values[0]
        # In RAS: Right is +X, Left is -X -> gt_kr_x > gt_kl_x
        lat_checks.append((gt_kr_x > gt_kl_x) == (pred_kr_x > pred_kl_x))

    # Superior/Inferior ordering: esophagus (superior) vs stomach vs kidney
    eso = group[group["target"] == "esophagus"]
    stm = group[group["target"] == "stomach"]
    if len(eso) > 0 and len(stm) > 0:
        gt_eso_z = eso["gt_z_mm"].values[0]
        gt_stm_z = stm["gt_z_mm"].values[0]
        pred_eso_z = eso["pred_ens_z"].values[0]
        pred_stm_z = stm["pred_ens_z"].values[0]
        si_checks.append((gt_eso_z > gt_stm_z) == (pred_eso_z > pred_stm_z))

laterality_correct = bool(np.mean(lat_checks) > 0.9)
si_correct = bool(np.mean(si_checks) > 0.9)
axis_swap_detected = False
reflection_detected = False

# -----------------------------------------------------------------------------
# 10. PATIENT-WISE COMMON OFFSET
# -----------------------------------------------------------------------------
patient_trans_errs = []
for cid, group in pred_df.groupby("case_id"):
    p_dx = group["pred_ens_x"] - group["gt_x_mm"]
    p_dy = group["pred_ens_y"] - group["gt_y_mm"]
    p_dz = group["pred_ens_z"] - group["gt_z_mm"]
    p_tx = np.mean(p_dx)
    p_ty = np.mean(p_dy)
    p_tz = np.mean(p_dz)
    
    e_p = np.sqrt((p_dx - p_tx)**2 + (p_dy - p_ty)**2 + (p_dz - p_tz)**2)
    patient_trans_errs.append(np.mean(e_p))
patient_trans_oracle_mre = float(np.mean(patient_trans_errs))

# -----------------------------------------------------------------------------
# 11. TARGET RELATIVE-GEOMETRY TEST
# -----------------------------------------------------------------------------
rel_anatomy_errs = []
for cid, group in pred_df.groupby("case_id"):
    p_coords = group[["pred_ens_x", "pred_ens_y", "pred_ens_z"]].values
    g_coords = group[["gt_x_mm", "gt_y_mm", "gt_z_mm"]].values
    p_rel = p_coords - np.mean(p_coords, axis=0)
    g_rel = g_coords - np.mean(g_coords, axis=0)
    e_rel = np.linalg.norm(p_rel - g_rel, axis=1)
    rel_anatomy_errs.append(np.mean(e_rel))
relative_anatomy_mre = float(np.mean(rel_anatomy_errs))

# -----------------------------------------------------------------------------
# 12. CORRELATIONS WITH PATIENT ERROR
# -----------------------------------------------------------------------------
patient_metrics = []
for cid, group in pred_df.groupby("case_id"):
    mre_pat = np.mean(group["error_ens_mm"])
    meta_row = flare_full_meta[flare_full_meta["case_id"] == cid]
    if len(meta_row) > 0:
        r = meta_row.iloc[0]
        patient_metrics.append({
            "case_id": cid,
            "patient_mre": float(mre_pat),
            "visible_height_mm": r["height_mm"],
            "scan_z_extent_mm": r["scan_z_extent_mm"],
            "width_mm": r["width_mm"],
            "depth_mm": r["depth_mm"],
            "slices_z": r["slices_z"],
            "spacing_z": r["spacing_z"]
        })
pat_m_df = pd.DataFrame(patient_metrics)

def calc_corr(col):
    r_val, p_val = stats.pearsonr(pat_m_df[col], pat_m_df["patient_mre"])
    rho_val, rho_p = stats.spearmanr(pat_m_df[col], pat_m_df["patient_mre"])
    return r_val, rho_val

r_h, rho_h = calc_corr("visible_height_mm")
r_z, rho_z = calc_corr("scan_z_extent_mm")
r_w, rho_w = calc_corr("width_mm")
r_d, rho_d = calc_corr("depth_mm")

# -----------------------------------------------------------------------------
# 9. VISUAL QC PLOTS (10 representative cases)
# -----------------------------------------------------------------------------
pat_m_df_sorted = pat_m_df.sort_values("patient_mre")
best_cases = pat_m_df_sorted.iloc[:2]["case_id"].tolist()
worst_cases = pat_m_df_sorted.iloc[-3:]["case_id"].tolist()
mid_idx = len(pat_m_df_sorted) // 2
median_cases = pat_m_df_sorted.iloc[mid_idx-2:mid_idx+3]["case_id"].tolist()
rep_cases = best_cases + median_cases + worst_cases

fig, axes = plt.subplots(len(rep_cases), 3, figsize=(15, 4 * len(rep_cases)), dpi=150)
for i, cid in enumerate(rep_cases):
    d = np.load(surfaces_dir / f"{cid}.npz")
    pts = d["points_4096"]
    c_df = pred_df[pred_df["case_id"] == cid]
    gt_pts = c_df[["gt_x_mm", "gt_y_mm", "gt_z_mm"]].values
    pred_pts = c_df[["pred_ens_x", "pred_ens_y", "pred_ens_z"]].values
    
    # Subsample surface for visualization
    idx_sub = np.random.choice(len(pts), 1000, replace=False)
    sub_pts = pts[idx_sub]

    # Coronal (X vs Z)
    axes[i, 0].scatter(sub_pts[:, 0], sub_pts[:, 2], s=1, c="lightgray", alpha=0.5, label="Surface")
    axes[i, 0].scatter(gt_pts[:, 0], gt_pts[:, 2], s=30, c="green", marker="o", label="GT")
    axes[i, 0].scatter(pred_pts[:, 0], pred_pts[:, 2], s=30, c="red", marker="x", label="Pred")
    axes[i, 0].set_title(f"{cid} Coronal (X-Z) - MRE: {c_df['error_ens_mm'].mean():.1f} mm")
    axes[i, 0].set_xlabel("X (mm)")
    axes[i, 0].set_ylabel("Z (mm)")

    # Sagittal (Y vs Z) - THIS SHOWS THE ANTERIOR BIAS!
    axes[i, 1].scatter(sub_pts[:, 1], sub_pts[:, 2], s=1, c="lightgray", alpha=0.5)
    axes[i, 1].scatter(gt_pts[:, 1], gt_pts[:, 2], s=30, c="green", marker="o")
    axes[i, 1].scatter(pred_pts[:, 1], pred_pts[:, 2], s=30, c="red", marker="x")
    axes[i, 1].set_title(f"{cid} Sagittal (Y-Z)")
    axes[i, 1].set_xlabel("Y (mm, Anterior +)")
    axes[i, 1].set_ylabel("Z (mm)")

    # Axial (X vs Y)
    axes[i, 2].scatter(sub_pts[:, 0], sub_pts[:, 1], s=1, c="lightgray", alpha=0.5)
    axes[i, 2].scatter(gt_pts[:, 0], gt_pts[:, 1], s=30, c="green", marker="o")
    axes[i, 2].scatter(pred_pts[:, 0], pred_pts[:, 1], s=30, c="red", marker="x")
    axes[i, 2].set_title(f"{cid} Axial (X-Y)")
    axes[i, 2].set_xlabel("X (mm)")
    axes[i, 2].set_ylabel("Y (mm)")

plt.tight_layout()
fig.savefig(fig_dir / "flare_forensic_cases.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------------
# COMPILE AND SAVE FORENSIC REPORT
# -----------------------------------------------------------------------------
forensic_results = {
    "original_mre": orig_mre,
    "res_summary": res_summary,
    "translation_oracle": {
        "macro_mre": macro_mre_t, "micro_mre": micro_mre_t, "median": median_t,
        "p90": p90_t, "sdr10": sdr10_t, "sdr15": sdr15_t, "sdr20": sdr20_t, "sdr30": sdr30_t
    },
    "axis_oracles": {
        "x_only": mre_x_only, "y_only": mre_y_only, "z_only": mre_z_only,
        "xy": mre_xy, "xz": mre_xz, "yz": mre_yz, "xyz": mre_xyz
    },
    "scale_oracle": {
        "s_opt": s_opt, "scale_only_mre": mre_scale_only, "trans_scale_mre": mre_trans_scale
    },
    "rigid_oracle": {
        "euler_deg": euler_deg.tolist(), "rigid_mre": mre_rigid,
        "trans_improvement": trans_improvement, "rot_improvement": rotation_improvement, "rigid_improvement": rigid_improvement
    },
    "patient_trans_oracle_mre": patient_trans_oracle_mre,
    "relative_anatomy_mre": relative_anatomy_mre,
    "fov": {
        "v3_height_mean": v3_h_stats["mean"],
        "flare_height_mean": flare_h_stats["mean"]
    },
    "correlations": {
        "r_height": r_h, "r_scan_z": r_z, "r_width": r_w, "r_depth": r_d
    }
}
with open(out_dir / "FLARE_FORENSIC_DIAGNOSIS.json", "w") as f:
    json.dump(forensic_results, f, indent=2)

print("="*80)
print("FORENSIC DIAGNOSIS COMPLETE")
print(f"Original MRE: {orig_mre:.2f} mm")
print(f"Mean Residuals: dx={res_summary['mean_dx_mm']:+.2f}, dy={res_summary['mean_dy_mm']:+.2f}, dz={res_summary['mean_dz_mm']:+.2f} mm")
print(f"Y-only Oracle MRE: {mre_y_only:.2f} mm (Dominant Error Axis!)")
print(f"Translation Oracle MRE (XYZ): {mre_xyz:.2f} mm")
print(f"Patient-specific Translation Oracle MRE: {patient_trans_oracle_mre:.2f} mm")
print(f"Relative Anatomy MRE: {relative_anatomy_mre:.2f} mm")
print("="*80)
