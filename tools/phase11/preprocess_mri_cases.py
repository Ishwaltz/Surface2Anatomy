#!/usr/bin/env python3
"""
tools/phase11/preprocess_mri_cases.py

Batch preprocessing for AMOS MRI cases (60 cases: 40 train + 20 val):
- Canonical RAS reorientation
- Intensity-only surface extraction (corner background threshold + 3D morphological closing)
- Marching cubes to physical skin mesh in mm
- 4096 uniform point sampling with body centering and S_global = 500 mm
- GT centroid extraction for AMOS organs 1-15
- FOV metrics and categorization
- Visual QC panels for 20 MRI cases
- Multi-process execution using 8 workers
"""

import os
import sys
import glob
import time
import json
import concurrent.futures
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage import measure
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POINTCLOUDS_DIR = "data_external/AMOS22/processed/pointclouds"
GT_CENTROIDS_PATH = "data_external/AMOS22/processed/AMOS_mri_GT_centroids.csv"
FOV_AUDIT_PATH = "reports/phase11/amos/AMOS_mri_fov_audit.csv"
QC_DIR = "reports/phase11/qc/AMOS"
MAPPING_PATH = "reports/phase11/mappings/AMOS_target_mapping.csv"

S_GLOBAL_MM = 500.0
NUM_POINTS = 4096

os.makedirs(POINTCLOUDS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GT_CENTROIDS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FOV_AUDIT_PATH), exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)

target_mapping = pd.read_csv(MAPPING_PATH)

def extract_mri_body_mask(volume_mri):
    corners = np.concatenate([
        volume_mri[:10, :10, :].ravel(),
        volume_mri[-10:, :10, :].ravel(),
        volume_mri[:10, -10:, :].ravel(),
        volume_mri[-10:, -10:, :].ravel()
    ])
    bg_mean = np.mean(corners)
    bg_std = np.std(corners)
    thresh = bg_mean + 1.5 * max(bg_std, 1.0)
    binary = volume_mri > thresh
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return binary
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    body_mask = (labeled == largest_label)
    structure = ndimage.generate_binary_structure(3, 1)
    closed_mask = ndimage.binary_closing(body_mask, structure=structure, iterations=2)
    for z in range(closed_mask.shape[2]):
        if np.any(closed_mask[:, :, z]):
            closed_mask[:, :, z] = ndimage.binary_fill_holes(closed_mask[:, :, z])
    return closed_mask

def process_case_task(args):
    img_path, lbl_path, case_id, modality, do_qc = args
    try:
        npz_out = os.path.join(POINTCLOUDS_DIR, f"{case_id}.npz")
        
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path) if lbl_path and os.path.exists(lbl_path) else None

        img_ras = nib.as_closest_canonical(img_nii)
        img_data = img_ras.get_fdata()
        affine_ras = img_ras.affine

        body_mask = extract_mri_body_mask(img_data)
        if not np.any(body_mask):
            return None, None, {"case_id": case_id, "valid": False, "reason": "Empty body mask"}

        verts_vox, faces, normals, values = measure.marching_cubes(body_mask.astype(np.uint8), level=0.5, step_size=1)
        verts_world = nib.affines.apply_affine(affine_ras, verts_vox)

        min_xyz = verts_world.min(axis=0)
        max_xyz = verts_world.max(axis=0)
        body_width_lr_mm = float(max_xyz[0] - min_xyz[0])
        body_depth_ap_mm = float(max_xyz[1] - min_xyz[1])
        body_height_si_mm = float(max_xyz[2] - min_xyz[2])

        touch_x_lo = bool(np.any(body_mask[0, :, :]))
        touch_x_hi = bool(np.any(body_mask[-1, :, :]))
        touch_y_lo = bool(np.any(body_mask[:, 0, :]))
        touch_y_hi = bool(np.any(body_mask[:, -1, :]))
        touch_z_lo = bool(np.any(body_mask[:, :, 0]))
        touch_z_hi = bool(np.any(body_mask[:, :, -1]))
        touches_count = sum([touch_x_lo, touch_x_hi, touch_y_lo, touch_y_hi, touch_z_lo, touch_z_hi])

        if body_height_si_mm >= 350.0 and touches_count <= 2:
            fov_cat = "FOV-A"
        elif body_height_si_mm >= 200.0:
            fov_cat = "FOV-B"
        elif body_height_si_mm >= 100.0:
            fov_cat = "FOV-C"
        else:
            fov_cat = "FOV-D"

        fov_record = {
            "case_id": case_id,
            "modality": modality,
            "body_width_lr_mm": body_width_lr_mm,
            "body_depth_ap_mm": body_depth_ap_mm,
            "body_height_si_mm": body_height_si_mm,
            "touch_x_lo": touch_x_lo,
            "touch_x_hi": touch_x_hi,
            "touch_y_lo": touch_y_lo,
            "touch_y_hi": touch_y_hi,
            "touch_z_lo": touch_z_lo,
            "touch_z_hi": touch_z_hi,
            "touches_count": touches_count,
            "fov_category": fov_cat,
            "valid": True
        }

        body_center = (min_xyz + max_xyz) / 2.0
        verts_centered = verts_world - body_center

        mesh = trimesh.Trimesh(vertices=verts_centered, faces=faces, process=False)
        pts_4096, _ = trimesh.sample.sample_surface(mesh, count=NUM_POINTS, seed=42)
        pts_4096 = pts_4096.astype(np.float32)
        pts_norm = pts_4096 / S_GLOBAL_MM

        np.savez_compressed(
            npz_out,
            case_id=case_id,
            modality=modality,
            body_center_mm=body_center,
            body_extent_mm=np.array([body_width_lr_mm, body_depth_ap_mm, body_height_si_mm], dtype=np.float32),
            points_normalized=pts_norm,
            points_unnormalized=pts_4096,
            fov_category=fov_cat,
            affine_ras=affine_ras
        )

        gt_records = []
        if lbl_nii is not None:
            lbl_ras = nib.as_closest_canonical(lbl_nii)
            lbl_data = lbl_ras.get_fdata().astype(np.int32)
            affine_lbl = lbl_ras.affine

            for _, row in target_mapping.iterrows():
                amos_id = int(row["AMOS_label_id"])
                t_name = row["Phase10R_target_name"]
                t_idx = int(row["Phase10R_target_index"])
                vox_coords = np.argwhere(lbl_data == amos_id)
                if len(vox_coords) >= 10:
                    world_coords = nib.affines.apply_affine(affine_lbl, vox_coords)
                    c_world = np.mean(world_coords, axis=0)
                    c_centered = c_world - body_center
                    c_norm = c_centered / S_GLOBAL_MM
                    gt_records.append({
                        "case_id": case_id,
                        "modality": modality,
                        "target_name": t_name,
                        "target_index": t_idx,
                        "amos_label_id": amos_id,
                        "x_world_mm": float(c_world[0]),
                        "y_world_mm": float(c_world[1]),
                        "z_world_mm": float(c_world[2]),
                        "x_centered_mm": float(c_centered[0]),
                        "y_centered_mm": float(c_centered[1]),
                        "z_centered_mm": float(c_centered[2]),
                        "x_norm": float(c_norm[0]),
                        "y_norm": float(c_norm[1]),
                        "z_norm": float(c_norm[2]),
                        "voxel_count": len(vox_coords),
                        "valid": True
                    })

        if do_qc:
            qc_img_path = os.path.join(QC_DIR, f"{case_id}_qc.png")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            mid_z = img_data.shape[2] // 2
            mid_y = img_data.shape[1] // 2

            axes[0].imshow(np.rot90(img_data[:, :, mid_z]), cmap="gray")
            axes[0].contour(np.rot90(body_mask[:, :, mid_z]), levels=[0.5], colors="cyan", linewidths=1.5)
            axes[0].set_title(f"{case_id} MRI Axial (z={mid_z})")
            axes[0].axis("off")

            axes[1].imshow(np.rot90(img_data[:, mid_y, :]), cmap="gray")
            axes[1].contour(np.rot90(body_mask[:, mid_y, :]), levels=[0.5], colors="cyan", linewidths=1.5)
            axes[1].set_title(f"{case_id} MRI Coronal (y={mid_y})")
            axes[1].axis("off")

            axes[2].scatter(pts_4096[::4, 0], pts_4096[::4, 2], s=1, c=pts_4096[::4, 1], cmap="viridis", alpha=0.6)
            for r in gt_records:
                axes[2].scatter(r["x_centered_mm"], r["z_centered_mm"], s=25, c="red", marker="x")
            axes[2].set_title(f"{case_id} Centered Pts & GT Organs ({fov_cat})")
            axes[2].set_aspect("equal")
            axes[2].set_xlabel("LR (mm)")
            axes[2].set_ylabel("SI (mm)")

            plt.tight_layout()
            plt.savefig(qc_img_path, dpi=120)
            plt.close(fig)

        return case_id, gt_records, fov_record
    except Exception as e:
        print(f"Error processing {case_id}: {e}")
        return None, None, {"case_id": case_id, "valid": False, "reason": str(e)}

def main():
    print("=== AMOS MRI Batch Preprocessing (60 cases) ===")
    tasks = []
    qc_count = 0
    
    # Train MRI (40 cases)
    for f in sorted(glob.glob("data_external/AMOS22/extracted/imagesTr/amos_05*.nii.gz")):
        bname = os.path.basename(f)
        lbl_p = os.path.join("data_external/AMOS22/extracted/labelsTr", bname)
        if os.path.exists(lbl_p):
            cid = bname.split(".")[0]
            do_qc = (qc_count < 20)
            if do_qc:
                qc_count += 1
            tasks.append((f, lbl_p, cid, "MRI", do_qc))

    # Val MRI (20 cases)
    for f in sorted(glob.glob("data_external/AMOS22/extracted/imagesVa/amos_05*.nii.gz")):
        bname = os.path.basename(f)
        lbl_p = os.path.join("data_external/AMOS22/extracted/labelsVa", bname)
        if os.path.exists(lbl_p):
            cid = bname.split(".")[0]
            do_qc = (qc_count < 20)
            if do_qc:
                qc_count += 1
            tasks.append((f, lbl_p, cid, "MRI", do_qc))

    print(f"Discovered {len(tasks)} paired MRI cases.")
    t0 = time.time()
    all_gt = []
    all_fov = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_case_task, t): t[2] for t in tasks}
        completed = 0
        for f in concurrent.futures.as_completed(futures):
            cid, gts, fov = f.result()
            completed += 1
            if fov and fov.get("valid", False):
                all_fov.append(fov)
                if gts:
                    all_gt.extend(gts)
            if completed % 10 == 0 or completed == len(tasks):
                print(f"  Processed {completed}/{len(tasks)} MRI cases ({time.time()-t0:.1f}s elapsed)...")

    df_gt = pd.DataFrame(all_gt)
    df_gt.to_csv(GT_CENTROIDS_PATH, index=False)
    print(f"Saved {len(df_gt)} MRI GT records to {GT_CENTROIDS_PATH}")

    df_fov = pd.DataFrame(all_fov)
    df_fov.to_csv(FOV_AUDIT_PATH, index=False)
    print(f"Saved {len(df_fov)} MRI FOV records to {FOV_AUDIT_PATH}")
    print("MRI Batch Preprocessing Complete!")

if __name__ == "__main__":
    main()
