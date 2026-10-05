#!/usr/bin/env python3
"""
tools/phase14_ctorg/03_eligibility_and_preprocessing.py
======================================================
Sections 9-15: Case Eligibility, Surface Extraction, Canonical Alignment, and GT Targets.
Outputs:
- reports/phase14_ctorg/05_case_eligibility_PREINFERENCE.csv
- reports/phase14_ctorg/06_geometry_qc.csv
- external_validation/CT_ORG/processed_surfaces/{case_id}.npz
"""

import os
import sys
import csv
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import trimesh
import nibabel as nib
import joblib

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from tools.dataset_v3.apply_canonical_alignment import extract_alignment_features

vols_dir = Path("/home/sharon/Datasets/CT_ORG/extracted/volumes")
lbls_dir = Path("/home/sharon/Datasets/CT_ORG/extracted/labels")
reports_dir = repo_root / "reports" / "phase14_ctorg"
processed_surfaces_dir = repo_root / "external_validation/CT_ORG/processed_surfaces"

reports_dir.mkdir(parents=True, exist_ok=True)
processed_surfaces_dir.mkdir(parents=True, exist_ok=True)

# Load frozen Ridge model
ridge_path = repo_root / "experiments/phase10R/checkpoints/canonical_alignment_v3_ridge.joblib"
if not ridge_path.exists():
    raise FileNotFoundError(f"Frozen Ridge model not found at {ridge_path}")
align_model = joblib.load(ridge_path)

def extract_body_surface_mesh(ct_canon):
    """
    Extracts external body surface mesh using CT voxel intensities alone.
    HU > -300, largest connected component, 2D axial hole-filling.
    """
    ct_data = np.asanyarray(ct_canon.dataobj, dtype=np.float32)
    foreground = ct_data > -300.0
    
    labeled, num_cc = ndi.label(foreground)
    if num_cc == 0:
        return None, None, "EMPTY_FOREGROUND"
        
    counts = np.bincount(labeled.ravel())
    largest_label = np.argmax(counts[1:]) + 1
    body_mask = (labeled == largest_label)
    
    # 2D hole filling slice-by-slice along axial axis (Z-axis in RAS)
    for z in range(body_mask.shape[2]):
        if np.any(body_mask[:, :, z]):
            body_mask[:, :, z] = ndi.binary_fill_holes(body_mask[:, :, z])
            
    # Marching cubes with step=2 for robust, fast, high-density surface
    step = 2
    try:
        verts_vox, faces, _, _ = measure.marching_cubes(body_mask.astype(float), level=0.5, step_size=step)
    except Exception as e:
        return None, None, f"MARCHING_CUBES_FAIL_{str(e)[:20]}"
        
    # Voxel to metric world mm using canonical affine
    verts_world = nib.affines.apply_affine(ct_canon.affine, verts_vox)
    return verts_world, faces, "SUCCESS"

def sample_4096_points(verts_world, faces, seed=42):
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    pts_4096, _ = trimesh.sample.sample_surface(mesh, count=4096, seed=seed)
    return pts_4096.astype(np.float32)

def extract_ground_truth_targets(lbl_canon):
    """
    Computes ground-truth internal centroids strictly from label segmentation in canonical RAS.
    Returns: dict[target_slot -> (3,) np.ndarray]
    Slots:
    - 4: liver
    - 20: urinary_bladder
    - 89: brain
    - 1: kidney_right
    - 2: kidney_left
    """
    # Round floating-point labels to exact integers
    lbl_data = np.round(np.asanyarray(lbl_canon.dataobj)).astype(np.int32)
    affine = lbl_canon.affine
    targets = {}
    
    # 1. Liver (CT-ORG ID 1 -> Slot 4)
    liver_vox = np.argwhere(lbl_data == 1)
    if len(liver_vox) >= 10:
        targets[4] = np.mean(nib.affines.apply_affine(affine, liver_vox), axis=0).astype(np.float32)
        
    # 2. Bladder (CT-ORG ID 2 -> Slot 20)
    bladder_vox = np.argwhere(lbl_data == 2)
    if len(bladder_vox) >= 10:
        targets[20] = np.mean(nib.affines.apply_affine(affine, bladder_vox), axis=0).astype(np.float32)
        
    # 3. Brain (CT-ORG ID 6 -> Slot 89)
    brain_vox = np.argwhere(lbl_data == 6)
    if len(brain_vox) >= 10:
        targets[89] = np.mean(nib.affines.apply_affine(affine, brain_vox), axis=0).astype(np.float32)
        
    # 4. Kidneys (CT-ORG ID 4 -> Slots 1, 2)
    kidney_mask = (lbl_data == 4)
    k_vox_count = int(np.sum(kidney_mask))
    if k_vox_count >= 10:
        labeled_k, num_k = ndi.label(kidney_mask)
        counts = np.bincount(labeled_k.ravel())
        sorted_indices = np.argsort(counts[1:])[::-1] + 1
        if num_k >= 2:
            top2_vol = counts[sorted_indices[0]] + counts[sorted_indices[1]]
            if top2_vol / k_vox_count >= 0.95:
                c1_vox = np.argwhere(labeled_k == sorted_indices[0])
                c2_vox = np.argwhere(labeled_k == sorted_indices[1])
                c1_w = np.mean(nib.affines.apply_affine(affine, c1_vox), axis=0).astype(np.float32)
                c2_w = np.mean(nib.affines.apply_affine(affine, c2_vox), axis=0).astype(np.float32)
                
                # In canonical RAS: +X is Patient Right, -X is Patient Left
                if c1_w[0] > c2_w[0]:
                    targets[1] = c1_w  # kidney_right
                    targets[2] = c2_w  # kidney_left
                else:
                    targets[1] = c2_w  # kidney_right
                    targets[2] = c1_w  # kidney_left
                    
    return targets

def run_preprocessing():
    print("=" * 80)
    print("PHASE 14 CT-ORG: CASE ELIGIBILITY, SURFACE EXTRACTION & ALIGNMENT")
    print("=" * 80)
    
    # Load orientation QC and duplicate audit
    qc_file = reports_dir / "03_orientation_qc.csv"
    dup_file = reports_dir / "04_duplicate_audit.csv"
    
    qc_dict = {}
    if qc_file.exists():
        with open(qc_file) as f:
            for r in csv.DictReader(f):
                qc_dict[r["case_id"]] = r
                
    dup_dict = {}
    if dup_file.exists():
        with open(dup_file) as f:
            for r in csv.DictReader(f):
                dup_dict[r["case_id"]] = r

def process_single_case(case_idx, qc_dict, dup_dict):
    case_id = f"volume-{case_idx}"
    ct_file = vols_dir / f"volume-{case_idx}.nii.gz"
    lbl_file = lbls_dir / f"labels-{case_idx}.nii.gz"
    
    ct_exists = ct_file.exists()
    lbl_exists = lbl_file.exists()
    
    if not ct_exists or not lbl_exists:
        reason = "MISSING_CT" if not ct_exists else "MISSING_LABEL"
        el_row = {
            "case_id": case_id, "ct_exists": "YES" if ct_exists else "NO",
            "label_exists": "YES" if lbl_exists else "NO", "nifti_valid": "NO",
            "affine_valid": "NO", "orientation_resolved": "NO", "surface_extractable": "NO",
            "surface_points_ge_4096": "NO", "num_overlapping_targets": 0,
            "is_duplicate": "NO", "is_possible_overlap": "NO", "eligible": "NO",
            "exclusion_reason": reason
        }
        return case_idx, el_row, None, None, 0.0
        
    # Duplicate status
    d_info = dup_dict.get(case_id, {})
    is_dup = "YES" if d_info.get("classification") == "CONFIRMED_DUPLICATE" else "NO"
    is_overlap = "YES" if d_info.get("classification") == "POSSIBLE_OVERLAP" else "NO"
    
    if is_dup == "YES":
        el_row = {
            "case_id": case_id, "ct_exists": "YES", "label_exists": "YES", "nifti_valid": "YES",
            "affine_valid": "YES", "orientation_resolved": "YES", "surface_extractable": "N/A",
            "surface_points_ge_4096": "N/A", "num_overlapping_targets": 0,
            "is_duplicate": "YES", "is_possible_overlap": "NO", "eligible": "NO",
            "exclusion_reason": "CONFIRMED_DUPLICATE"
        }
        return case_idx, el_row, None, None, 0.0
        
    # Orientation status
    q_info = qc_dict.get(case_id, {})
    orient_stat = q_info.get("orientation_status", "PASS")
    if orient_stat != "PASS":
        el_row = {
            "case_id": case_id, "ct_exists": "YES", "label_exists": "YES", "nifti_valid": "NO",
            "affine_valid": "NO", "orientation_resolved": "NO", "surface_extractable": "N/A",
            "surface_points_ge_4096": "N/A", "num_overlapping_targets": 0,
            "is_duplicate": "NO", "is_possible_overlap": "NO", "eligible": "NO",
            "exclusion_reason": "ORIENTATION_AMBIGUOUS"
        }
        return case_idx, el_row, None, None, 0.0
        
    try:
        t_case = time.time()
        ct_nii = nib.load(ct_file)
        lbl_nii = nib.load(lbl_file)
        ct_canon = nib.as_closest_canonical(ct_nii)
        lbl_canon = nib.as_closest_canonical(lbl_nii)
        
        # Ground-truth centroids
        gt_targets = extract_ground_truth_targets(lbl_canon)
        num_targets = len(gt_targets)
        
        if num_targets == 0:
            el_row = {
                "case_id": case_id, "ct_exists": "YES", "label_exists": "YES", "nifti_valid": "YES",
                "affine_valid": "YES", "orientation_resolved": "YES", "surface_extractable": "YES",
                "surface_points_ge_4096": "YES", "num_overlapping_targets": 0,
                "is_duplicate": "NO", "is_possible_overlap": "NO", "eligible": "NO",
                "exclusion_reason": "NO_OVERLAPPING_TARGETS"
            }
            return case_idx, el_row, None, None, 0.0
            
        # Surface extraction
        verts_world, faces, surf_status = extract_body_surface_mesh(ct_canon)
        if surf_status != "SUCCESS" or len(verts_world) < 4096:
            el_row = {
                "case_id": case_id, "ct_exists": "YES", "label_exists": "YES", "nifti_valid": "YES",
                "affine_valid": "YES", "orientation_resolved": "YES", "surface_extractable": "NO",
                "surface_points_ge_4096": "NO", "num_overlapping_targets": num_targets,
                "is_duplicate": "NO", "is_possible_overlap": "NO", "eligible": "NO",
                "exclusion_reason": "INSUFFICIENT_SURFACE"
            }
            return case_idx, el_row, None, None, 0.0
            
        # Roundtrip verification: world -> voxel -> world
        aff_inv = np.linalg.inv(ct_canon.affine)
        verts_vox_rt = nib.affines.apply_affine(aff_inv, verts_world)
        verts_world_rt = nib.affines.apply_affine(ct_canon.affine, verts_vox_rt)
        max_rt_err = float(np.max(np.linalg.norm(verts_world - verts_world_rt, axis=1)))
        
        # Deterministic surface sampling (4096 points, seed 42)
        pts_4096 = sample_4096_points(verts_world, faces, seed=42)
        
        # Bounding box & geometry features
        bbox_min = np.min(verts_world, axis=0)
        bbox_max = np.max(verts_world, axis=0)
        body_dims = bbox_max - bbox_min
        mid_raw = 0.5 * (bbox_min + bbox_max)
        surf_centroid = np.mean(verts_world, axis=0)
        
        # Canonical alignment using frozen Ridge model
        f_vec, m_vec = extract_alignment_features(pts_4096)
        pred_offset = align_model.predict(f_vec.reshape(1, -1))[0].astype(np.float32)
        c_external = (m_vec + pred_offset).astype(np.float32)
        
        # Centered & normalized surface
        pts_centered = pts_4096 - c_external
        pts_norm = pts_centered / 500.0
        
        el_row = {
            "case_id": case_id, "ct_exists": "YES", "label_exists": "YES", "nifti_valid": "YES",
            "affine_valid": "YES", "orientation_resolved": "YES", "surface_extractable": "YES",
            "surface_points_ge_4096": "YES", "num_overlapping_targets": num_targets,
            "is_duplicate": "NO", "is_possible_overlap": "NO", "eligible": "YES",
            "exclusion_reason": "NONE"
        }
        
        geo_row = {
            "case_id": case_id,
            "voxel_world_roundtrip_max_mm": round(max_rt_err, 8),
            "body_width_mm": round(float(body_dims[0]), 2),
            "body_depth_mm": round(float(body_dims[1]), 2),
            "visible_body_height_mm": round(float(body_dims[2]), 2),
            "surface_point_count": 4096,
            "bbox_mid_x": round(float(m_vec[0]), 2),
            "bbox_mid_y": round(float(m_vec[1]), 2),
            "bbox_mid_z": round(float(m_vec[2]), 2),
            "surface_centroid_x": round(float(surf_centroid[0]), 2),
            "surface_centroid_y": round(float(surf_centroid[1]), 2),
            "surface_centroid_z": round(float(surf_centroid[2]), 2),
            "canonical_offset_x": round(float(pred_offset[0]), 2),
            "canonical_offset_y": round(float(pred_offset[1]), 2),
            "canonical_offset_z": round(float(pred_offset[2]), 2),
            "c_external_x": round(float(c_external[0]), 2),
            "c_external_y": round(float(c_external[1]), 2),
            "c_external_z": round(float(c_external[2]), 2),
        }
        
        # Save preprocessed surface package
        np.savez_compressed(
            processed_surfaces_dir / f"{case_id}.npz",
            points_4096=pts_4096,
            points_norm=pts_norm,
            c_external=c_external,
            pred_offset=pred_offset,
            bbox_mid=m_vec,
            gt_slots=np.array(list(gt_targets.keys()), dtype=np.int32),
            gt_coords=np.array(list(gt_targets.values()), dtype=np.float32)
        )
        print(f"[{case_idx+1}/140] {case_id} processed in {time.time()-t_case:.2f}s (targets: {num_targets})", flush=True)
        return case_idx, el_row, geo_row, pred_offset, max_rt_err
        
    except Exception as e:
        el_row = {
            "case_id": case_id, "ct_exists": "YES", "label_exists": "YES", "nifti_valid": "NO",
            "affine_valid": "NO", "orientation_resolved": "NO", "surface_extractable": "NO",
            "surface_points_ge_4096": "NO", "num_overlapping_targets": 0,
            "is_duplicate": "NO", "is_possible_overlap": "NO", "eligible": "NO",
            "exclusion_reason": f"CORRUPT_{str(e)[:20]}"
        }
        return case_idx, el_row, None, None, 0.0

def run_preprocessing():
    print("=" * 80)
    print("PHASE 14 CT-ORG: CASE ELIGIBILITY, SURFACE EXTRACTION & ALIGNMENT (PARALLEL)")
    print("=" * 80)
    
    # Load orientation QC and duplicate audit
    qc_file = reports_dir / "03_orientation_qc.csv"
    dup_file = reports_dir / "04_duplicate_audit.csv"
    
    qc_dict = {}
    if qc_file.exists():
        with open(qc_file) as f:
            for r in csv.DictReader(f):
                qc_dict[r["case_id"]] = r
                
    dup_dict = {}
    if dup_file.exists():
        with open(dup_file) as f:
            for r in csv.DictReader(f):
                dup_dict[r["case_id"]] = r

    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    t0_all = time.time()
    num_workers = 8
    print(f"Launching {num_workers} parallel workers for 140 CT-ORG volumes...", flush=True)
    
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_case, idx, qc_dict, dup_dict): idx for idx in range(140)}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            
    # Sort by case_idx strictly
    results.sort(key=lambda x: x[0])
    
    eligibility_rows = []
    geometry_rows = []
    eligible_count = 0
    excluded_count = 0
    all_offset_x, all_offset_y, all_offset_z = [], [], []
    max_roundtrip_global = 0.0
    
    for case_idx, el_row, geo_row, pred_offset, max_rt_err in results:
        eligibility_rows.append(el_row)
        if el_row["eligible"] == "YES":
            eligible_count += 1
            geometry_rows.append(geo_row)
            all_offset_x.append(pred_offset[0])
            all_offset_y.append(pred_offset[1])
            all_offset_z.append(pred_offset[2])
            max_roundtrip_global = max(max_roundtrip_global, max_rt_err)
        else:
            excluded_count += 1
            
    # Save 05_case_eligibility_PREINFERENCE.csv
    el_file = reports_dir / "05_case_eligibility_PREINFERENCE.csv"
    el_fields = [
        "case_id", "ct_exists", "label_exists", "nifti_valid", "affine_valid",
        "orientation_resolved", "surface_extractable", "surface_points_ge_4096",
        "num_overlapping_targets", "is_duplicate", "is_possible_overlap", "eligible",
        "exclusion_reason"
    ]
    with open(el_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=el_fields)
        writer.writeheader()
        writer.writerows(eligibility_rows)
        
    # SHA256 of eligibility file
    el_sha = hashlib.sha256(el_file.read_bytes()).hexdigest()
    with open(reports_dir / "05_case_eligibility_PREINFERENCE.sha256", "w") as f:
        f.write(f"{el_sha}  05_case_eligibility_PREINFERENCE.csv\n")
        
    # Save 06_geometry_qc.csv
    geo_file = reports_dir / "06_geometry_qc.csv"
    geo_fields = [
        "case_id", "voxel_world_roundtrip_max_mm", "body_width_mm", "body_depth_mm",
        "visible_body_height_mm", "surface_point_count", "bbox_mid_x", "bbox_mid_y", "bbox_mid_z",
        "surface_centroid_x", "surface_centroid_y", "surface_centroid_z",
        "canonical_offset_x", "canonical_offset_y", "canonical_offset_z",
        "c_external_x", "c_external_y", "c_external_z"
    ]
    with open(geo_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=geo_fields)
        writer.writeheader()
        writer.writerows(geometry_rows)
        
    print("\n" + "=" * 80)
    print("PREPROCESSING & ELIGIBILITY AUDIT COMPLETE")
    print("=" * 80)
    print(f"Eligible cases: {eligible_count}")
    print(f"Excluded cases: {excluded_count}")
    print(f"Max numerical roundtrip error: {max_roundtrip_global:.8f} mm (PASS < 0.001 mm: {max_roundtrip_global < 0.001})")
    print(f"Mean canonical offset: X={np.mean(all_offset_x):.2f} mm, Y={np.mean(all_offset_y):.2f} mm, Z={np.mean(all_offset_z):.2f} mm")
    print(f"Eligibility file SHA256: {el_sha}")

if __name__ == "__main__":
    run_preprocessing()
