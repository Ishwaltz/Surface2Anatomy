#!/usr/bin/env python3
"""
tools/phase14_ctorg/01_orientation_qc.py
========================================
Section 7: Orientation / Left-Right Audit for CT-ORG
Generates reports/phase14_ctorg/03_orientation_qc.csv
"""

import sys
import csv
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import nibabel as nib

repo_root = Path(__file__).resolve().parent.parent.parent
vols_dir = Path("/home/sharon/Datasets/CT_ORG/extracted/volumes")
lbls_dir = Path("/home/sharon/Datasets/CT_ORG/extracted/labels")
out_dir = repo_root / "reports" / "phase14_ctorg"
out_dir.mkdir(parents=True, exist_ok=True)

def run_orientation_qc():
    print("=" * 80)
    print("PHASE 14 CT-ORG: ORIENTATION / LEFT-RIGHT AUDIT")
    print("=" * 80)
    
    rows = []
    ambiguous_count = 0
    valid_kidneys_count = 0
    
    for case_idx in range(140):
        case_id = f"volume-{case_idx}"
        ct_file = vols_dir / f"volume-{case_idx}.nii.gz"
        lbl_file = lbls_dir / f"labels-{case_idx}.nii.gz"
        
        if not ct_file.exists() or not lbl_file.exists():
            rows.append({
                "case_id": case_id,
                "original_axcodes": "MISSING",
                "canonical_axcodes": "MISSING",
                "affine_determinant": 0.0,
                "left_right_flip_applied": "NO",
                "orientation_status": "MISSING_FILE",
                "kidney_components": 0,
                "kidney_lr_resolved": "NO"
            })
            continue
            
        try:
            ct_nii = nib.load(ct_file)
            lbl_nii = nib.load(lbl_file)
            
            orig_affine = ct_nii.affine
            orig_axcodes = "".join(nib.aff2axcodes(orig_affine))
            det = float(np.linalg.det(orig_affine[:3, :3]))
            
            # Canonicalize to RAS
            ct_canon = nib.as_closest_canonical(ct_nii)
            lbl_canon = nib.as_closest_canonical(lbl_nii)
            canon_axcodes = "".join(nib.aff2axcodes(ct_canon.affine))
            
            # Check if left-right flip was applied
            lr_flip = "YES" if (len(orig_axcodes) >= 1 and orig_axcodes[0] == 'L') else "NO"
            
            # Verify volume and label shapes match
            if ct_canon.shape != lbl_canon.shape:
                orientation_status = "SHAPE_MISMATCH"
                ambiguous_count += 1
            elif np.isnan(det) or np.isinf(det) or abs(det) < 1e-6:
                orientation_status = "INVALID_AFFINE"
                ambiguous_count += 1
            else:
                orientation_status = "PASS"
                
            # Kidney component analysis in canonical RAS (using rounded integer labels)
            lbl_data = np.round(np.asanyarray(lbl_canon.dataobj)).astype(np.int32)
            kidney_mask = (lbl_data == 4)
            kidney_vox_count = int(np.sum(kidney_mask))
            
            if kidney_vox_count >= 10:
                labeled_k, num_k = ndi.label(kidney_mask)
                counts = np.bincount(labeled_k.ravel())
                sorted_indices = np.argsort(counts[1:])[::-1] + 1
                
                if num_k >= 2:
                    top2_vol = counts[sorted_indices[0]] + counts[sorted_indices[1]]
                    if top2_vol / kidney_vox_count >= 0.95:
                        kidney_components = 2
                        kidney_lr_resolved = "YES"
                        valid_kidneys_count += 1
                    else:
                        kidney_components = num_k
                        kidney_lr_resolved = "NO"
                elif num_k == 1:
                    kidney_components = 1
                    kidney_lr_resolved = "NO"
                else:
                    kidney_components = 0
                    kidney_lr_resolved = "NO"
            else:
                kidney_components = 0
                kidney_lr_resolved = "NOT_APPLICABLE"
                
        except Exception as e:
            orientation_status = f"CORRUPT_{str(e)[:20]}"
            ambiguous_count += 1
            orig_axcodes = "ERROR"
            canon_axcodes = "ERROR"
            det = 0.0
            lr_flip = "NO"
            kidney_components = 0
            kidney_lr_resolved = "NO"
            
        rows.append({
            "case_id": case_id,
            "original_axcodes": orig_axcodes,
            "canonical_axcodes": canon_axcodes,
            "affine_determinant": round(det, 4),
            "left_right_flip_applied": lr_flip,
            "orientation_status": orientation_status,
            "kidney_components": kidney_components,
            "kidney_lr_resolved": kidney_lr_resolved
        })
        
    out_file = out_dir / "03_orientation_qc.csv"
    fieldnames = [
        "case_id", "original_axcodes", "canonical_axcodes", "affine_determinant",
        "left_right_flip_applied", "orientation_status", "kidney_components", "kidney_lr_resolved"
    ]
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Orientation QC complete: {out_file}")
    print(f"  Ambiguous cases: {ambiguous_count}")
    print(f"  Valid bilateral kidney cases: {valid_kidneys_count}")
    return rows

if __name__ == "__main__":
    run_orientation_qc()
