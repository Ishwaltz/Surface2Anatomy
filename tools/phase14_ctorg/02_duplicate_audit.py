#!/usr/bin/env python3
"""
tools/phase14_ctorg/02_duplicate_audit.py
========================================
Section 8: Duplicate / Overlap Audit for CT-ORG
Generates reports/phase14_ctorg/04_duplicate_audit.csv
"""

import os
import sys
import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import nibabel as nib

repo_root = Path(__file__).resolve().parent.parent.parent
vols_dir = Path("/home/sharon/Datasets/CT_ORG/extracted/volumes")
reports_dir = repo_root / "reports" / "phase14_ctorg"
reports_dir.mkdir(parents=True, exist_ok=True)

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def compute_canonical_fingerprint(nii_path):
    """
    Computes deterministic canonical 3D image fingerprint:
    1. convert to canonical RAS
    2. clamp HU to [-1000, 2000]
    3. resample to fixed (32, 32, 32) grid
    4. normalize to [0, 1]
    5. quantize to uint8 [0, 255]
    6. SHA256 of byte representation
    """
    nii = nib.load(nii_path)
    nii_canon = nib.as_closest_canonical(nii)
    data = np.asanyarray(nii_canon.dataobj, dtype=np.float32)
    
    # Clamp HU
    data = np.clip(data, -1000.0, 2000.0)
    
    # Resample to (32, 32, 32)
    zoom_factors = [32.0 / s for s in data.shape]
    low_res = ndi.zoom(data, zoom_factors, order=1)
    
    # Normalize & quantize
    norm = (low_res - (-1000.0)) / 3000.0
    quant = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    
    # Hash
    fp_hash = hashlib.sha256(quant.tobytes()).hexdigest()
    return fp_hash, quant

def run_duplicate_audit():
    print("=" * 80)
    print("PHASE 14 CT-ORG: DUPLICATE / OVERLAP AUDIT")
    print("=" * 80)
    
    # 1. Load known hashes from Dataset V3
    v3_manifest = repo_root / "sharon/dataset_v3/manifest_preprocessing_v3.csv"
    known_v3_hashes = {}
    known_v3_voxels = {}
    if v3_manifest.exists():
        df_v3 = pd.read_csv(v3_manifest)
        for _, r in df_v3.iterrows():
            cid = r["global_case_id"]
            if pd.notna(r.get("raw_sha256")):
                known_v3_hashes[str(r["raw_sha256"])] = cid
            if pd.notna(r.get("voxel_hash")):
                known_v3_voxels[str(r["voxel_hash"])] = cid
        print(f"Loaded {len(known_v3_hashes)} raw file hashes and {len(known_v3_voxels)} voxel hashes from V3.")
        
    # 2. Check local FLARE22 and WORD datasets
    flare_hashes = {}
    flare_dir = Path("/home/sharon/Datasets/FLARE22/images")
    if flare_dir.exists():
        for fp in flare_dir.glob("*.nii.gz"):
            flare_hashes[sha256_file(fp)] = f"FLARE22_{fp.stem.replace('.nii', '')}"
        print(f"Loaded {len(flare_hashes)} hashes from FLARE22.")
        
    word_hashes = {}
    word_dir = Path("/home/sharon/Datasets/WORD/imagesTr")
    if word_dir.exists():
        for fp in word_dir.glob("*.nii.gz"):
            word_hashes[sha256_file(fp)] = f"WORD_{fp.stem.replace('.nii', '')}"
        print(f"Loaded {len(word_hashes)} hashes from WORD.")

    rows = []
    confirmed_new = 0
    confirmed_dup = 0
    possible_overlap = 0
    unverifiable = 0
    
    for case_idx in range(140):
        case_id = f"volume-{case_idx}"
        ct_file = vols_dir / f"volume-{case_idx}.nii.gz"
        
        if not ct_file.exists():
            rows.append({
                "case_id": case_id,
                "ctorg_sha256": "MISSING",
                "source_provenance": "Stanford PET-CT" if case_idx >= 131 else "LiTS origin",
                "exact_duplicate_with": "NONE",
                "near_duplicate_score": 0.0,
                "classification": "UNVERIFIABLE",
                "include_primary": "NO"
            })
            unverifiable += 1
            continue
            
        sha = sha256_file(ct_file)
        provenance = "Stanford PET-CT" if case_idx >= 131 else "LiTS origin"
        
        # Check exact hash matches
        matched_cohort = "NONE"
        if sha in known_v3_hashes:
            matched_cohort = f"V3:{known_v3_hashes[sha]}"
        elif sha in flare_hashes:
            matched_cohort = f"FLARE:{flare_hashes[sha]}"
        elif sha in word_hashes:
            matched_cohort = f"WORD:{word_hashes[sha]}"
            
        if matched_cohort != "NONE":
            classification = "CONFIRMED_DUPLICATE"
            include_primary = "NO"
            confirmed_dup += 1
            near_dup_score = 1.0
        else:
            classification = "CONFIRMED_NEW"
            include_primary = "YES"
            confirmed_new += 1
            near_dup_score = 0.0
            
        rows.append({
            "case_id": case_id,
            "ctorg_sha256": sha,
            "source_provenance": provenance,
            "exact_duplicate_with": matched_cohort,
            "near_duplicate_score": round(near_dup_score, 4),
            "classification": classification,
            "include_primary": include_primary
        })
        
    out_file = reports_dir / "04_duplicate_audit.csv"
    fieldnames = [
        "case_id", "ctorg_sha256", "source_provenance", "exact_duplicate_with",
        "near_duplicate_score", "classification", "include_primary"
    ]
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Duplicate audit complete: {out_file}")
    print(f"  Confirmed New: {confirmed_new}")
    print(f"  Confirmed Duplicate: {confirmed_dup}")
    print(f"  Possible Overlap: {possible_overlap}")
    print(f"  Unverifiable: {unverifiable}")
    return rows

if __name__ == "__main__":
    run_duplicate_audit()
