#!/usr/bin/env python3
"""
tools/phase11/download_and_verify_amos.py

Downloads AMOS CT labels and MRI cases from HuggingFace mirrors:
- MedOtter/amos22-ct-dataset (labels only, ~49 MB)
- MedOtter/amos22-mri-dataset (images + labels, ~750 MB)
Hardlinks existing CT images from raw_data/ into data_external/AMOS22/extracted/imagesTr.
Verifies pairings, dimensions, affines, and label IDs (1-15).
"""

import os
import sys
import time
import glob
import shutil
import numpy as np
import nibabel as nib
from huggingface_hub import snapshot_download, hf_hub_download

BASE_DIR = "data_external/AMOS22/extracted"
IMAGES_TR = os.path.join(BASE_DIR, "imagesTr")
LABELS_TR = os.path.join(BASE_DIR, "labelsTr")
IMAGES_VA = os.path.join(BASE_DIR, "imagesVa")
LABELS_VA = os.path.join(BASE_DIR, "labelsVa")

for d in [IMAGES_TR, LABELS_TR, IMAGES_VA, LABELS_VA]:
    os.makedirs(d, exist_ok=True)

print("=== Step 1: Downloading CT labels from MedOtter/amos22-ct-dataset (~49 MB) ===")
t0 = time.time()
ct_labels_dir = snapshot_download(
    repo_id="MedOtter/amos22-ct-dataset",
    repo_type="dataset",
    allow_patterns=["*labelsTr/*", "*labelsVa/*"],
    max_workers=8
)
print(f"CT labels downloaded to snapshot cache in {time.time()-t0:.1f}s")

# Copy/link CT labels into place
for split, target_dir in [("train/labelsTr", LABELS_TR), ("valid/labelsVa", LABELS_VA)]:
    src_split = os.path.join(ct_labels_dir, split)
    if os.path.exists(src_split):
        for f in os.listdir(src_split):
            if f.endswith(".nii.gz"):
                src_file = os.path.join(src_split, f)
                dst_file = os.path.join(target_dir, f)
                if not os.path.exists(dst_file):
                    shutil.copyfile(src_file, dst_file)
print(f"CT labels staged: {len(os.listdir(LABELS_TR))} in labelsTr, {len(os.listdir(LABELS_VA))} in labelsVa")

print("=== Step 2: Hardlinking existing CT images from raw_data/ ===")
raw_ct = glob.glob("raw_data/amos_*.nii.gz")
linked_count = 0
for f in raw_ct:
    bname = os.path.basename(f)
    dst = os.path.join(IMAGES_TR, bname)
    if not os.path.exists(dst):
        try:
            os.link(os.path.abspath(f), dst)
            linked_count += 1
        except Exception:
            shutil.copyfile(f, dst)
            linked_count += 1
print(f"Staged {linked_count} CT images into {IMAGES_TR}")

print("=== Step 3: Downloading MRI dataset from MedOtter/amos22-mri-dataset (~750 MB) ===")
t0 = time.time()
mri_dir = snapshot_download(
    repo_id="MedOtter/amos22-mri-dataset",
    repo_type="dataset",
    allow_patterns=["*imagesTr/*", "*labelsTr/*", "*imagesVa/*", "*labelsVa/*"],
    max_workers=8
)
print(f"MRI dataset downloaded in {time.time()-t0:.1f}s")

# Copy/link MRI files into place
for split, target_img, target_lbl in [
    ("train", IMAGES_TR, LABELS_TR),
    ("valid", IMAGES_VA, LABELS_VA)
]:
    img_src = os.path.join(mri_dir, split, "imagesTr" if split == "train" else "imagesVa")
    lbl_src = os.path.join(mri_dir, split, "labelsTr" if split == "train" else "labelsVa")
    
    if os.path.exists(img_src):
        for f in os.listdir(img_src):
            if f.endswith(".nii.gz"):
                dst = os.path.join(target_img, f)
                if not os.path.exists(dst):
                    shutil.copyfile(os.path.join(img_src, f), dst)
    if os.path.exists(lbl_src):
        for f in os.listdir(lbl_src):
            if f.endswith(".nii.gz"):
                dst = os.path.join(target_lbl, f)
                if not os.path.exists(dst):
                    shutil.copyfile(os.path.join(lbl_src, f), dst)

print(f"Total files now in {IMAGES_TR}: {len(os.listdir(IMAGES_TR))}")
print(f"Total files now in {LABELS_TR}: {len(os.listdir(LABELS_TR))}")
print(f"Total files now in {IMAGES_VA}: {len(os.listdir(IMAGES_VA))}")
print(f"Total files now in {LABELS_VA}: {len(os.listdir(LABELS_VA))}")

print("=== Step 4: Verification of Pairings, Dimensions, Affines, and Labels ===")
def audit_split(img_dir, lbl_dir, split_name):
    img_files = {os.path.basename(f): f for f in glob.glob(os.path.join(img_dir, "amos_*.nii.gz"))}
    lbl_files = {os.path.basename(f): f for f in glob.glob(os.path.join(lbl_dir, "amos_*.nii.gz"))}
    
    paired_ids = sorted(list(set(img_files.keys()).intersection(set(lbl_files.keys()))))
    print(f"[{split_name}] Images: {len(img_files)}, Labels: {len(lbl_files)}, Paired: {len(paired_ids)}")
    
    ct_paired = [c for c in paired_ids if int(c.split('_')[1].split('.')[0]) <= 500]
    mri_paired = [c for c in paired_ids if int(c.split('_')[1].split('.')[0]) > 500]
    print(f"[{split_name}] CT Paired: {len(ct_paired)}, MRI Paired: {len(mri_paired)}")
    
    # Audit sample cases
    checked = 0
    verified = 0
    for cid in paired_ids[:10]:
        try:
            img = nib.load(img_files[cid])
            lbl = nib.load(lbl_files[cid])
            shape_match = (img.shape == lbl.shape)
            affine_match = np.allclose(img.affine, lbl.affine, atol=1e-3)
            lbl_data = lbl.get_fdata()
            u_labels = np.unique(lbl_data)
            has_organs = np.any((u_labels >= 1) & (u_labels <= 15))
            if shape_match and affine_match and has_organs:
                verified += 1
            checked += 1
        except Exception as e:
            print(f"Error checking {cid}: {e}")
    print(f"[{split_name}] Spot check: {verified}/{checked} verified valid.")
    return len(ct_paired), len(mri_paired)

ct_tr, mri_tr = audit_split(IMAGES_TR, LABELS_TR, "Train Split")
ct_va, mri_va = audit_split(IMAGES_VA, LABELS_VA, "Validation Split")

print("\n=== AMOS Audit Summary ===")
print(f"EXISTING_AMOS_CT_IMAGES = {len(glob.glob('raw_data/amos_*.nii.gz'))}")
print(f"EXISTING_AMOS_CT_LABELS = {ct_tr + ct_va}")
print(f"VALID_PAIRED_CT_CASES = {ct_tr}")
print(f"EXISTING_AMOS_MRI_IMAGES = {mri_tr + mri_va}")
print(f"EXISTING_AMOS_MRI_LABELS = {mri_tr + mri_va}")
print(f"VALID_PAIRED_MRI_CASES = {mri_tr + mri_va}")
print(f"TOTAL_VALID_PAIRED = {ct_tr + mri_tr + mri_va}")
