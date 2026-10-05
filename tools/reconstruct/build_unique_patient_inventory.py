import sys
import json
import csv
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

def compute_hashes_for_case(case_dir):
    ct_path = case_dir / "ct.nii.gz"
    if not ct_path.exists():
        return None
    
    # 1. File SHA256
    with open(ct_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    # 2. Canonical Voxel Hash (array data + shape + affine rounded)
    img = nib.load(str(ct_path))
    data = img.get_fdata(dtype=np.float32)
    shape = img.shape
    affine = np.round(img.affine, 4)
    
    voxel_hasher = hashlib.sha256()
    voxel_hasher.update(data.tobytes())
    voxel_hasher.update(str(shape).encode())
    voxel_hasher.update(affine.tobytes())
    voxel_hash = voxel_hasher.hexdigest()

    affine_hasher = hashlib.sha256()
    affine_hasher.update(affine.tobytes())
    affine_hash = affine_hasher.hexdigest()

    return file_hash, voxel_hash, shape, affine_hash

def main():
    print("=" * 80)
    print("STAGE 3: BUILDING TRUE UNIQUE-PATIENT INVENTORY & DUPLICATE GROUPS")
    print("=" * 80)

    dataset_dir = repo_root / "sharon" / "dataset"
    cases = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and (d / "segmentation.nii.gz").exists()])
    print(f"Total available cases on disk: {len(cases)}")

    splits_file = repo_root / "sharon" / "outputs" / "splits_pointcloud.json"
    with open(splits_file) as f:
        sp = json.load(f)
    tr_cases = set(sp["train_cases"])
    val_cases = set(sp["val_cases"])
    te_cases = set(sp["test_cases"])

    meta_file = dataset_dir / "metadata.json"
    meta = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

    # Process all cases
    records = []
    voxel_to_cases = defaultdict(list)

    for c in cases:
        cid = c.name
        res = compute_hashes_for_case(c)
        if res is None:
            continue
        f_hash, v_hash, shape, a_hash = res

        m = meta.get(cid, {})
        sex_val = m.get("sex", "UNKNOWN")
        source_patient_id = m.get("source", "UNKNOWN")

        cur_split = "train" if cid in tr_cases else ("val" if cid in val_cases else ("test" if cid in te_cases else "none"))

        rec = {
            "case_id": cid,
            "source_patient_id_if_available": source_patient_id,
            "ct_file_hash": f_hash,
            "ct_voxel_hash": v_hash,
            "shape": f"{shape[0]}x{shape[1]}x{shape[2]}",
            "affine_hash": a_hash[:16],
            "sex": sex_val,
            "current_split": cur_split
        }
        records.append(rec)
        voxel_to_cases[v_hash].append(rec)

    # Assign duplicate group IDs
    dup_group_counter = 0
    group_assigned_rows = []
    
    # Sort groups so identical scans are grouped together
    for v_hash, group in voxel_to_cases.items():
        if len(group) > 1:
            group_id = f"DUP_GRP_{dup_group_counter:03d}"
            dup_group_counter += 1
        else:
            group_id = "UNIQUE"
        
        for r in group:
            r["duplicate_group_id"] = group_id
            group_assigned_rows.append(r)

    # Sort rows by case_id
    group_assigned_rows.sort(key=lambda x: x["case_id"])

    out_csv = repo_root / "reports" / "phase1r" / "duplicate_groups.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "duplicate_group_id", "case_id", "source_patient_id_if_available",
            "ct_file_hash", "ct_voxel_hash", "shape", "affine_hash", "sex", "current_split"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(group_assigned_rows)

    print(f"\nInventory complete:")
    print(f"  Total scans evaluated: {len(group_assigned_rows)}")
    print(f"  Unique voxel volumes:  {len(voxel_to_cases)}")
    print(f"  Duplicate groups:      {dup_group_counter}")
    
    for v_hash, group in voxel_to_cases.items():
        if len(group) > 1:
            cids = [r["case_id"] for r in group]
            splits = [r["current_split"] for r in group]
            sexes = [r["sex"] for r in group]
            print(f"  Duplicate Group ({len(group)} cases): {cids} | Splits: {splits} | Sex: {sexes}")

    print(f"[Done] Output written to: {out_csv}")

if __name__ == "__main__":
    main()
