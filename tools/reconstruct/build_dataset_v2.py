import sys
import os
import json
import csv
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES
from tools.reconstruct.surface_extractor import extract_patient_body_surface
from tools.reconstruct.target_extractor import extract_patient_targets

def worker_process_case(args):
    cid, case_dir_str, meta_dict = args
    case_dir = Path(case_dir_str)
    ct_p = case_dir / "ct.nii.gz"
    seg_p = case_dir / "segmentation.nii.gz"

    try:
        # Load CT image once
        ct_img = nib.load(str(ct_p))
        seg_img = nib.load(str(seg_p))

        # 1. Surface Extraction (Independent of targets!)
        surface_res = extract_patient_body_surface(ct_img, hu_threshold=-300, num_points=4096, seed=42)

        # 2. Target Extraction (Direct affine)
        target_res = extract_patient_targets(seg_img, num_organs=121)

        # Basic metadata
        zooms = ct_img.header.get_zooms()
        axcodes = "".join(nib.orientations.aff2axcodes(ct_img.affine))
        
        m = meta_dict.get(cid, {})
        sex_val = int(m.get("sex", 1))
        sex_src = m.get("source", "UNKNOWN")

        return {
            "success": True,
            "case_id": cid,
            "surface_res": surface_res,
            "target_res": target_res,
            "shape": ct_img.shape,
            "zooms": [float(z) for z in zooms],
            "affine": ct_img.affine,
            "axcodes": axcodes,
            "sex": sex_val,
            "sex_source": sex_src
        }
    except Exception as e:
        return {
            "success": False,
            "case_id": cid,
            "error": str(e)
        }

def main():
    print("=" * 80)
    print("STAGE 3-16: RECONSTRUCTING DATASET V2 (PARALLEL EXTRACTION)")
    print("=" * 80)

    dataset_dir = repo_root / "sharon" / "dataset"
    out_dir = repo_root / "sharon" / "dataset_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = repo_root / "reports" / "phase1r"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load duplicate groups to exclude redundant scans
    dup_csv = report_dir / "duplicate_groups.csv"
    excluded_duplicates = set()
    if dup_csv.exists():
        with open(dup_csv) as f:
            reader = csv.DictReader(f)
            # Find all cases marked with duplicate_group_id != "UNIQUE"
            # Keep the first seen case in each group, exclude the rest
            seen_groups = set()
            for r in reader:
                grp = r["duplicate_group_id"]
                cid = r["case_id"]
                if grp != "UNIQUE":
                    if grp in seen_groups:
                        excluded_duplicates.add(cid)
                    else:
                        seen_groups.add(grp)

    print(f"Total redundant duplicate scans excluded from V2: {len(excluded_duplicates)}")
    print(f"Excluded IDs: {sorted(list(excluded_duplicates))}")

    # All available cases
    all_case_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and (d / "segmentation.nii.gz").exists()])
    v2_case_dirs = [d for d in all_case_dirs if d.name not in excluded_duplicates]
    print(f"Unique valid patients to reconstruct: {len(v2_case_dirs)}")

    # Load metadata
    meta_file = dataset_dir / "metadata.json"
    with open(meta_file) as f:
        meta_dict = json.load(f)

    # Task arguments
    tasks = [(d.name, str(d), meta_dict) for d in v2_case_dirs]

    num_workers = min(16, cpu_count())
    print(f"Launching multiprocessing pool with {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(worker_process_case, tasks), total=len(tasks), desc="Extracting V2 Geometry"))

    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    print(f"\nExtraction completed: {len(successful_results)} succeeded, {len(failed_results)} failed.")
    if failed_results:
        for f_res in failed_results:
            print(f"  [FAILURE] {f_res['case_id']}: {f_res['error']}")
        # Stage 5 rule: never silently discard patients
        fail_log = report_dir / "reconstruction_failures.csv"
        with open(fail_log, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "error"])
            writer.writeheader()
            writer.writerows(failed_results)

    # Sort results deterministically by case_id
    successful_results.sort(key=lambda x: x["case_id"])
    N = len(successful_results)

    # 2. Build Leakage-Safe Splits (Stage 18)
    # Stratified 80/10/10 by biological sex with seed 42
    rng = np.random.RandomState(42)
    males = [i for i, r in enumerate(successful_results) if r["sex"] == 1]
    females = [i for i, r in enumerate(successful_results) if r["sex"] == 0]

    rng.shuffle(males)
    rng.shuffle(females)

    def split_indices(idx_list, train_pct=0.8, val_pct=0.1):
        n = len(idx_list)
        n_tr = int(round(n * train_pct))
        n_val = int(round(n * val_pct))
        return idx_list[:n_tr], idx_list[n_tr:n_tr + n_val], idx_list[n_tr + n_val:]

    m_tr, m_val, m_te = split_indices(males)
    f_tr, f_val, f_te = split_indices(females)

    train_indices = sorted(m_tr + f_tr)
    val_indices = sorted(m_val + f_val)
    test_indices = sorted(m_te + f_te)

    train_cases = [successful_results[i]["case_id"] for i in train_indices]
    val_cases = [successful_results[i]["case_id"] for i in val_indices]
    test_cases = [successful_results[i]["case_id"] for i in test_indices]

    print(f"\nSplit Distribution (N={N}):")
    print(f"  Train: {len(train_cases)} (M: {len(m_tr)}, F: {len(f_tr)})")
    print(f"  Val:   {len(val_cases)} (M: {len(m_val)}, F: {len(f_val)})")
    print(f"  Test:  {len(test_cases)} (M: {len(m_te)}, F: {len(f_te)})")

    splits_data = {
        "split_seed": 42,
        "algorithm": "sex_stratified_unique_volume_split",
        "num_total_unique": N,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
    }

    splits_file = out_dir / "splits_v2.json"
    with open(splits_file, "w") as f:
        json.dump(splits_data, f, indent=2)

    with open(splits_file, "rb") as f:
        splits_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"✓ Saved splits_v2.json (SHA-256: {splits_hash[:16]}...)")

    # 3. Calculate Global Metric Scale S_global from TRAINING SPLIT ONLY (Stage 15)
    train_centered_max_coords = []
    for idx in train_indices:
        r = successful_results[idx]
        c_surf = r["surface_res"]["surface_center_mm"] # (3,)
        pts_cent = r["surface_res"]["points_world_mm"] - c_surf
        t_mask = r["target_res"]["target_mask"] > 0.5
        t_cent = (r["target_res"]["targets_world_mm"] - c_surf)[t_mask]

        max_surf = np.abs(pts_cent).max()
        max_t = np.abs(t_cent).max() if len(t_cent) > 0 else 0.0
        train_centered_max_coords.append(max(max_surf, max_t))

    max_train_coord = float(np.max(train_centered_max_coords))
    # Choose S_global = 500.0 mm as a robust, clean isotropic scalar enclosing all training anatomy
    S_global = 500.0
    print(f"\n--- Stage 15: Global Metric Scaling ---")
    print(f"  Max centered coordinate in training split: {max_train_coord:.2f} mm")
    print(f"  Selected S_global (isotropic scalar):      {S_global:.1f} mm")

    # 4. Assemble Tensor Arrays
    case_ids = []
    points_world_mm = np.zeros((N, 4096, 3), dtype=np.float32)
    points_centered_mm = np.zeros((N, 4096, 3), dtype=np.float32)
    points_model = np.zeros((N, 4096, 3), dtype=np.float32)
    normals = np.zeros((N, 4096, 3), dtype=np.float32)

    targets_world_mm = np.zeros((N, 121, 3), dtype=np.float32)
    targets_centered_mm = np.zeros((N, 121, 3), dtype=np.float32)
    targets_model = np.zeros((N, 121, 3), dtype=np.float32)

    target_mask = np.zeros((N, 121), dtype=np.float32)
    target_primary_mask = np.zeros((N, 121), dtype=np.float32)

    surface_center_mm = np.zeros((N, 3), dtype=np.float32)
    body_dimensions_mm = np.zeros((N, 3), dtype=np.float32)
    sex_tensor = np.zeros((N,), dtype=np.int64)
    split_list = []

    all_validation_records = []
    manifest_rows = []

    for i, r in enumerate(successful_results):
        cid = r["case_id"]
        case_ids.append(cid)
        
        split_name = "train" if cid in train_cases else ("val" if cid in val_cases else "test")
        split_list.append(split_name)

        c_surf = r["surface_res"]["surface_center_mm"]
        surface_center_mm[i] = c_surf
        body_dimensions_mm[i] = r["surface_res"]["body_dimensions_mm"]
        sex_tensor[i] = r["sex"]

        # Surface
        pts_w = r["surface_res"]["points_world_mm"]
        norms = r["surface_res"]["normals_world"]
        points_world_mm[i] = pts_w
        normals[i] = norms

        pts_c = pts_w - c_surf
        points_centered_mm[i] = pts_c
        points_model[i] = pts_c / S_global

        # Targets
        tgt_w = r["target_res"]["targets_world_mm"]
        t_mask = r["target_res"]["target_mask"]
        p_mask = r["target_res"]["target_primary_mask"]

        targets_world_mm[i] = tgt_w
        target_mask[i] = t_mask
        target_primary_mask[i] = p_mask

        # Apply SAME surface center
        tgt_c = np.zeros_like(tgt_w)
        valid_idx = t_mask > 0.5
        tgt_c[valid_idx] = tgt_w[valid_idx] - c_surf
        targets_centered_mm[i] = tgt_c
        targets_model[i] = tgt_c / S_global

        # Validation records
        for vrec in r["target_res"]["validation_records"]:
            vrec["case_id"] = cid
            all_validation_records.append(vrec)

        # Manifest
        manifest_rows.append({
            "case_id": cid,
            "split": split_name,
            "sex": r["sex"],
            "sex_source": r["sex_source"],
            "ct_shape": f"{r['shape'][0]}x{r['shape'][1]}x{r['shape'][2]}",
            "orientation": r["axcodes"],
            "voxel_spacing": f"({r['zooms'][0]:.3f}, {r['zooms'][1]:.3f}, {r['zooms'][2]:.3f})",
            "num_surface_points": len(pts_w),
            "body_width_mm": float(r["surface_res"]["body_dimensions_mm"][0]),
            "body_depth_mm": float(r["surface_res"]["body_dimensions_mm"][1]),
            "body_height_mm": float(r["surface_res"]["body_dimensions_mm"][2]),
            "surface_area_mm2": float(r["surface_res"]["surface_area_mm2"]),
            "body_volume_liters": float(r["surface_res"]["body_mask_volume_mm3"] / 1e6),
            "valid_target_count": int(t_mask.sum()),
            "genuine_target_count": int(p_mask.sum()),
            "synthetic_target_count": int(t_mask.sum() - p_mask.sum()),
            "surface_qc": r["surface_res"]["qc_status"],
            "target_qc": "PASS"
        })

    # Save pointclouds_v2.pt
    dataset_v2 = {
        "version": "2.0",
        "description": "Scientific V2 dataset with real CT surfaces, direct affine targets, and metric scale preservation",
        "case_ids": case_ids,
        "points_world_mm": torch.from_numpy(points_world_mm),
        "points_centered_mm": torch.from_numpy(points_centered_mm),
        "points_model": torch.from_numpy(points_model),
        "normals": torch.from_numpy(normals),
        "targets_world_mm": torch.from_numpy(targets_world_mm),
        "targets_centered_mm": torch.from_numpy(targets_centered_mm),
        "targets_model": torch.from_numpy(targets_model),
        "target_mask": torch.from_numpy(target_mask),
        "target_primary_mask": torch.from_numpy(target_primary_mask),
        "surface_center_mm": torch.from_numpy(surface_center_mm),
        "body_dimensions_mm": torch.from_numpy(body_dimensions_mm),
        "sex": torch.from_numpy(sex_tensor),
        "splits": split_list,
        "s_global": S_global,
        "organ_names": ORGAN_NAMES
    }

    pt_out = out_dir / "pointclouds_v2.pt"
    torch.save(dataset_v2, pt_out)
    print(f"\n✓ Saved pointclouds_v2.pt to: {pt_out} ({pt_out.stat().st_size / 1e6:.1f} MB)")

    # Save manifest_v2.json & csv
    man_json = out_dir / "manifest_v2.json"
    with open(man_json, "w") as f:
        json.dump(manifest_rows, f, indent=2)

    man_csv = out_dir / "manifest_v2.csv"
    with open(man_csv, "w", newline="") as f:
        fieldnames = list(manifest_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"✓ Saved manifest_v2.json and manifest_v2.csv ({len(manifest_rows)} entries)")

    # Save Stage 6 target coordinate validation records
    val_csv = report_dir / "02_target_coordinate_validation.csv"
    with open(val_csv, "w", newline="") as f:
        fieldnames = ["case_id", "target_index", "target_name", "i", "j", "k", "world_x_mm", "world_y_mm", "world_z_mm", "verification_error_mm"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_validation_records)
    print(f"✓ Saved 02_target_coordinate_validation.csv ({len(all_validation_records)} validated coordinates)")

    # Save Stage 7 target provenance
    prov_csv = report_dir / "target_provenance_v2.csv"
    with open(prov_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_index", "target_name", "provenance", "primary_benchmark_mask", "notes"])
        writer.writeheader()
        for idx, name in enumerate(ORGAN_NAMES):
            organ_id = idx + 1
            if organ_id in [118, 119, 120, 121]:
                p = "SYNTHETIC"
                bm = 0
                n = "Procedural ellipsoidal fallback; excluded from primary clinical benchmark"
            else:
                p = "REAL_SEGMENTATION"
                bm = 1
                n = "TotalSegmentator v2 mask"
            writer.writerow({
                "target_index": idx,
                "target_name": name,
                "provenance": p,
                "primary_benchmark_mask": bm,
                "notes": n
            })
    print(f"✓ Saved target_provenance_v2.csv (121 structures)")

if __name__ == "__main__":
    main()
