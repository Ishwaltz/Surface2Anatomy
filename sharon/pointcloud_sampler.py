import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from multiprocessing import Pool

import numpy as np
import torch
import nibabel as nib
import scipy.ndimage as ndi
from tqdm import tqdm

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from sharon.labels import (
        ORGAN_NAMES, NUM_ORGANS, NUM_REGIONS,
        get_sex_organ_mask, get_organ_region_indices, compute_regional_ground_truth
    )
except ImportError:
    from labels import (
        ORGAN_NAMES, NUM_ORGANS, NUM_REGIONS,
        get_sex_organ_mask, get_organ_region_indices, compute_regional_ground_truth
    )


def load_canonical_surface_points(mesh_path: str = "sharon/outputs/meshes/skin.obj", num_points: int = 4096) -> np.ndarray:
    p = Path(mesh_path)
    if not p.exists() and Path("outputs/meshes/skin.obj").exists():
        p = Path("outputs/meshes/skin.obj")
    
    if p.exists() and HAS_TRIMESH:
        mesh = trimesh.load(str(p), force="mesh")
        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        if len(pts) >= num_points:
            return pts[:num_points].astype(np.float32)
        pad = np.random.choice(len(pts), num_points - len(pts), replace=True)
        return np.vstack([pts, pts[pad]]).astype(np.float32)
    
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    u = np.random.uniform(0, 1, num_points)
    theta = np.arccos(costheta)
    r = 150.0 * (u ** (1/3))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi) * 0.75
    z = r * np.cos(theta) * 2.5
    return np.column_stack([x, y, z]).astype(np.float32)


def _worker_normalized_case(args_tuple):
    case_path_str, meta_dict, base_skin_pts, num_points, num_organs = args_tuple
    case_dir = Path(case_path_str)
    seg_path = case_dir / "segmentation.nii.gz"
    case_id = case_dir.name

    if not seg_path.exists():
        return None

    try:
        seg_img = nib.load(str(seg_path))
        seg_data = seg_img.get_fdata().astype(np.int16)
        affine = seg_img.affine

        # Vectorized single-pass center of mass for all 121 classes
        coms = ndi.center_of_mass(np.ones_like(seg_data, dtype=np.uint8), labels=seg_data, index=np.arange(1, num_organs + 1))

        raw_centroids = np.zeros((num_organs, 3), dtype=np.float32)
        presence_mask = np.zeros(num_organs, dtype=np.float32)
        all_coords = []

        for org_idx, com in enumerate(coms):
            if not np.isnan(com[0]):
                cz, cy, cx = com
                phys = (affine @ np.array([cx, cy, cz, 1.0], dtype=np.float32))[:3]
                raw_centroids[org_idx] = phys
                presence_mask[org_idx] = 1.0
                all_coords.append(phys)

        if not all_coords:
            return None

        # Patient sex
        case_meta = meta_dict.get(case_id, {})
        sex_val = case_meta.get("sex", None)
        if sex_val is None:
            prostate_idx = ORGAN_NAMES.index("prostate")
            sex_val = 1 if presence_mask[prostate_idx] > 0.5 else 0

        sex_val = int(sex_val)
        one_hot_sex = np.array([1.0, 0.0] if sex_val == 0 else [0.0, 1.0], dtype=np.float32)
        sex_organ_mask = get_sex_organ_mask(sex_val, num_organs=num_organs).numpy()
        combined_mask = (presence_mask * sex_organ_mask).astype(np.float32)

        # Fit skin surface points to patient-specific physical bounding geometry
        all_coords = np.array(all_coords)
        min_ras = all_coords.min(axis=0) - 15.0
        max_ras = all_coords.max(axis=0) + 15.0
        center_ras = (min_ras + max_ras) / 2.0
        scale_ras = np.maximum(max_ras - min_ras, 10.0)

        ref_min = base_skin_pts.min(axis=0)
        ref_max = base_skin_pts.max(axis=0)
        ref_center = (ref_min + ref_max) / 2.0
        ref_scale = np.maximum(ref_max - ref_min, 1.0)

        raw_points = (base_skin_pts - ref_center) / ref_scale * scale_ras + center_ras
        raw_points = raw_points + np.random.randn(*raw_points.shape).astype(np.float32) * 0.5

        # STRICT UNIT BOUNDING BOX NORMALIZATION: [-1.0, 1.0]^3
        pc_min = raw_points.min(axis=0)
        pc_max = raw_points.max(axis=0)
        center_c = (pc_min + pc_max) / 2.0
        scale_s = np.maximum((pc_max - pc_min) / 2.0, 1e-4)

        norm_points = (raw_points - center_c) / scale_s
        norm_centroids = (raw_centroids - center_c) / scale_s

        # Mask invalid organ centroids to 0
        norm_centroids = norm_centroids * combined_mask[:, None]

        return (
            case_id,
            norm_points.astype(np.float32),
            norm_centroids.astype(np.float32),
            raw_points.astype(np.float32),
            raw_centroids.astype(np.float32),
            center_c.astype(np.float32),
            scale_s.astype(np.float32),
            one_hot_sex,
            sex_val,
            combined_mask,
            sex_organ_mask,
        )
    except Exception:
        return None


class PointCloudSampler:
    def __init__(
        self,
        dataset_dir: str,
        metadata_file: Optional[str] = None,
        mesh_dir: Optional[str] = None,
        num_points: int = 4096,
        num_organs: int = 121,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.num_points = num_points
        self.num_organs = num_organs
        self.mesh_dir = Path(mesh_dir) if mesh_dir else None

        if metadata_file and Path(metadata_file).exists():
            meta_path = Path(metadata_file)
        elif (self.dataset_dir / "metadata.json").exists():
            meta_path = self.dataset_dir / "metadata.json"
        elif Path("sharon/dataset/metadata.json").exists():
            meta_path = Path("sharon/dataset/metadata.json")
        elif Path("dataset/metadata.json").exists():
            meta_path = Path("dataset/metadata.json")
        else:
            meta_path = None

        self.metadata = {}
        if meta_path and meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

    def sample_and_export_dataset(
        self,
        output_file: str,
        max_cases: Optional[int] = None,
        num_workers: int = 16,
    ) -> str:
        cases = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir() and (d / "segmentation.nii.gz").exists()])
        if max_cases:
            cases = cases[:max_cases]

        print(f"\n==================================================")
        print(f" Normalized Point Cloud Sampling ([-1, 1]^3, N={self.num_points}, K={self.num_organs})")
        print(f" Dataset dir:  {self.dataset_dir}")
        print(f" Total cases:  {len(cases)}")
        print(f" Processes:    {num_workers}")
        print(f" Output file:  {output_file}")
        print(f"==================================================")

        base_skin = load_canonical_surface_points(num_points=self.num_points)
        task_args = [(str(c), self.metadata, base_skin, self.num_points, self.num_organs) for c in cases]

        with Pool(processes=num_workers) as pool:
            raw_results = list(tqdm(pool.imap_unordered(_worker_normalized_case, task_args, chunksize=8), total=len(cases), desc="Normalizing Extraction"))

        valid_results = [r for r in raw_results if r is not None]
        valid_results.sort(key=lambda x: x[0])

        all_case_ids = [r[0] for r in valid_results]
        all_norm_points = np.stack([r[1] for r in valid_results], axis=0)
        all_norm_centroids = np.stack([r[2] for r in valid_results], axis=0)
        all_raw_points = np.stack([r[3] for r in valid_results], axis=0)
        all_raw_centroids = np.stack([r[4] for r in valid_results], axis=0)
        all_centers = np.stack([r[5] for r in valid_results], axis=0)
        all_scales = np.stack([r[6] for r in valid_results], axis=0)
        all_sex_priors = np.stack([r[7] for r in valid_results], axis=0)
        all_sex_vals = np.array([r[8] for r in valid_results], dtype=np.int64)
        all_masks = np.stack([r[9] for r in valid_results], axis=0)
        all_sex_masks = np.stack([r[10] for r in valid_results], axis=0)

        # Precompute regional ground truth
        t_norm_centroids = torch.from_numpy(all_norm_centroids).float()
        t_masks = torch.from_numpy(all_masks).float()
        region_gt = compute_regional_ground_truth(t_norm_centroids, t_masks)

        dataset_tensor_dict = {
            "points": torch.from_numpy(all_norm_points).float(),             # (B, 4096, 3) in [-1, 1]
            "centroids": t_norm_centroids,                                    # (B, 121, 3) in [-1, 1]
            "regional_gt": region_gt,                                         # (B, 4, 3) in [-1, 1]
            "raw_points": torch.from_numpy(all_raw_points).float(),          # (B, 4096, 3) in physical mm
            "raw_centroids": torch.from_numpy(all_raw_centroids).float(),    # (B, 121, 3) in physical mm
            "centers": torch.from_numpy(all_centers).float(),                # (B, 3) physical center
            "scales": torch.from_numpy(all_scales).float(),                  # (B, 3) physical half-extent
            "sex_priors": torch.from_numpy(all_sex_priors).float(),          # (B, 2)
            "sex_vals": torch.from_numpy(all_sex_vals).long(),               # (B,)
            "masks": t_masks,                                                # (B, 121)
            "sex_masks": torch.from_numpy(all_sex_masks).float(),            # (B, 121)
            "region_indices": get_organ_region_indices(),                    # (121,)
            "case_ids": all_case_ids,
            "organ_names": ORGAN_NAMES,
            "num_organs": self.num_organs,
            "num_regions": NUM_REGIONS,
            "num_points": self.num_points,
        }

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset_tensor_dict, str(out_path))

        if "sharon" in str(out_path):
            mirror_path = Path(str(out_path).replace("sharon/", ""))
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dataset_tensor_dict, str(mirror_path))

        print("\n" + "=" * 50)
        print("NORMALIZED POINT CLOUD DATASET EXPORT COMPLETE")
        print(f"Total Patients:   {len(all_case_ids)}")
        print(f"Norm Points:      {dataset_tensor_dict['points'].shape} (Range: [{dataset_tensor_dict['points'].min():.2f}, {dataset_tensor_dict['points'].max():.2f}])")
        print(f"Norm Centroids:   {dataset_tensor_dict['centroids'].shape}")
        print(f"Regional GT:      {dataset_tensor_dict['regional_gt'].shape}")
        print(f"Centers/Scales:   {dataset_tensor_dict['centers'].shape}, {dataset_tensor_dict['scales'].shape}")
        print(f"Export Path:      {out_path}")
        print("=" * 50)
        return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Normalized Point Cloud Surface Sampler")
    parser.add_argument("--dataset_dir", type=str, default="sharon/dataset", help="Path to preprocessed dataset")
    parser.add_argument("--metadata_file", type=str, default="sharon/dataset/metadata.json", help="Path to metadata.json")
    parser.add_argument("--mesh_dir", type=str, default="sharon/outputs/meshes", help="Directory containing skin.obj")
    parser.add_argument("--output_file", type=str, default="sharon/dataset/pointclouds_450.pt", help="Path for exported .pt")
    parser.add_argument("--num_points", type=int, default=4096, help="Number of surface points")
    parser.add_argument("--num_organs", type=int, default=121, help="Number of organ classes")
    parser.add_argument("--num_workers", type=int, default=16, help="Parallel worker processes")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit max cases to process")
    args = parser.parse_args()

    sampler = PointCloudSampler(
        dataset_dir=args.dataset_dir,
        metadata_file=args.metadata_file,
        mesh_dir=args.mesh_dir,
        num_points=args.num_points,
        num_organs=args.num_organs,
    )
    sampler.sample_and_export_dataset(args.output_file, max_cases=args.max_cases, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
