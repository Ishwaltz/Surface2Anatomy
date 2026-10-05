import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Union

from labels import (
    NUM_ORGANS, ORGAN_NAMES, TOTAL_CLASSES_121,
    get_sex_organ_mask, get_organ_region_indices, compute_regional_ground_truth
)

class PointCloudOrganDataset(Dataset):
    """
    Dataset loader for 3D Torso Surface Point Clouds and 121 Organ Centroids.
    Supports strict unit bounding box normalized coordinates in [-1, 1]^3
    with centers and scales for exact physical mm de-normalization.
    """
    def __init__(
        self,
        pt_path: str = "sharon/dataset/pointclouds_450.pt",
        augment: bool = False,
        jitter_std: float = 0.005,
        rotation_aug: bool = False,
    ):
        self.pt_path = Path(pt_path)
        if not self.pt_path.exists() and Path(str(pt_path).replace("sharon/", "")).exists():
            self.pt_path = Path(str(pt_path).replace("sharon/", ""))

        data = torch.load(str(self.pt_path), map_location="cpu", weights_only=False)
        self.points = data["points"]                 # (N, 4096, 3) in [-1, 1]
        self.centroids = data["centroids"]           # (N, 121, 3) in [-1, 1]
        self.sex_priors = data["sex_priors"]         # (N, 2)
        self.sex_vals = data["sex_vals"] if "sex_vals" in data else torch.argmax(self.sex_priors, dim=-1)
        self.masks = data["masks"]                   # (N, 121)
        self.sex_masks = data["sex_masks"]           # (N, 121)
        self.case_ids = data["case_ids"]

        self.centers = data.get("centers", torch.zeros((len(self.case_ids), 3)))
        self.scales = data.get("scales", torch.ones((len(self.case_ids), 3)))
        self.raw_centroids = data.get("raw_centroids", self.centroids)
        self.region_indices = data.get("region_indices", get_organ_region_indices())
        self.regional_gt = data.get("regional_gt", compute_regional_ground_truth(self.centroids, self.masks, self.region_indices))

        self.augment = augment
        self.jitter_std = jitter_std
        self.rotation_aug = rotation_aug

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pts = self.points[idx].clone()
        ctr = self.centroids[idx].clone()
        reg_gt = self.regional_gt[idx].clone()
        sex = self.sex_priors[idx].clone()
        mask = self.masks[idx].clone()
        sex_mask = self.sex_masks[idx].clone()
        center = self.centers[idx].clone()
        scale = self.scales[idx].clone()
        raw_ctr = self.raw_centroids[idx].clone()

        if self.augment:
            # Jitter point cloud slightly in normalized space
            noise = torch.randn_like(pts) * self.jitter_std
            pts = torch.clamp(pts + noise, -1.2, 1.2)

        return {
            "points": pts,                           # (4096, 3) in [-1, 1]
            "centroids": ctr,                        # (121, 3) in [-1, 1]
            "regional_gt": reg_gt,                   # (4, 3) in [-1, 1]
            "sex_prior": sex,                        # (2,)
            "sex_val": self.sex_vals[idx],           # scalar
            "mask": mask,                            # (121,)
            "sex_mask": sex_mask,                    # (121,)
            "center": center,                        # (3,)
            "scale": scale,                          # (3,)
            "raw_centroids": raw_ctr,                # (121, 3) physical mm
            "case_id": self.case_ids[idx],
        }
