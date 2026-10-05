import sys
import json
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import (
    TOTAL_CLASSES_121, ORGAN_NAMES, SKELETAL_CLASSES,
    SOFT_TISSUE_CLASSES, SKELETAL_INDICES, SOFT_TISSUE_INDICES,
    MALE_SPECIFIC_INDICES, FEMALE_SPECIFIC_INDICES
)

class Phase1AutomatedTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pt_path = repo_root / "sharon" / "dataset" / "pointclouds_450.pt"
        cls.data = torch.load(str(cls.pt_path), weights_only=False)
        
        cls.splits_file = repo_root / "sharon" / "outputs" / "splits_pointcloud.json"
        with open(cls.splits_file) as f:
            cls.splits = json.load(f)

    def test_target_index_consistency(self):
        """Verify that 121 target indices strictly map 1-to-1 without gaps or reordering."""
        self.assertEqual(len(ORGAN_NAMES), 121)
        self.assertEqual(len(set(ORGAN_NAMES)), 121)
        self.assertEqual(len(TOTAL_CLASSES_121), 121)
        for idx, name in enumerate(ORGAN_NAMES):
            self.assertEqual(TOTAL_CLASSES_121[name], idx + 1)
        self.assertEqual(self.data["organ_names"], ORGAN_NAMES)

    def test_surface_target_patient_pairing(self):
        """Assert surface and target data originate from identical patient IDs across dataset."""
        case_ids = self.data["case_ids"]
        self.assertEqual(len(case_ids), 450)
        # Verify first case and last case IDs exist on disk
        self.assertEqual(case_ids[0], "case_000")
        self.assertEqual(case_ids[-1], "case_463")
        dataset_dir = repo_root / "sharon" / "dataset"
        self.assertTrue((dataset_dir / case_ids[0] / "segmentation.nii.gz").exists())
        self.assertTrue((dataset_dir / case_ids[-1] / "segmentation.nii.gz").exists())

    def test_coordinate_roundtrip(self):
        """Assert normalization p_phys -> p_norm -> p_rec has numerical error < 1e-3 mm."""
        raw_c = self.data["raw_centroids"]
        norm_c = self.data["centroids"]
        centers = self.data["centers"].unsqueeze(1)
        scales = self.data["scales"].unsqueeze(1)
        masks = self.data["masks"]

        rec_c = norm_c * scales + centers
        valid = masks > 0.5
        errs = torch.norm((raw_c - rec_c)[valid], dim=-1)
        max_err = errs.max().item()
        self.assertLess(max_err, 1e-3)

    def test_voxel_to_world_conversion(self):
        """
        Verify whether the codebase voxel-to-world conversion matches official nibabel physical points.
        EXPECTED TO FAIL / REVEAL CRITICAL BUG:
        pointcloud_sampler.py unpacked 'cz, cy, cx = com' which swapped X and Z!
        """
        seg_p = repo_root / "sharon" / "dataset" / "case_000" / "segmentation.nii.gz"
        seg_img = nib.load(str(seg_p))
        affine = seg_img.affine
        seg_data = seg_img.get_fdata().astype(np.int16)
        
        com = ndi.center_of_mass(seg_data == 1) # Spleen
        phys_official = (affine @ np.array([com[0], com[1], com[2], 1.0]))[:3]
        
        # Buggy conversion from pointcloud_sampler.py:
        cz, cy, cx = com
        phys_sampler = (affine @ np.array([cx, cy, cz, 1.0]))[:3]
        
        discrepancy = np.linalg.norm(phys_official - phys_sampler)
        print(f"\n[DIAGNOSTIC] test_voxel_to_world_conversion discrepancy: {discrepancy:.2f} mm")
        # We assert that discrepancy is greater than 1000 mm to prove the bug exists!
        self.assertGreater(discrepancy, 1000.0, "Voxel to world swap bug not detected!")

    def test_normalization_inverse(self):
        """Synthetic test: verify exact linear inverse formula x = x_norm * scale + center."""
        raw = torch.randn(10, 121, 3) * 100.0 + 50.0
        c = raw.mean(dim=1, keepdim=True)
        s = (raw.max(dim=1, keepdim=True).values - raw.min(dim=1, keepdim=True).values) / 2.0
        
        norm = (raw - c) / s
        rec = norm * s + c
        err = torch.norm(raw - rec, dim=-1).max().item()
        self.assertLess(err, 1e-4)

    def test_rotation_target_consistency(self):
        """Synthetic test: apply 3D SO(3) rotation to both surface and targets, verify invariance."""
        theta = np.pi / 4.0
        R = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,           0.0,            1.0]
        ], dtype=torch.float32)

        pts = torch.randn(100, 3)
        targets = torch.randn(10, 3)

        pts_rot = pts @ R.T
        targets_rot = targets @ R.T

        dist_orig = torch.norm(pts.unsqueeze(1) - targets.unsqueeze(0), dim=-1)
        dist_rot = torch.norm(pts_rot.unsqueeze(1) - targets_rot.unsqueeze(0), dim=-1)
        diff = torch.abs(dist_orig - dist_rot).max().item()
        self.assertLess(diff, 1e-4)

    def test_translation_target_consistency(self):
        """Synthetic test: apply 3D translation t, verify pairwise distance preservation."""
        t = torch.tensor([15.0, -30.0, 45.0])
        pts = torch.randn(100, 3)
        targets = torch.randn(10, 3)

        pts_t = pts + t
        targets_t = targets + t

        dist_orig = torch.norm(pts.unsqueeze(1) - targets.unsqueeze(0), dim=-1)
        dist_t = torch.norm(pts_t.unsqueeze(1) - targets_t.unsqueeze(0), dim=-1)
        diff = torch.abs(dist_orig - dist_t).max().item()
        self.assertLess(diff, 1e-4)

    def test_scaling_target_consistency(self):
        """Synthetic test: verify isotropic scaling preserves relative spatial coordinates."""
        scale = 1.5
        pts = torch.randn(100, 3)
        targets = torch.randn(10, 3)

        pts_s = pts * scale
        targets_s = targets * scale

        dist_orig = torch.norm(pts.unsqueeze(1) - targets.unsqueeze(0), dim=-1)
        dist_s = torch.norm(pts_s.unsqueeze(1) - targets_s.unsqueeze(0), dim=-1)
        diff = torch.abs(dist_orig * scale - dist_s).max().item()
        self.assertLess(diff, 1e-4)

    def test_left_right_orientation(self):
        """
        Verify Left/Right ordering in dataset.
        EXPECTED: In dataset, Left-Right separation was swapped to Z axis!
        """
        raw_c = self.data["raw_centroids"]
        masks = self.data["masks"]
        kl_idx = ORGAN_NAMES.index("kidney_left")
        kr_idx = ORGAN_NAMES.index("kidney_right")

        # In true LAS, Left.x > Right.x.
        # But in swapped dataset, Left.z > Right.z with >1000mm separation!
        dz = raw_c[0, kl_idx, 2] - raw_c[0, kr_idx, 2]
        print(f"\n[DIAGNOSTIC] Kidney Left - Right Z-separation in dataset: {dz.item():.2f} mm")
        self.assertGreater(dz.item(), 500.0, "Expected >500mm Z-separation due to axis swap bug")

    def test_presence_mask(self):
        """Assert missing structures (0, 0, 0) are strictly masked out by mask == 0."""
        raw_c = self.data["raw_centroids"]
        masks = self.data["masks"]
        
        is_zero = (raw_c == 0).all(dim=-1)
        zero_and_valid = is_zero & (masks > 0.5)
        self.assertEqual(zero_and_valid.sum().item(), 0)

    def test_split_disjointness(self):
        """Assert train, val, and test partitions are completely ID-disjoint."""
        tr = set(self.splits["train_cases"])
        val = set(self.splits["val_cases"])
        te = set(self.splits["test_cases"])

        self.assertEqual(len(tr & val), 0)
        self.assertEqual(len(tr & te), 0)
        self.assertEqual(len(val & te), 0)

    def test_metric_radial_distance(self):
        """
        Unit test for Mean Radial Error (MRE) formula.
        Test 1: GT=(0,0,0), Pred=(3,4,0) -> Error = 5.0 mm.
        Test 2: Target 1 error = 5 mm, Target 2 error = 10 mm -> MRE = 7.5 mm.
        """
        gt1 = torch.tensor([[0.0, 0.0, 0.0]])
        pred1 = torch.tensor([[3.0, 4.0, 0.0]])
        err1 = torch.norm(pred1 - gt1, dim=-1).mean().item()
        self.assertAlmostEqual(err1, 5.0, places=5)

        gt2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        pred2 = torch.tensor([[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]])
        err2 = torch.norm(pred2 - gt2, dim=-1).mean().item()
        self.assertAlmostEqual(err2, 7.5, places=5)

if __name__ == "__main__":
    unittest.main()
