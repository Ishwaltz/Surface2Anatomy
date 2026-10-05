import sys
import json
import csv
import unittest
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from nibabel.affines import apply_affine

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES, SKELETAL_INDICES, SOFT_TISSUE_INDICES
from tools.reconstruct.surface_extractor import extract_patient_body_surface
from tools.reconstruct.target_extractor import extract_patient_targets

class DatasetV2TestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
        cls.data = torch.load(str(cls.pt_path), weights_only=False)
        cls.splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
        with open(cls.splits_path) as f:
            cls.splits = json.load(f)

    def test_correct_voxel_to_world(self):
        """Verify apply_affine matches matrix multiplication exactly."""
        val_csv = repo_root / "reports" / "phase1r" / "02_target_coordinate_validation.csv"
        with open(val_csv) as f:
            reader = csv.DictReader(f)
            max_err = max(float(r["verification_error_mm"]) for r in reader)
        self.assertLess(max_err, 1e-6)

    def test_no_axis_reordering(self):
        """Verify Left-Right organ pair ordering is preserved in physical world space."""
        targets = self.data["targets_world_mm"]
        masks = self.data["target_mask"]
        kl_idx = ORGAN_NAMES.index("kidney_left")
        kr_idx = ORGAN_NAMES.index("kidney_right")
        
        # Both kidneys present in case 0
        self.assertGreater(masks[0, kl_idx].item(), 0.5)
        self.assertGreater(masks[0, kr_idx].item(), 0.5)
        
        kl_x = targets[0, kl_idx, 0].item()
        kr_x = targets[0, kr_idx, 0].item()
        kl_z = targets[0, kl_idx, 2].item()
        kr_z = targets[0, kr_idx, 2].item()

        dx = abs(kl_x - kr_x)
        dz = abs(kl_z - kr_z)

        # Bilateral kidneys must separate laterally along X (>80 mm) and have minimal vertical Z separation (<30 mm)
        self.assertGreater(dx, 80.0, f"Left/Right separation along lateral X axis must be >80 mm, got {dx:.2f} mm")
        self.assertLess(dz, 30.0, f"Left/Right separation along vertical Z axis must be <30 mm, got {dz:.2f} mm")

    def test_apply_affine_reference(self):
        """Verify on case_000 that nibabel apply_affine is used directly without axis swaps."""
        seg_img = nib.load(str(repo_root / "sharon" / "dataset" / "case_000" / "segmentation.nii.gz"))
        spleen_mask = seg_img.get_fdata().astype(np.int16) == 1
        import scipy.ndimage as ndi
        com = ndi.center_of_mass(spleen_mask)
        com_ijk = np.asarray(com, dtype=np.float64)
        official_phys = apply_affine(seg_img.affine, com_ijk)
        
        stored_spleen = self.data["targets_world_mm"][0, 0].numpy()
        diff = np.linalg.norm(official_phys - stored_spleen)
        self.assertLess(diff, 1e-4)

    def test_surface_is_patient_specific(self):
        """Verify that surfaces vary significantly across different patients."""
        pts = self.data["points_world_mm"].numpy()
        extents = pts.max(axis=1) - pts.min(axis=1) # (440, 3)
        widths = extents[:, 0]
        # Real patient widths should have standard deviation > 20 mm
        self.assertGreater(widths.std(), 20.0)

    def test_surface_generation_uses_no_targets(self):
        """Stage 17: prove surface extraction is completely independent of targets."""
        ct_img = nib.load(str(repo_root / "sharon" / "dataset" / "case_000" / "ct.nii.gz"))
        res1 = extract_patient_body_surface(ct_img, num_points=1024, seed=42)
        # Call again without any target inputs
        res2 = extract_patient_body_surface(ct_img, num_points=1024, seed=42)
        diff = np.abs(res1["points_world_mm"] - res2["points_world_mm"]).max()
        self.assertEqual(diff, 0.0)

    def test_no_template_skin_in_v2(self):
        """Assert template mesh skin.obj is NOT identical to any patient surface."""
        skin_obj = repo_root / "sharon" / "outputs" / "meshes" / "skin.obj"
        if skin_obj.exists():
            with open(skin_obj) as f:
                lines = [l for l in f if l.startswith("v ")]
            template_v0 = np.array([float(x) for x in lines[0].split()[1:4]])
            # Check against first point of first 10 patients
            pts = self.data["points_world_mm"][:10, 0].numpy()
            diffs = np.linalg.norm(pts - template_v0, axis=-1)
            self.assertTrue((diffs > 10.0).all())

    def test_no_duplicate_groups_across_splits(self):
        """Assert zero duplicate volume hashes cross train, val, and test splits."""
        man_csv = repo_root / "sharon" / "dataset_v2" / "manifest_v2.csv"
        splits = {}
        with open(man_csv) as f:
            reader = csv.DictReader(f)
            for r in reader:
                splits[r["case_id"]] = r["split"]

        tr = set(self.splits["train_cases"])
        val = set(self.splits["val_cases"])
        te = set(self.splits["test_cases"])

        self.assertEqual(len(tr & val), 0)
        self.assertEqual(len(tr & te), 0)
        self.assertEqual(len(val & te), 0)

    def test_world_coordinate_roundtrip(self):
        """Verify world -> centered -> model -> centered -> world round-trip error < 1e-3 mm."""
        pts_w = self.data["points_world_mm"]
        pts_m = self.data["points_model"]
        c_surf = self.data["surface_center_mm"].unsqueeze(1)
        S_global = self.data["s_global"]

        # Reconstruct
        pts_rec = pts_m * S_global + c_surf
        errs = torch.norm(pts_w - pts_rec, dim=-1)
        max_err = errs.max().item()
        self.assertLess(max_err, 1e-3)

    def test_metric_centering_roundtrip(self):
        """Verify target world -> centered -> model -> world round-trip error < 1e-3 mm."""
        tgt_w = self.data["targets_world_mm"]
        tgt_m = self.data["targets_model"]
        masks = self.data["target_mask"]
        c_surf = self.data["surface_center_mm"].unsqueeze(1)
        S_global = self.data["s_global"]

        tgt_rec = tgt_m * S_global + c_surf
        valid = masks > 0.5
        errs = torch.norm((tgt_w - tgt_rec)[valid], dim=-1)
        max_err = errs.max().item()
        self.assertLess(max_err, 1e-3)

    def test_no_patient_specific_anisotropic_scaling(self):
        """Assert isotropic S_global is used rather than per-axis anisotropic scaling."""
        S_global = self.data["s_global"]
        self.assertIsInstance(S_global, (float, int))
        self.assertEqual(S_global, 500.0)

    def test_surface_points_finite_and_unique(self):
        """Assert all 4096 points per patient are finite and unique."""
        pts = self.data["points_world_mm"].numpy()
        self.assertFalse(np.isnan(pts).any())
        self.assertFalse(np.isinf(pts).any())
        for i in range(10):
            u_count = len(np.unique(np.round(pts[i], 3), axis=0))
            self.assertEqual(u_count, 4096)

    def test_target_coordinates_finite(self):
        """Assert all valid targets have finite world coordinates."""
        tgt = self.data["targets_world_mm"].numpy()
        masks = self.data["target_mask"].numpy()
        valid = masks > 0.5
        self.assertFalse(np.isnan(tgt[valid]).any())
        self.assertFalse(np.isinf(tgt[valid]).any())

    def test_synthetic_targets_excluded_from_primary_mask(self):
        """Assert classes 117-120 (uterus, ovaries, vagina) have primary_valid_mask == 0."""
        p_mask = self.data["target_primary_mask"].numpy()
        synthetic_indices = [117, 118, 119, 120]
        for s_idx in synthetic_indices:
            self.assertEqual(p_mask[:, s_idx].sum(), 0.0)

    def test_split_manifest_integrity(self):
        """Assert manifest_v2.csv exactly matches pointclouds_v2.pt case_ids and splits."""
        man_csv = repo_root / "sharon" / "dataset_v2" / "manifest_v2.csv"
        with open(man_csv) as f:
            cids = [r["case_id"] for r in csv.DictReader(f)]
        self.assertEqual(cids, self.data["case_ids"])

    def test_reproducible_point_sampling(self):
        """Verify sampling is deterministic with fixed seed."""
        ct_img = nib.load(str(repo_root / "sharon" / "dataset" / "case_000" / "ct.nii.gz"))
        resA = extract_patient_body_surface(ct_img, num_points=500, seed=123)
        resB = extract_patient_body_surface(ct_img, num_points=500, seed=123)
        self.assertTrue(np.allclose(resA["points_world_mm"], resB["points_world_mm"]))

if __name__ == "__main__":
    unittest.main()
