import unittest
import torch
import numpy as np
from pathlib import Path

from labels import (
    NUM_ORGANS, NUM_REGIONS, ORGAN_NAMES,
    get_sex_organ_mask, get_organ_region_indices, compute_regional_ground_truth
)
from model_gnn import HierarchicalSexConditionedGNN
from losses import HierarchicalDynamicMaskedLoss


class TestHierarchicalPipeline(unittest.TestCase):
    def test_region_mapping(self):
        reg_idx = get_organ_region_indices()
        self.assertEqual(reg_idx.shape, (121,))
        self.assertTrue(torch.all(reg_idx >= 0) and torch.all(reg_idx < NUM_REGIONS))

        # Check key organs
        lung_idx = ORGAN_NAMES.index("lung_upper_lobe_left")
        liver_idx = ORGAN_NAMES.index("liver")
        uterus_idx = ORGAN_NAMES.index("uterus")
        vertebra_idx = ORGAN_NAMES.index("vertebrae_L1")

        self.assertEqual(reg_idx[lung_idx].item(), 0)     # Thoracic
        self.assertEqual(reg_idx[liver_idx].item(), 1)    # Abdominal
        self.assertEqual(reg_idx[uterus_idx].item(), 2)   # Pelvic
        self.assertEqual(reg_idx[vertebra_idx].item(), 3) # Skeletal

    def test_regional_ground_truth_computation(self):
        B = 4
        centroids = torch.randn(B, 121, 3)
        mask = torch.ones(B, 121)
        r_gt = compute_regional_ground_truth(centroids, mask)
        self.assertEqual(r_gt.shape, (B, 4, 3))

    def test_model_forward_pass_and_regions(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HierarchicalSexConditionedGNN(
            num_organs=121,
            num_regions=4,
            pointnet_feat_dim=512,
            sex_emb_dim=64,
            hidden_dim=128,
            gnn_layers=2,
        ).to(device)

        B = 2
        pts = torch.randn(B, 4096, 3, device=device).clamp(-1.0, 1.0)
        sex = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
        mask = get_sex_organ_mask(sex, num_organs=121, device=device)

        pred_ctr, pred_reg = model(pts, sex, mask=mask, return_regional=True)
        self.assertEqual(pred_ctr.shape, (B, 121, 3))
        self.assertEqual(pred_reg.shape, (B, 4, 3))

        # Check male patient (batch 1): female organs must be zeroed out
        uterus_idx = ORGAN_NAMES.index("uterus")
        self.assertAlmostEqual(pred_ctr[1, uterus_idx].abs().sum().item(), 0.0, places=5)

        # Check female patient (batch 0): prostate must be zeroed out
        prostate_idx = ORGAN_NAMES.index("prostate")
        self.assertAlmostEqual(pred_ctr[0, prostate_idx].abs().sum().item(), 0.0, places=5)

    def test_multi_task_loss_and_gradients(self):
        criterion = HierarchicalDynamicMaskedLoss(beta=0.01, lambda_regional=1.0, lambda_offset=2.0)
        B = 2
        pred_ctr = torch.randn(B, 121, 3, requires_grad=True)
        gt_ctr = torch.randn(B, 121, 3)
        mask = torch.ones(B, 121)
        pred_reg = torch.randn(B, 4, 3, requires_grad=True)
        gt_reg = torch.randn(B, 4, 3)

        loss = criterion(pred_ctr, gt_ctr, mask, pred_reg, gt_reg)
        self.assertTrue(loss.item() > 0)
        loss.backward()
        self.assertIsNotNone(pred_ctr.grad)
        self.assertIsNotNone(pred_reg.grad)

    def test_denormalization_round_trip(self):
        # Physical coordinates
        raw_xyz = torch.tensor([[100.0, -200.0, 500.0], [300.0, -100.0, 900.0]])
        center = (raw_xyz.min(dim=0)[0] + raw_xyz.max(dim=0)[0]) / 2.0
        scale = (raw_xyz.max(dim=0)[0] - raw_xyz.min(dim=0)[0]) / 2.0

        norm_xyz = (raw_xyz - center) / scale
        self.assertTrue(torch.all(norm_xyz >= -1.0) and torch.all(norm_xyz <= 1.0))

        reconstructed_xyz = norm_xyz * scale + center
        diff = torch.norm(raw_xyz - reconstructed_xyz)
        self.assertAlmostEqual(diff.item(), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
