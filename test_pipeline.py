import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add paths
sharon_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(sharon_dir))
sys.path.insert(0, str(sharon_dir.parent))

from labels import (
    NUM_ORGANS, ORGAN_NAMES, TOTAL_CLASSES_121,
    MALE_SPECIFIC_INDICES, FEMALE_SPECIFIC_INDICES,
    get_sex_organ_mask
)
from model_gnn import SexConditionedOrganGNN, PointNet2Encoder, SexEncoder
from losses import DynamicMaskedCentroidLoss, MultiTaskLoss
from dataset import PointCloudOrganDataset


class TestSexConditionedPipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.num_points = 4096
        self.num_organs = 121

        # Batch with 1 Female (0) and 1 Male (1)
        self.points = torch.randn(self.batch_size, self.num_points, 3, device=self.device)
        self.sex_priors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=self.device) # [F, M]
        self.gt_centroids = torch.randn(self.batch_size, self.num_organs, 3, device=self.device) * 50.0

    def test_labels_and_classes(self):
        """Test 121 classes and sex organ definitions."""
        self.assertEqual(NUM_ORGANS, 121)
        self.assertEqual(len(ORGAN_NAMES), 121)
        self.assertEqual(TOTAL_CLASSES_121["uterus"], 118)
        self.assertEqual(TOTAL_CLASSES_121["ovary_left"], 119)
        self.assertEqual(TOTAL_CLASSES_121["ovary_right"], 120)
        self.assertEqual(TOTAL_CLASSES_121["vagina"], 121)
        self.assertEqual(TOTAL_CLASSES_121["prostate"], 22)

        # Test mask generation
        female_mask = get_sex_organ_mask(0, num_organs=121)
        male_mask = get_sex_organ_mask(1, num_organs=121)

        prostate_idx = ORGAN_NAMES.index("prostate")
        uterus_idx = ORGAN_NAMES.index("uterus")
        ovary_l_idx = ORGAN_NAMES.index("ovary_left")
        ovary_r_idx = ORGAN_NAMES.index("ovary_right")
        vagina_idx = ORGAN_NAMES.index("vagina")

        # Female: prostate is 0, female organs are 1
        self.assertEqual(female_mask[prostate_idx].item(), 0.0)
        self.assertEqual(female_mask[uterus_idx].item(), 1.0)
        self.assertEqual(female_mask[ovary_l_idx].item(), 1.0)
        self.assertEqual(female_mask[ovary_r_idx].item(), 1.0)
        self.assertEqual(female_mask[vagina_idx].item(), 1.0)

        # Male: prostate is 1, female organs are 0
        self.assertEqual(male_mask[prostate_idx].item(), 1.0)
        self.assertEqual(male_mask[uterus_idx].item(), 0.0)
        self.assertEqual(male_mask[ovary_l_idx].item(), 0.0)
        self.assertEqual(male_mask[ovary_r_idx].item(), 0.0)
        self.assertEqual(male_mask[vagina_idx].item(), 0.0)
        print("✓ Test 1 Passed: 121 Classes and Sex Organ Masks Verified.")

    def test_forward_pass_shape(self):
        """Test forward pass output shape is strictly (B, 121, 3)."""
        model = SexConditionedOrganGNN(
            num_organs=self.num_organs,
            pointnet_feat_dim=1024,
            sex_emb_dim=128,
            gnn_hidden_dim=256,
            gnn_layers=2, # Fast test with 2 layers
        ).to(self.device)

        model.eval()
        with torch.no_grad():
            pred_centroids = model(self.points, self.sex_priors)

        self.assertEqual(pred_centroids.shape, (self.batch_size, self.num_organs, 3))

        # Check that output for invalid organs is cleanly masked to 0
        prostate_idx = ORGAN_NAMES.index("prostate")
        uterus_idx = ORGAN_NAMES.index("uterus")

        # Batch 0 is Female -> prostate should be [0, 0, 0]
        self.assertTrue(torch.allclose(pred_centroids[0, prostate_idx], torch.zeros(3, device=self.device)))
        # Batch 1 is Male -> uterus should be [0, 0, 0]
        self.assertTrue(torch.allclose(pred_centroids[1, uterus_idx], torch.zeros(3, device=self.device)))
        print(f"✓ Test 2 Passed: Forward Pass Shape {tuple(pred_centroids.shape)} and Zero Outputs for Inactive Organs Verified.")

    def test_zero_gradients_on_masked_organs(self):
        """Test that loss and gradients are STRICTLY ZERO for masked-out organs."""
        criterion = DynamicMaskedCentroidLoss()

        # Variable predictions requiring gradient
        pred_centroids = torch.randn(self.batch_size, self.num_organs, 3, requires_grad=True, device=self.device)
        mask = get_sex_organ_mask(self.sex_priors, num_organs=self.num_organs, device=self.device)

        loss = criterion(pred_centroids, self.gt_centroids, mask=mask)
        loss.backward()

        self.assertIsNotNone(pred_centroids.grad)
        grad = pred_centroids.grad

        prostate_idx = ORGAN_NAMES.index("prostate")
        female_indices = FEMALE_SPECIFIC_INDICES

        # Case 0 is Female: gradient for prostate MUST BE ZERO
        self.assertTrue(torch.allclose(grad[0, prostate_idx], torch.zeros(3, device=self.device)),
                        f"Expected zero grad for prostate in female case, got: {grad[0, prostate_idx]}")
        # Valid organs in Female case must have non-zero grad
        self.assertGreater(torch.norm(grad[0, female_indices[0]]).item(), 0.0)

        # Case 1 is Male: gradient for female organs MUST BE ZERO
        for f_idx in female_indices:
            self.assertTrue(torch.allclose(grad[1, f_idx], torch.zeros(3, device=self.device)),
                            f"Expected zero grad for {ORGAN_NAMES[f_idx]} in male case, got: {grad[1, f_idx]}")
        # Valid prostate in Male case must have non-zero grad
        self.assertGreater(torch.norm(grad[1, prostate_idx]).item(), 0.0)

        print("✓ Test 3 Passed: Zero Gradients on Masked-Out Organs Backpropagation Verified.")

    def test_training_epoch_convergence(self):
        """Test running 1 full training epoch loop on point cloud dataset."""
        N_samples = 8
        synthetic_data = {
            "points": torch.randn(N_samples, self.num_points, 3),
            "centroids": torch.randn(N_samples, self.num_organs, 3) * 20.0,
            "sex_priors": torch.tensor([[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0] for i in range(N_samples)], dtype=torch.float32),
            "masks": torch.stack([get_sex_organ_mask(0 if i % 2 == 0 else 1, num_organs=self.num_organs) for i in range(N_samples)]),
            "case_ids": [f"case_{i:03d}" for i in range(N_samples)],
        }

        dataset = PointCloudOrganDataset(data_dict=synthetic_data, augment=True)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        model = SexConditionedOrganGNN(
            num_organs=self.num_organs,
            pointnet_feat_dim=1024,
            sex_emb_dim=128,
            gnn_hidden_dim=256,
            gnn_layers=2,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = DynamicMaskedCentroidLoss()

        model.train()
        losses = []
        for step, batch in enumerate(loader):
            pts = batch["points"].to(self.device)
            ctr = batch["centroids"].to(self.device)
            sex = batch["sex_prior"].to(self.device)
            mask = batch["mask"].to(self.device)

            optimizer.zero_grad()
            preds = model(pts, sex, mask=mask)
            loss = criterion(preds, ctr, mask=mask)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        self.assertFalse(torch.isnan(torch.tensor(avg_loss)).item())
        print(f"✓ Test 4 Passed: 1 Training Epoch Completed Successfully (Avg Loss: {avg_loss:.4f}).")


if __name__ == "__main__":
    unittest.main()
