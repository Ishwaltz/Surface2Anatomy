import unittest
import torch
import numpy as np

from labels import (
    NUM_ORGANS, ORGAN_NAMES, get_sex_organ_mask
)
from model_evidential import EvidentialOrganGNN
from losses import DeepEvidentialRegressionLoss, BiomechanicalCollisionLoss, EvidentialJointLoss


class TestEvidentialPipeline(unittest.TestCase):
    def test_evidential_model_forward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EvidentialOrganGNN(
            pointnet_feat_dim=512,
            sex_emb_dim=64,
            latent_dim=256,
            hidden_dim=128,
            gnn_layers=2,
        ).to(device)

        B = 2
        pts = torch.randn(B, 4096, 3, device=device).clamp(-1.0, 1.0)
        sex = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
        mask = get_sex_organ_mask(sex, num_organs=121, device=device)

        out = model(pts, sex, mask=mask)
        gamma = out["gamma"]
        nu = out["nu"]
        alpha = out["alpha"]
        beta = out["beta"]
        radius = out["radius"]

        self.assertEqual(gamma.shape, (B, 121, 3))
        self.assertEqual(nu.shape, (B, 121, 3))
        self.assertEqual(alpha.shape, (B, 121, 3))
        self.assertEqual(beta.shape, (B, 121, 3))
        self.assertEqual(radius.shape, (B, 121))

        # Constraint verification
        self.assertTrue(torch.all(nu > 0))
        self.assertTrue(torch.all(alpha > 1.0))
        self.assertTrue(torch.all(beta > 0))
        self.assertTrue(torch.all(radius > 0))
        self.assertTrue(torch.all(gamma >= -1.0) and torch.all(gamma <= 1.0))

        # Biological Masking Check
        uterus_idx = ORGAN_NAMES.index("uterus")
        self.assertAlmostEqual(gamma[1, uterus_idx].abs().sum().item(), 0.0, places=5)

    def test_evidential_loss_and_gradients(self):
        B = 2
        gamma = torch.randn(B, 121, 3, requires_grad=True)
        nu = (torch.rand(B, 121, 3) + 0.5).requires_grad_()
        alpha = (torch.rand(B, 121, 3) + 2.0).requires_grad_()
        beta = (torch.rand(B, 121, 3) + 0.2).requires_grad_()
        y_gt = torch.randn(B, 121, 3)
        mask = torch.ones(B, 121)

        criterion = DeepEvidentialRegressionLoss(lambda_reg=0.1)
        loss_edl, loss_nll, loss_reg = criterion(gamma, nu, alpha, beta, y_gt, mask)

        self.assertFalse(torch.isnan(loss_edl))
        self.assertFalse(torch.isnan(loss_nll))
        self.assertFalse(torch.isnan(loss_reg))

        loss_edl.backward()
        self.assertIsNotNone(gamma.grad)
        self.assertFalse(torch.isnan(gamma.grad).any())
        self.assertIsNotNone(nu.grad)
        self.assertFalse(torch.isnan(nu.grad).any())
        self.assertIsNotNone(alpha.grad)
        self.assertFalse(torch.isnan(alpha.grad).any())
        self.assertIsNotNone(beta.grad)
        self.assertFalse(torch.isnan(beta.grad).any())

    def test_biomechanical_collision_loss(self):
        B = 1
        # Two organs right on top of each other
        gamma = torch.zeros(B, 121, 3)
        for i in range(121):
            gamma[0, i] = torch.tensor([float(i) * 10.0, 0.0, 0.0])
        # Place organ 1 at 0.05 from organ 0
        gamma[0, 1] = torch.tensor([0.05, 0.0, 0.0])

        radius = torch.ones(B, 121) * 0.1
        mask = torch.ones(B, 121)

        bio_loss_fn = BiomechanicalCollisionLoss()
        loss = bio_loss_fn(gamma, radius, mask)
        self.assertTrue(loss.item() > 0)

        # Separate organ 1 far away
        gamma[0, 1] = torch.tensor([10.0, 0.0, 0.0])
        loss_separated = bio_loss_fn(gamma, radius, mask)
        self.assertAlmostEqual(loss_separated.item(), 0.0, places=5)

    def test_joint_evidential_loss_step(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EvidentialOrganGNN(
            pointnet_feat_dim=512,
            sex_emb_dim=64,
            latent_dim=256,
            hidden_dim=128,
            gnn_layers=2,
        ).to(device)

        B = 2
        pts = torch.randn(B, 4096, 3, device=device)
        sex = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
        mask = get_sex_organ_mask(sex, num_organs=121, device=device)
        y_gt = torch.randn(B, 121, 3, device=device)

        out = model(pts, sex, mask=mask)
        criterion = EvidentialJointLoss(lambda_reg=0.1, lambda_bio=5.0)
        total_loss, l_edl, l_nll, l_bio = criterion(out, y_gt, mask)

        self.assertFalse(torch.isnan(total_loss))
        total_loss.backward()


if __name__ == "__main__":
    unittest.main()
