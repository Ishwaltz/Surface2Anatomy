import unittest
import torch
import numpy as np

from labels import (
    NUM_ORGANS, NUM_SKELETAL, NUM_SOFT_TISSUE,
    SKELETAL_CLASSES, SOFT_TISSUE_CLASSES,
    SKELETAL_INDICES, SOFT_TISSUE_INDICES,
    ORGAN_NAMES, get_sex_organ_mask,
    get_skeletal_indices, get_soft_tissue_indices,
    get_skeletal_mask, get_soft_tissue_mask,
    combine_skeletal_and_soft_predictions
)
from model_same import SAMeOrganGNN
from losses import GaussianNegativeLogLikelihoodLoss, SAMeJointLoss


class TestSAMePipeline(unittest.TestCase):
    def test_partition_integrity(self):
        self.assertEqual(len(SKELETAL_CLASSES), 63)
        self.assertEqual(len(SOFT_TISSUE_CLASSES), 58)
        self.assertEqual(len(SKELETAL_CLASSES) + len(SOFT_TISSUE_CLASSES), 121)

        # Disjoint check
        overlap = set(SKELETAL_CLASSES).intersection(set(SOFT_TISSUE_CLASSES))
        self.assertEqual(len(overlap), 0)

        # Union check
        all_union = set(SKELETAL_CLASSES).union(set(SOFT_TISSUE_CLASSES))
        self.assertEqual(all_union, set(ORGAN_NAMES))

    def test_combine_skeletal_and_soft_predictions(self):
        B = 2
        y_skel = torch.ones(B, 63, 3) * 1.0
        mu_soft = torch.ones(B, 58, 3) * 2.0
        full = combine_skeletal_and_soft_predictions(y_skel, mu_soft)
        self.assertEqual(full.shape, (B, 121, 3))

        for idx in SKELETAL_INDICES:
            self.assertTrue(torch.allclose(full[:, idx, :], torch.ones(B, 3) * 1.0))
        for idx in SOFT_TISSUE_INDICES:
            self.assertTrue(torch.allclose(full[:, idx, :], torch.ones(B, 3) * 2.0))

    def test_cholesky_positive_definiteness(self):
        B, N = 2, 58
        diag = torch.tensor([0.5, 1.2, 0.8]).view(1, 1, 3).expand(B, N, 3)
        log_diag = torch.log(diag)
        l21, l31, l32 = 0.2, -0.1, 0.3

        L = torch.zeros(B, N, 3, 3)
        L[:, :, 0, 0] = diag[:, :, 0]
        L[:, :, 1, 0] = l21
        L[:, :, 1, 1] = diag[:, :, 1]
        L[:, :, 2, 0] = l31
        L[:, :, 2, 1] = l32
        L[:, :, 2, 2] = diag[:, :, 2]

        Sigma = torch.matmul(L, L.transpose(-1, -2))
        
        # Verify all eigenvalues are positive
        eigvals = torch.linalg.eigvalsh(Sigma)
        self.assertTrue(torch.all(eigvals > 0))

        # Verify determinant calculation
        det_exact = torch.linalg.det(Sigma)
        det_formula = torch.exp(2.0 * log_diag.sum(dim=-1))
        self.assertTrue(torch.allclose(det_exact, det_formula, atol=1e-5))

    def test_gaussian_nll_solver_and_gradients(self):
        B, N = 4, 58
        mu = torch.randn(B, N, 3, requires_grad=True)
        log_diag = torch.randn(B, N, 3, requires_grad=True)
        L_off = torch.randn(B, N, 3, requires_grad=True)

        diag = torch.exp(log_diag)
        L = torch.zeros(B, N, 3, 3)
        L[:, :, 0, 0] = diag[:, :, 0]
        L[:, :, 1, 0] = L_off[:, :, 0]
        L[:, :, 1, 1] = diag[:, :, 1]
        L[:, :, 2, 0] = L_off[:, :, 1]
        L[:, :, 2, 1] = L_off[:, :, 2]
        L[:, :, 2, 2] = diag[:, :, 2]

        y_gt = torch.randn(B, N, 3)
        mask = torch.ones(B, N)

        criterion = GaussianNegativeLogLikelihoodLoss()
        loss = criterion(mu, L, log_diag, y_gt, mask)
        
        self.assertFalse(torch.isnan(loss))
        self.assertTrue(loss.item() > 0)
        
        loss.backward()
        self.assertIsNotNone(mu.grad)
        self.assertFalse(torch.isnan(mu.grad).any())
        self.assertIsNotNone(log_diag.grad)
        self.assertFalse(torch.isnan(log_diag.grad).any())
        self.assertIsNotNone(L_off.grad)
        self.assertFalse(torch.isnan(L_off.grad).any())

    def test_same_forward_and_loss(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SAMeOrganGNN(
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
        self.assertEqual(out["skel_coords"].shape, (B, 63, 3))
        self.assertEqual(out["soft_mu"].shape, (B, 58, 3))
        self.assertEqual(out["soft_L"].shape, (B, 58, 3, 3))
        self.assertEqual(out["soft_Sigma"].shape, (B, 58, 3, 3))
        self.assertEqual(out["full_coords"].shape, (B, 121, 3))
        self.assertEqual(out["vol_uncertainty"].shape, (B, 58))

        # Check male batch 1 female organ mask
        uterus_idx = ORGAN_NAMES.index("uterus")
        self.assertAlmostEqual(out["full_coords"][1, uterus_idx].abs().sum().item(), 0.0, places=5)

        # Joint Loss
        criterion = SAMeJointLoss()
        gt = torch.randn(B, 121, 3, device=device)
        loss, l_skel, l_soft = criterion(out, gt, mask)
        
        self.assertFalse(torch.isnan(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
