import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Dict

try:
    from sharon.labels import (
        NUM_ORGANS, NUM_SKELETAL, NUM_SOFT_TISSUE,
        SKELETAL_INDICES, SOFT_TISSUE_INDICES,
        get_skeletal_mask, get_soft_tissue_mask
    )
except ImportError:
    from labels import (
        NUM_ORGANS, NUM_SKELETAL, NUM_SOFT_TISSUE,
        SKELETAL_INDICES, SOFT_TISSUE_INDICES,
        get_skeletal_mask, get_soft_tissue_mask
    )


class DynamicMaskedCentroidLoss(nn.Module):
    """
    Computes masked Smooth L1 loss between predicted and ground-truth centroids.
    """
    def __init__(self, beta: float = 0.01):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pred_centroids: torch.Tensor,
        gt_centroids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        diff = F.smooth_l1_loss(pred_centroids, gt_centroids, beta=self.beta, reduction="none")
        per_organ_loss = diff.sum(dim=-1)
        masked_loss = per_organ_loss * mask
        num_valid = mask.sum().clamp(min=1.0)
        return masked_loss.sum() / num_valid


class HierarchicalDynamicMaskedLoss(nn.Module):
    """
    Joint Multi-Task Loss for Hierarchical Coarse-to-Fine Point Cloud Regression.
    """
    def __init__(self, beta: float = 0.01, lambda_regional: float = 1.0, lambda_offset: float = 2.0):
        super().__init__()
        self.beta = beta
        self.lambda_regional = lambda_regional
        self.lambda_offset = lambda_offset
        self.centroid_loss = DynamicMaskedCentroidLoss(beta=beta)

    def forward(
        self,
        pred_centroids: torch.Tensor,
        gt_centroids: torch.Tensor,
        mask: torch.Tensor,
        pred_regions: Optional[torch.Tensor] = None,
        gt_regions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l_offset = self.centroid_loss(pred_centroids, gt_centroids, mask)
        if pred_regions is not None and gt_regions is not None:
            l_reg = F.smooth_l1_loss(pred_regions, gt_regions, beta=self.beta, reduction="mean")
            return self.lambda_regional * l_reg + self.lambda_offset * l_offset
        return l_offset


class GaussianNegativeLogLikelihoodLoss(nn.Module):
    """
    Numerically Stable Gaussian Negative Log-Likelihood (NLL) Loss with Cholesky Decomposition.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.const_term = 3.0 * math.log(2.0 * math.pi)

    def forward(
        self,
        mu: torch.Tensor,
        L: torch.Tensor,
        log_diag: torch.Tensor,
        y_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mu = mu.float()
        L = L.float()
        log_diag = log_diag.float()
        y_gt = y_gt.float()
        mask = mask.float()

        diff = (y_gt - mu).unsqueeze(-1)
        x = torch.linalg.solve_triangular(L, diff, upper=False)
        m_dist = torch.sum(x ** 2, dim=[-2, -1])
        log_det = 2.0 * torch.sum(log_diag, dim=-1)
        nll = 0.5 * (m_dist + log_det + self.const_term)

        masked_nll = nll * mask
        num_valid = mask.sum().clamp(min=1.0)
        return masked_nll.sum() / num_valid


class SAMeJointLoss(nn.Module):
    """
    Joint Multi-Task Loss for SAMe Architecture.
    """
    def __init__(self, beta: float = 0.01, lambda_skel: float = 1.0, lambda_soft: float = 0.5):
        super().__init__()
        self.lambda_skel = lambda_skel
        self.lambda_soft = lambda_soft
        self.skel_loss = DynamicMaskedCentroidLoss(beta=beta)
        self.nll_loss = GaussianNegativeLogLikelihoodLoss()

    def forward(
        self,
        same_output: Dict[str, torch.Tensor],
        gt_centroids: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_gt_skel = gt_centroids[:, SKELETAL_INDICES, :]
        y_gt_soft = gt_centroids[:, SOFT_TISSUE_INDICES, :]
        mask_skel = get_skeletal_mask(full_mask)
        mask_soft = get_soft_tissue_mask(full_mask)

        l_skel = self.skel_loss(same_output["skel_coords"], y_gt_skel, mask_skel)
        l_soft_nll = self.nll_loss(
            mu=same_output["soft_mu"],
            L=same_output["soft_L"],
            log_diag=same_output["soft_log_diag"],
            y_gt=y_gt_soft,
            mask=mask_soft,
        )
        total_loss = self.lambda_skel * l_skel + self.lambda_soft * l_soft_nll
        return total_loss, l_skel, l_soft_nll


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENTIAL DEEP LEARNING & BIOMECHANICAL NON-INTERPENETRATION LOSSES
# ─────────────────────────────────────────────────────────────────────────────

class DeepEvidentialRegressionLoss(nn.Module):
    """
    Deep Evidential Regression Loss (Amini et al., NeurIPS):
    Models continuous targets under a Normal-Inverse-Gamma distribution (gamma, nu, alpha, beta).
    
    L_NLL = 0.5*log(pi/nu) - alpha*log(beta) + (alpha + 0.5)*log(beta + nu*(y - gamma)^2 / 2) 
            + lgamma(alpha) - lgamma(alpha + 0.5)
    L_Reg = |y - gamma| * (2*nu + alpha)
    L_EDL = L_NLL + lambda_reg * L_Reg
    """
    def __init__(self, lambda_reg: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.eps = eps
        self.log_pi = math.log(math.pi)

    def forward(
        self,
        gamma: torch.Tensor,   # (B, 121, 3) predicted means
        nu: torch.Tensor,      # (B, 121, 3) evidence count (> 0)
        alpha: torch.Tensor,   # (B, 121, 3) shape parameter (> 1.0)
        beta: torch.Tensor,    # (B, 121, 3) scale parameter (> 0)
        y_gt: torch.Tensor,    # (B, 121, 3) ground truth coordinates
        mask: torch.Tensor,    # (B, 121) biological sex validity mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Cast to float32 for exact log-gamma stability
        gamma = gamma.float()
        nu = nu.float()
        alpha = alpha.float()
        beta = beta.float()
        y_gt = y_gt.float()
        mask = mask.float()

        # Coordinate error per axis: (B, 121, 3)
        err_sq = (y_gt - gamma) ** 2
        abs_err = torch.abs(y_gt - gamma)

        # 1. Negative Log-Likelihood of Student-t marginal: (B, 121, 3)
        term1 = 0.5 * (self.log_pi - torch.log(nu))
        term2 = -alpha * torch.log(beta)
        term3 = (alpha + 0.5) * torch.log(beta + 0.5 * nu * err_sq)
        term4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        l_nll_per_axis = term1 + term2 + term3 + term4 # (B, 121, 3)

        # 2. Evidential Regularizer to penalize overconfident incorrect predictions: (B, 121, 3)
        l_reg_per_axis = abs_err * (2.0 * nu + alpha)

        # Combined EDL loss per organ: (B, 121)
        l_nll_organ = torch.mean(l_nll_per_axis, dim=-1)
        l_reg_organ = torch.mean(l_reg_per_axis, dim=-1)
        l_edl_organ = l_nll_organ + self.lambda_reg * l_reg_organ

        # Biological Masking
        num_valid = mask.sum().clamp(min=1.0)
        loss_nll = (l_nll_organ * mask).sum() / num_valid
        loss_reg = (l_reg_organ * mask).sum() / num_valid
        loss_edl = (l_edl_organ * mask).sum() / num_valid

        return loss_edl, loss_nll, loss_reg


class BiomechanicalCollisionLoss(nn.Module):
    """
    Differentiable Biomechanical Non-Interpenetration Collision Loss:
    Penalizes physical volumetric overlap between any two distinct anatomical structures i != j:
        L_bio = sum_{i != j} m_i * m_j * max(0, (r_i + r_j) - ||gamma_i - gamma_j||_2)
    where r_i is the 3D epistemic spatial uncertainty radius of organ i.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        gamma: torch.Tensor,   # (B, 121, 3) predicted 3D organ centers
        radius: torch.Tensor,  # (B, 121) predicted 3D epistemic uncertainty radii
        mask: torch.Tensor,    # (B, 121) biological sex validity mask
    ) -> torch.Tensor:
        B, K, _ = gamma.shape
        gamma = gamma.float()
        radius = radius.float()
        mask = mask.float()

        # Pairwise Euclidean Distance Matrix between organ centers: (B, K, K)
        diff = gamma.unsqueeze(2) - gamma.unsqueeze(1) # (B, K, K, 3)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + self.eps) # (B, K, K)

        # Pairwise required separation distance: (B, K, K)
        sum_radii = radius.unsqueeze(2) + radius.unsqueeze(1) # (B, K, K)

        # Non-interpenetration overlap penetration depth: (B, K, K)
        overlap = F.relu(sum_radii - dist)

        # Zero out diagonal self-interpenetration (i == j)
        eye_mask = (1.0 - torch.eye(K, device=gamma.device, dtype=gamma.dtype)).unsqueeze(0) # (1, K, K)
        overlap = overlap * eye_mask

        # Pairwise biological sex validity mask: (B, K, K)
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1) * eye_mask

        num_pairs = pair_mask.sum().clamp(min=1.0)
        loss_bio = (overlap * pair_mask).sum() / num_pairs
        return loss_bio


class EvidentialJointLoss(nn.Module):
    """
    Joint Multi-Task Loss for Evidential Biomechanical GNN:
        L_total = L_EDL(gamma, nu, alpha, beta, y_gt, mask) + lambda_bio * L_bio(gamma, radius, mask)
    """
    def __init__(
        self,
        lambda_reg: float = 0.1,
        lambda_bio: float = 5.0,
    ):
        super().__init__()
        self.lambda_bio = lambda_bio
        self.edl_loss = DeepEvidentialRegressionLoss(lambda_reg=lambda_reg)
        self.bio_loss = BiomechanicalCollisionLoss()

    def forward(
        self,
        evidential_output: Dict[str, torch.Tensor],
        gt_centroids: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss: scalar tensor
            l_edl:      scalar tensor (EDL loss)
            l_nll:      scalar tensor (Student-t NLL)
            l_bio:      scalar tensor (Biomechanical collision penalty)
        """
        gamma = evidential_output["gamma"]
        nu = evidential_output["nu"]
        alpha = evidential_output["alpha"]
        beta = evidential_output["beta"]
        radius = evidential_output["radius"]

        l_edl, l_nll, l_reg = self.edl_loss(gamma, nu, alpha, beta, gt_centroids, mask)
        l_bio = self.bio_loss(gamma, radius, mask)

        total_loss = l_edl + self.lambda_bio * l_bio
        return total_loss, l_edl, l_nll, l_bio
