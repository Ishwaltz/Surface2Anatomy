import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

try:
    from sharon.labels import (
        NUM_ORGANS, NUM_SKELETAL, NUM_SOFT_TISSUE,
        SKELETAL_INDICES, SOFT_TISSUE_INDICES,
        get_sex_organ_mask, get_skeletal_mask, get_soft_tissue_mask,
        combine_skeletal_and_soft_predictions
    )
    from sharon.model_gnn import PointNet2Encoder, SexEncoder, DynamicMaskedGNNLayer
except ImportError:
    from labels import (
        NUM_ORGANS, NUM_SKELETAL, NUM_SOFT_TISSUE,
        SKELETAL_INDICES, SOFT_TISSUE_INDICES,
        get_sex_organ_mask, get_skeletal_mask, get_soft_tissue_mask,
        combine_skeletal_and_soft_predictions
    )
    from model_gnn import PointNet2Encoder, SexEncoder, DynamicMaskedGNNLayer


class SkeletalInstantiationHead(nn.Module):
    """
    Stage 1: Regresses deterministic 3D physical coordinates exclusively
    for the 63 SKELETAL_CLASSES (vertebrae, ribs, pelvis, skull, sacrum, etc.)
    using the global patient latent embedding.
    """
    def __init__(self, latent_dim: int = 512, hidden_dim: int = 256, num_skeletal: int = NUM_SKELETAL):
        super().__init__()
        self.num_skeletal = num_skeletal
        self.skel_queries = nn.Parameter(torch.randn(num_skeletal, hidden_dim) * 0.02)
        
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Tanh(),  # strictly bounded within [-1.0, 1.0]
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, latent_dim) patient latent feature
        Returns:
            y_skel: (B, 63, 3) normalized coordinates in [-1, 1]^3
            h_skel: (B, 63, hidden_dim) skeletal feature embeddings
        """
        B = z.shape[0]
        q_exp = self.skel_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 63, hidden_dim)
        z_exp = z.unsqueeze(1).expand(-1, self.num_skeletal, -1)  # (B, 63, latent_dim)
        
        h_skel = self.node_proj(torch.cat([q_exp, z_exp], dim=-1)) # (B, 63, hidden_dim)
        y_skel = self.coord_mlp(h_skel)                            # (B, 63, 3)
        return y_skel, h_skel


class SkeletonConditionedSoftTissueGNN(nn.Module):
    """
    Stage 2: Conditions on Stage 1 skeletal coordinates and regresses
    Multivariate Gaussian distributions (mean mu in R^3, Cholesky lower-triangular L in R^(3x3))
    for all 58 SOFT_TISSUE_CLASSES via dynamic sex-masked message passing.
    """
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        num_skeletal: int = NUM_SKELETAL,
        num_soft: int = NUM_SOFT_TISSUE,
        gnn_layers: int = 4,
    ):
        super().__init__()
        self.num_skeletal = num_skeletal
        self.num_soft = num_soft

        # Soft tissue learnable query embeddings
        self.soft_queries = nn.Parameter(torch.randn(num_soft, hidden_dim) * 0.02)

        # Soft tissue node initialization
        self.soft_init = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Skeletal node geo-contextual refinement
        self.skel_refine = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Inter-Organ Dynamic Masked Message Passing Layers (N=121 total nodes)
        self.gnn_layers = nn.ModuleList([DynamicMaskedGNNLayer(hidden_dim=hidden_dim) for _ in range(gnn_layers)])

        # Soft-Tissue Distribution Head: Mean (3) + Lower-Triangular Cholesky Factors (6)
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3 + 6),  # 3 mean coords + 3 log-diagonal scales + 3 off-diagonal elements
        )

    def forward(
        self,
        h_skel: torch.Tensor,
        y_skel: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_skel: (B, 63, hidden_dim) skeletal feature embeddings from Stage 1
            y_skel: (B, 63, 3) predicted skeletal coordinates
            z:      (B, latent_dim) patient latent feature
            mask:   (B, 121) full biological sex mask
        Returns:
            mu:       (B, 58, 3) soft-tissue mean coordinates in [-1, 1]^3
            L:        (B, 58, 3, 3) lower-triangular Cholesky covariance factors
            log_diag: (B, 58, 3) log-diagonal scales
            Sigma:    (B, 58, 3, 3) reconstructed covariance matrices (L L^T)
        """
        B = z.shape[0]
        device = z.device

        # 1. Initialize soft-tissue nodes
        q_soft = self.soft_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 58, hidden_dim)
        z_soft = z.unsqueeze(1).expand(-1, self.num_soft, -1)      # (B, 58, latent_dim)
        h_soft = self.soft_init(torch.cat([q_soft, z_soft], dim=-1)) # (B, 58, hidden_dim)

        # 2. Refine skeletal nodes with predicted coordinates
        h_skel_geo = self.skel_refine(torch.cat([h_skel, y_skel], dim=-1)) # (B, 63, hidden_dim)

        # 3. Assemble full 121-node graph: [63 skeletal || 58 soft-tissue]
        # Re-order mask to match [skeletal || soft]
        mask_skel = get_skeletal_mask(mask)       # (B, 63)
        mask_soft = get_soft_tissue_mask(mask)   # (B, 58)
        mask_ordered = torch.cat([mask_skel, mask_soft], dim=1) # (B, 121)

        h_full = torch.cat([h_skel_geo, h_soft], dim=1) # (B, 121, hidden_dim)

        # 4. Dynamic Message Passing across full anatomical graph
        for layer in self.gnn_layers:
            h_full = layer(h_full, mask_ordered)

        # 5. Extract soft-tissue representations: last 58 nodes
        h_soft_final = h_full[:, self.num_skeletal:, :] # (B, 58, hidden_dim)

        # 6. Predict Multivariate Gaussian Parameters
        raw_params = self.dist_head(h_soft_final) # (B, 58, 9)
        mu = raw_params[:, :, 0:3]                 # (B, 58, 3)

        # Cholesky Parameterization:
        # Diagonal elements: strictly positive via exp(clamp(d, -6, 3))
        # This bounds standard deviations in [0.002, 20.0] normalized space
        log_diag = torch.clamp(raw_params[:, :, 3:6], min=-6.0, max=3.0) # (B, 58, 3)
        diag = torch.exp(log_diag)                                       # (B, 58, 3)

        # Off-diagonal lower-triangular elements
        l_21 = raw_params[:, :, 6] # (B, 58)
        l_31 = raw_params[:, :, 7] # (B, 58)
        l_32 = raw_params[:, :, 8] # (B, 58)

        # Build lower-triangular matrix L: (B, 58, 3, 3)
        L = torch.zeros((B, self.num_soft, 3, 3), dtype=z.dtype, device=device)
        L[:, :, 0, 0] = diag[:, :, 0]
        L[:, :, 1, 0] = l_21
        L[:, :, 1, 1] = diag[:, :, 1]
        L[:, :, 2, 0] = l_31
        L[:, :, 2, 1] = l_32
        L[:, :, 2, 2] = diag[:, :, 2]

        # Reconstructed strictly positive semi-definite Covariance Sigma = L L^T
        Sigma = torch.matmul(L, L.transpose(-1, -2)) # (B, 58, 3, 3)

        # Apply biological masking to output means
        mu = mu * mask_soft.unsqueeze(-1)
        return mu, L, log_diag, Sigma


class SAMeOrganGNN(nn.Module):
    """
    SAMe: Skeleton-Conditioned Probabilistic Graph Neural Network.
    - Stage 1: Deterministic Skeletal Instantiation (63 classes).
    - Stage 2: Skeleton-Conditioned Probabilistic GNN with Cholesky Covariance (58 classes).
    """
    def __init__(
        self,
        pointnet_feat_dim: int = 1024,
        sex_emb_dim: int = 128,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        gnn_layers: int = 4,
    ):
        super().__init__()
        self.num_organs = NUM_ORGANS
        self.num_skeletal = NUM_SKELETAL
        self.num_soft = NUM_SOFT_TISSUE

        # Geometry & Sex Encoders
        self.point_encoder = PointNet2Encoder(in_channel=3, out_dim=pointnet_feat_dim)
        self.sex_encoder = SexEncoder(out_dim=sex_emb_dim)

        # Patient Latent Fusion
        self.fusion = nn.Sequential(
            nn.Linear(pointnet_feat_dim + sex_emb_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
        )

        # Stage 1: Skeletal Head
        self.skeletal_head = SkeletalInstantiationHead(latent_dim=latent_dim, hidden_dim=hidden_dim, num_skeletal=NUM_SKELETAL)

        # Stage 2: Soft-Tissue Probabilistic GNN
        self.soft_gnn = SkeletonConditionedSoftTissueGNN(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_skeletal=NUM_SKELETAL,
            num_soft=NUM_SOFT_TISSUE,
            gnn_layers=gnn_layers,
        )

    def forward(
        self,
        points: torch.Tensor,
        sex_prior: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            points:    (B, 4096, 3) normalized patient point clouds
            sex_prior: (B, 2) or (B,) one-hot/scalar sex prior
            mask:      (B, 121) biological sex validity mask
        Returns:
            dict containing:
                "skel_coords":   (B, 63, 3) deterministic skeletal predictions
                "soft_mu":       (B, 58, 3) soft-tissue mean coordinates
                "soft_L":        (B, 58, 3, 3) Cholesky lower-triangular factors
                "soft_log_diag": (B, 58, 3) log-diagonal scale parameters
                "soft_Sigma":    (B, 58, 3, 3) positive-definite covariance matrices
                "full_coords":   (B, 121, 3) combined canonical 121-organ coordinate predictions
                "vol_uncertainty": (B, 58) volumetric uncertainty determinant |Sigma|
        """
        device = points.device
        if mask is None:
            mask = get_sex_organ_mask(sex_prior, num_organs=self.num_organs, device=device)

        # 1. Encoders
        f_geo = self.point_encoder(points)
        f_sex = self.sex_encoder(sex_prior)
        z = self.fusion(torch.cat([f_geo, f_sex], dim=-1)) # (B, 512)

        # 2. Stage 1: Skeletal Instantiation
        y_skel, h_skel = self.skeletal_head(z) # (B, 63, 3), (B, 63, 256)

        # 3. Stage 2: Soft-Tissue Probabilistic GNN
        mu_soft, L_soft, log_diag_soft, Sigma_soft = self.soft_gnn(h_skel, y_skel, z, mask)

        # 4. Full Canonical 121-Organ Reconstruction
        full_coords = combine_skeletal_and_soft_predictions(y_skel, mu_soft)
        full_coords = full_coords * mask.unsqueeze(-1)

        # 5. Volumetric Uncertainty Determinant |Sigma| = exp(2 * sum(log_diag))
        vol_uncertainty = torch.exp(2.0 * log_diag_soft.sum(dim=-1)) # (B, 58)

        return {
            "skel_coords": y_skel,
            "soft_mu": mu_soft,
            "soft_L": L_soft,
            "soft_log_diag": log_diag_soft,
            "soft_Sigma": Sigma_soft,
            "full_coords": full_coords,
            "vol_uncertainty": vol_uncertainty,
        }
