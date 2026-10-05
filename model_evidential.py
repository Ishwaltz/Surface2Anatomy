import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

try:
    from sharon.labels import (
        NUM_ORGANS, NUM_REGIONS, ORGAN_NAMES,
        get_sex_organ_mask, get_organ_region_indices
    )
    from sharon.model_gnn import PointNet2Encoder, SexEncoder, DynamicMaskedGNNLayer
except ImportError:
    from labels import (
        NUM_ORGANS, NUM_REGIONS, ORGAN_NAMES,
        get_sex_organ_mask, get_organ_region_indices
    )
    from model_gnn import PointNet2Encoder, SexEncoder, DynamicMaskedGNNLayer


class EvidentialOrganGNN(nn.Module):
    """
    Evidential Dynamic GNN for 3D Organ Location Prediction.
    
    For each of the K=121 organs across 3 spatial dimensions (x, y, z), outputs
    the 4 parameters of the Normal-Inverse-Gamma distribution:
        - gamma in R       : Expected coordinate location
        - nu > 0           : Epistemic evidence confidence (virtual observation count)
        - alpha > 1.0      : Shape parameter
        - beta > 0         : Scale parameter
        
    Enables exact decomposition into:
        - Aleatoric Uncertainty: sigma_ale = sqrt(beta / (alpha - 1))
        - Epistemic Uncertainty: sigma_epi = sqrt(beta / (nu * (alpha - 1)))
        - Epistemic Volumetric Radius: r_i = sqrt(1/3 * sum(beta / (nu * (alpha - 1))))
    """
    def __init__(
        self,
        num_organs: int = NUM_ORGANS,
        pointnet_feat_dim: int = 1024,
        sex_emb_dim: int = 128,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        gnn_layers: int = 4,
    ):
        super().__init__()
        self.num_organs = num_organs

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

        # 121 Learnable Organ Queries
        self.organ_queries = nn.Parameter(torch.randn(num_organs, hidden_dim) * 0.02)

        # Query Contextual Projection
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 4-Layer Dynamic Masked Inter-Organ GNN
        self.gnn_layers = nn.ModuleList([DynamicMaskedGNNLayer(hidden_dim=hidden_dim) for _ in range(gnn_layers)])

        # Dense Evidential Regression Head: outputs 12 parameters per organ (4 per axis x, y, z)
        self.evidential_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3 * 4),  # 3 spatial axes * 4 NIG parameters (gamma, nu, alpha, beta)
        )

    def forward(
        self,
        points: torch.Tensor,
        sex_prior: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            points:    (B, 4096, 3) normalized patient surface point clouds
            sex_prior: (B, 2) or (B,) biological sex prior
            mask:      (B, 121) biological sex validity mask
        Returns:
            dict containing:
                "gamma":         (B, 121, 3) predicted mean coordinates in [-1, 1]^3
                "nu":            (B, 121, 3) epistemic evidence confidence (> 0)
                "alpha":         (B, 121, 3) NIG shape parameter (> 1.0)
                "beta":          (B, 121, 3) NIG scale parameter (> 0)
                "radius":        (B, 121) 3D epistemic spatial uncertainty radius
                "aleatoric_var": (B, 121, 3) aleatoric data noise variance
                "epistemic_var": (B, 121, 3) epistemic model uncertainty variance
                "full_coords":   (B, 121, 3) masked canonical coordinate predictions
        """
        B = points.shape[0]
        device = points.device

        if mask is None:
            mask = get_sex_organ_mask(sex_prior, num_organs=self.num_organs, device=device)

        # 1. Encoders & Patient Latent Fusion
        f_geo = self.point_encoder(points)
        f_sex = self.sex_encoder(sex_prior)
        z = self.fusion(torch.cat([f_geo, f_sex], dim=-1)) # (B, 512)

        # 2. Initialize Organ Query Graph
        q_exp = self.organ_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 121, hidden_dim)
        z_exp = z.unsqueeze(1).expand(-1, self.num_organs, -1)      # (B, 121, latent_dim)
        h = self.query_proj(torch.cat([q_exp, z_exp], dim=-1))     # (B, 121, hidden_dim)

        # 3. Dynamic Masked GNN Message Passing
        for layer in self.gnn_layers:
            h = layer(h, mask)

        # 4. Evidential Parameter Prediction
        raw = self.evidential_head(h).view(B, self.num_organs, 3, 4) # (B, 121, 3, 4)

        # Strict Parameter Parameterization:
        # gamma: bounded predicted coordinate
        gamma = torch.tanh(raw[:, :, :, 0]) # (B, 121, 3) in [-1.0, 1.0]

        # nu > 0 (virtual observations / evidence confidence)
        nu = F.softplus(raw[:, :, :, 1]) + 1e-4 # (B, 121, 3)

        # alpha > 1.0 (gamma shape parameter)
        alpha = F.softplus(raw[:, :, :, 2]) + 1.0 + 1e-4 # (B, 121, 3)

        # beta > 0 (gamma scale parameter)
        beta = F.softplus(raw[:, :, :, 3]) + 1e-4 # (B, 121, 3)

        # 5. Uncertainty Decompositions
        # Aleatoric Variance: Var[y] = beta / (alpha - 1)
        aleatoric_var = beta / (alpha - 1.0) # (B, 121, 3)

        # Epistemic Variance: Var[mu] = beta / (nu * (alpha - 1))
        epistemic_var = beta / (nu * (alpha - 1.0)) # (B, 121, 3)

        # 3D Epistemic Spatial Uncertainty Radius: sqrt(1/3 * sum_k Var[mu_k])
        radius = torch.sqrt(torch.mean(epistemic_var, dim=-1) + 1e-8) # (B, 121)

        # Apply biological sex masking to coordinates
        gamma = gamma * mask.unsqueeze(-1)

        return {
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "radius": radius,
            "aleatoric_var": aleatoric_var,
            "epistemic_var": epistemic_var,
            "full_coords": gamma,
        }
