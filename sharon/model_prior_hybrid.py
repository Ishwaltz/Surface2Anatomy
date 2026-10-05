import math
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from sharon.model_target_query import MultiScaleSurfacePointNet2Encoder

class SurfaceToLatentModel(nn.Module):
    """
    Model A3: Surface-to-Latent Prior-Only Architecture.
    Predicts patient-specific anatomical deformation coordinates z_hat in R^d
    from the external body surface alone, then reconstructs all primary landmarks:
        P_hat = A + U z_hat
    
    A and U are fixed buffers estimated strictly from the training cohort.
    No CT voxels, no internal masks at inference. Strictly surface-only.
    """
    def __init__(
        self,
        A_mean: torch.Tensor,       # (K, 3) in mm
        U_basis: torch.Tensor,      # (3K, d)
        latent_dim: int = 16,
        use_metadata: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_targets = A_mean.shape[0]
        self.latent_dim = latent_dim
        self.use_metadata = use_metadata

        # Register anatomical basis as frozen buffers
        self.register_buffer("A_mean", A_mean.clone().float()) # (K, 3)
        self.register_buffer("U_basis", U_basis.clone().float()) # (3K, d)

        # 1. Multi-scale PointNet++ surface encoder
        self.encoder = MultiScaleSurfacePointNet2Encoder(in_channel=3, out_dim=1024)

        # 2. Trunk input features: global_feat (1024) + pooled l2 (256) + pooled l3 (512) = 1792
        in_dim = 1024 + 256 + 512
        if use_metadata:
            in_dim += 6 # metadata features (sex, age, BMI, etc.)
        in_dim += 3     # body dimensions (width, depth, height in mm)

        # 3. Global anatomy latent trunk
        self.latent_mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(
        self,
        pts: torch.Tensor,                          # (B, N, 3)
        metadata: Optional[torch.Tensor] = None,    # (B, 6)
        body_dimensions: Optional[torch.Tensor] = None # (B, 3)
    ) -> Dict[str, torch.Tensor]:
        B = pts.shape[0]

        # 1. Extract surface features
        enc_out = self.encoder(pts)
        f_global = enc_out["global_feat"]            # (B, 1024)
        f_l2_pool = enc_out["l2_feat"].mean(dim=1)  # (B, 256)
        f_l3_pool = enc_out["l3_feat"].mean(dim=1)  # (B, 512)

        feat_list = [f_global, f_l2_pool, f_l3_pool]
        if self.use_metadata and metadata is not None:
            feat_list.append(metadata)
        elif self.use_metadata:
            feat_list.append(torch.zeros(B, 6, device=pts.device))

        if body_dimensions is not None:
            # Normalize body dimensions (roughly ~500mm scale)
            b_norm = body_dimensions / 500.0
            feat_list.append(b_norm)
        else:
            feat_list.append(torch.zeros(B, 3, device=pts.device))

        c = torch.cat(feat_list, dim=-1) # (B, in_dim)

        # 2. Predict latent deformation code z_hat in R^d
        z_hat = self.latent_mlp(c) # (B, d)

        # 3. Reconstruct landmark coordinates: P_hat = A + U z_hat
        # U_basis: (3K, d), z_hat.T: (d, B) -> (3K, B) -> (B, 3K) -> (B, K, 3)
        y_recon = torch.matmul(z_hat, self.U_basis.t()) # (B, 3K)
        y_recon = y_recon.view(B, self.num_targets, 3)  # (B, K, 3)
        p_prior = self.A_mean.unsqueeze(0) + y_recon     # (B, K, 3) in mm

        return {
            "z_hat": z_hat,
            "p_prior": p_prior,
            "p_final": p_prior
        }


class HybridPriorLocalModel(nn.Module):
    """
    Model A4: End-to-End Hybrid Architecture.
    Combines:
    1. Surface-to-latent global anatomical deformation prior: P_prior = A + U z_hat
    2. Target-query cross-attention decoder predicting local refinement residual: Delta P_local
    3. Adaptive gating mechanism:
       P_final = P_prior + (1 - alpha_k) * Delta P_local
       where alpha_k in [0, 1] is learned per target.
    
    Trained end-to-end with target localization loss and latent supervision.
    """
    def __init__(
        self,
        A_mean: torch.Tensor,       # (K, 3) in mm
        U_basis: torch.Tensor,      # (3K, d)
        latent_dim: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        use_metadata: bool = True,
        dropout: float = 0.05
    ):
        super().__init__()
        self.num_targets = A_mean.shape[0]
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_metadata = use_metadata

        # Register anatomical basis as frozen buffers
        self.register_buffer("A_mean", A_mean.clone().float())   # (K, 3)
        self.register_buffer("U_basis", U_basis.clone().float()) # (3K, d)

        # 1. Surface Encoder
        self.encoder = MultiScaleSurfacePointNet2Encoder(in_channel=3, out_dim=1024)

        # 2. Global Latent Deformation Trunk
        in_dim = 1024 + 256 + 512
        if use_metadata:
            in_dim += 6
        in_dim += 3 # body dimensions

        self.latent_mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # 3. Local Target-Query Decoder
        self.proj_l2 = nn.Linear(256, d_model)
        self.proj_l3 = nn.Linear(512, d_model)
        self.scale_embed = nn.Embedding(2, d_model)
        self.pe_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.target_embed = nn.Embedding(self.num_targets, d_model)
        self.atlas_pe_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        if use_metadata:
            self.meta_mlp = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, d_model)
            )

        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms_self = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            )
            for _ in range(num_layers)
        ])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # Local residual offset head (in mm, initial scale small)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        # Initialize residual head with small weights so training starts near prior
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        # Learned target gating head: alpha_k in [0, 1]
        self.gate_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        pts: torch.Tensor,                          # (B, N, 3) in model coordinates
        metadata: Optional[torch.Tensor] = None,    # (B, 6)
        body_dimensions: Optional[torch.Tensor] = None # (B, 3) in mm
    ) -> Dict[str, torch.Tensor]:
        B = pts.shape[0]
        K = self.num_targets

        # 1. Surface Encoder features
        enc_out = self.encoder(pts)
        f_global = enc_out["global_feat"]            # (B, 1024)
        f_l2 = enc_out["l2_feat"]                    # (B, 256, 256)
        f_l3 = enc_out["l3_feat"]                    # (B, 64, 512)
        l2_xyz = enc_out["l2_xyz"]                   # (B, 256, 3)
        l3_xyz = enc_out["l3_xyz"]                   # (B, 64, 3)

        # 2. Global Latent Deformation Trunk
        f_l2_pool = f_l2.mean(dim=1)
        f_l3_pool = f_l3.mean(dim=1)
        feat_list = [f_global, f_l2_pool, f_l3_pool]
        if self.use_metadata and metadata is not None:
            feat_list.append(metadata)
        elif self.use_metadata:
            feat_list.append(torch.zeros(B, 6, device=pts.device))

        if body_dimensions is not None:
            feat_list.append(body_dimensions / 500.0)
        else:
            feat_list.append(torch.zeros(B, 3, device=pts.device))

        c = torch.cat(feat_list, dim=-1)
        z_hat = self.latent_mlp(c) # (B, d)

        # Global prior reconstruction in mm
        y_recon = torch.matmul(z_hat, self.U_basis.t()).view(B, K, 3) # (B, K, 3)
        p_prior = self.A_mean.unsqueeze(0) + y_recon                   # (B, K, 3) in mm

        # 3. Local Target-Query Decoder
        t_l2 = self.proj_l2(f_l2) + self.scale_embed.weight[0]
        t_l3 = self.proj_l3(f_l3) + self.scale_embed.weight[1]
        mem_xyz = torch.cat([l2_xyz, l3_xyz], dim=1) # (B, 320, 3)
        mem_tokens = torch.cat([t_l2, t_l3], dim=1) + self.pe_mlp(mem_xyz) # (B, 320, d_model)

        # Target queries initialized from target embeddings + position embeddings of prior
        t_idx = torch.arange(K, device=pts.device)
        queries = self.target_embed(t_idx).unsqueeze(0).expand(B, -1, -1) # (B, K, d_model)
        # Position embedding from prior prediction (in normalized units: mm / 500)
        queries = queries + self.atlas_pe_mlp(p_prior / 500.0)

        if self.use_metadata and metadata is not None:
            m_feat = self.meta_mlp(metadata).unsqueeze(1) # (B, 1, d_model)
            queries = queries + m_feat

        h = queries
        for l in range(self.num_layers):
            # Cross-attention to surface tokens
            attn_out, _ = self.cross_attns[l](query=h, key=mem_tokens, value=mem_tokens)
            h = self.norms1[l](h + attn_out)

            # Self-attention between target queries
            self_out, _ = self.self_attns[l](query=h, key=h, value=h)
            h = self.norms_self[l](h + self_out)

            # FFN
            ffn_out = self.ffns[l](h)
            h = self.norms2[l](h + ffn_out)

        # Local residual offset: scale by 500 mm
        delta_p_local = self.residual_head(h) * 100.0 # (B, K, 3) in mm

        # Target gating: alpha in [0, 1]
        alpha = self.gate_head(h) # (B, K, 1)

        # Fused final prediction: P_prior + (1 - alpha) * delta_p_local
        # When alpha=1, strictly prior; when alpha=0, full prior + local residual
        p_final = p_prior + (1.0 - alpha) * delta_p_local

        return {
            "z_hat": z_hat,
            "p_prior": p_prior,
            "delta_local": delta_p_local,
            "alpha_gate": alpha,
            "p_final": p_final
        }
