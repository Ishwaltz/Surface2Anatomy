import math
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from sharon.model_gnn import PointNetSetAbstraction

class MultiScaleSurfacePointNet2Encoder(nn.Module):
    """
    PointNet++ encoder that exposes multi-scale intermediate surface tokens
    (SA1: 1024, SA2: 256, SA3: 64) and the global pooled vector.
    """
    def __init__(self, in_channel: int = 3, out_dim: int = 1024):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.8, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 512, out_dim], group_all=True)

    def forward(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1_xyz, l1_points = self.sa1(xyz, None)             # (B, 1024, 3), (B, 1024, 128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)     # (B, 256, 3), (B, 256, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)     # (B, 64, 3), (B, 64, 512)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)     # (B, 1, 3), (B, 1, out_dim)
        
        return {
            "l1_xyz": l1_xyz, "l1_feat": l1_points,
            "l2_xyz": l2_xyz, "l2_feat": l2_points,
            "l3_xyz": l3_xyz, "l3_feat": l3_points,
            "global_feat": l4_points.view(xyz.shape[0], -1)
        }

class TargetQueryTransformerDecoder(nn.Module):
    """
    Target-Query Cross-Attention Decoder:
    121 learned anatomical queries attend to multi-scale surface tokens
    and predict residual offsets from a canonical training atlas.
    """
    def __init__(
        self,
        atlas_coords: torch.Tensor, # (121, 3) in centered model coordinates
        num_organs: int = 121,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        use_metadata: bool = False,
        use_geo_bias: bool = False,
        use_self_attn: bool = False,
        global_only: bool = False,
        dropout: float = 0.05
    ):
        super().__init__()
        self.num_organs = num_organs
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_metadata = use_metadata
        self.use_geo_bias = use_geo_bias
        self.use_self_attn = use_self_attn
        self.global_only = global_only

        # Fixed anatomical atlas reference in model coordinates
        self.register_buffer("atlas_coords", atlas_coords.clone().float())

        # Encoder
        self.encoder = MultiScaleSurfacePointNet2Encoder(in_channel=3, out_dim=1024)

        # Projections for surface tokens
        if not global_only:
            self.proj_l2 = nn.Linear(256, d_model)
            self.proj_l3 = nn.Linear(512, d_model)
            self.scale_embed = nn.Embedding(2, d_model) # 0 for l2, 1 for l3
        else:
            self.proj_global = nn.Linear(1024, d_model)

        # 3D Positional encoding for surface tokens
        self.pe_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Learned target queries initialized with atlas position
        self.target_embed = nn.Embedding(num_organs, d_model)
        self.atlas_pe_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Metadata context encoder: [W, D, H, Area, Vol, Sex] (6 features)
        if use_metadata:
            self.meta_mlp = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, d_model)
            )

        # Geometric attention bias MLP: relative vector (x_j - a_k) -> nhead bias
        if use_geo_bias:
            self.geo_bias_mlp = nn.Sequential(
                nn.Linear(4, 32), # dx, dy, dz, distance
                nn.ReLU(),
                nn.Linear(32, nhead)
            )

        # Transformer layers
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        if use_self_attn:
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

        # Coordinate residual head: predicts delta_p_k in model space
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(
        self,
        pts: torch.Tensor,                   # (B, 4096, 3)
        metadata: Optional[torch.Tensor] = None, # (B, 6)
        query_perm: Optional[torch.Tensor] = None, # (num_organs,) optional diagnostic
        token_shuffle: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = pts.shape[0]
        enc_out = self.encoder(pts)

        # 1. Assemble Surface Memory Tokens
        if self.global_only:
            # Attend ONLY to single repeated global pooled vector
            mem_tokens = self.proj_global(enc_out["global_feat"]).unsqueeze(1) # (B, 1, d_model)
            mem_xyz = torch.zeros(B, 1, 3, device=pts.device)
        else:
            # Combine L2 (256 mid tokens) and L3 (64 coarse tokens) = 320 surface tokens
            l2_xyz, l2_f = enc_out["l2_xyz"], enc_out["l2_feat"]
            l3_xyz, l3_f = enc_out["l3_xyz"], enc_out["l3_feat"]

            t_l2 = self.proj_l2(l2_f) + self.scale_embed.weight[0]
            t_l3 = self.proj_l3(l3_f) + self.scale_embed.weight[1]

            mem_xyz = torch.cat([l2_xyz, l3_xyz], dim=1) # (B, 320, 3)
            mem_f = torch.cat([t_l2, t_l3], dim=1)       # (B, 320, d_model)
            mem_tokens = mem_f + self.pe_mlp(mem_xyz)     # (B, 320, d_model)

        if token_shuffle:
            # Diagnostic D3: permute memory tokens across patients in batch
            perm = torch.randperm(B, device=pts.device)
            mem_tokens = mem_tokens[perm]
            mem_xyz = mem_xyz[perm]

        # 2. Target Queries Initialization
        q_idx = torch.arange(self.num_organs, device=pts.device)
        if query_perm is not None:
            q_idx = q_idx[query_perm]

        target_emb = self.target_embed(q_idx) # (121, d_model)
        atlas_emb = self.atlas_pe_mlp(self.atlas_coords) # (121, d_model)
        queries = (target_emb + atlas_emb).unsqueeze(0).expand(B, -1, -1) # (B, 121, d_model)

        # 3. Metadata Conditioning
        if self.use_metadata and metadata is not None:
            m_ctx = self.meta_mlp(metadata).unsqueeze(1) # (B, 1, d_model)
            queries = queries + m_ctx

        # 4. Geometric Attention Bias (if active and not global_only)
        attn_bias = None
        if self.use_geo_bias and not self.global_only:
            # Compute relative vector: mem_xyz - atlas_coords
            # mem_xyz: (B, 320, 3), atlas_coords: (121, 3)
            diff = mem_xyz.unsqueeze(1) - self.atlas_coords.unsqueeze(0).unsqueeze(2) # (B, 121, 320, 3)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            r_feat = torch.cat([diff, dist], dim=-1) # (B, 121, 320, 4)
            # Output: (B, 121, 320, nhead) -> permute to (B * nhead, 121, 320)
            geo_b = self.geo_bias_mlp(r_feat)
            attn_bias = geo_b.permute(0, 3, 1, 2).reshape(B * self.nhead, self.num_organs, -1)

        # 5. Transformer Layers
        last_attn_weights = None
        for i in range(self.num_layers):
            # Target self-attention (if enabled)
            if self.use_self_attn:
                q_self, _ = self.self_attns[i](queries, queries, queries)
                queries = self.norms_self[i](queries + q_self)

            # Cross-attention: Queries attend to Surface Tokens
            # In PyTorch MultiheadAttention, attn_mask can be an additive bias (B*nhead, Q, K)
            q_cross, attn_w = self.cross_attns[i](
                queries, mem_tokens, mem_tokens,
                attn_mask=attn_bias,
                need_weights=True
            )
            queries = self.norms1[i](queries + q_cross)
            queries = self.norms2[i](queries + self.ffns[i](queries))
            last_attn_weights = attn_w # (B, 121, 320)

        # 6. Coordinate Residual Output
        delta_p = self.coord_head(queries) # (B, 121, 3)
        # Final coordinate: atlas reference + residual offset
        pred_centroids = self.atlas_coords.unsqueeze(0) + delta_p

        return pred_centroids, last_attn_weights
