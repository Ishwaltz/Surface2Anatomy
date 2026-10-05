"""
Surface2Anatomy Neural Network Architecture:
PointNet++ Multi-Scale Surface Encoder + Target-Query Cross-Attention Transformer Decoder.
100% Identical Architecture Definition to Frozen Checkpoints.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Computes pairwise Euclidean distance squared between two point sets."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Indexes points along batch dimension."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative farthest point sampling algorithm."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Ball query to find all points within radius of new_xyz."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction layer with hierarchical feature extraction."""
    def __init__(self, npoint: Optional[int], radius: Optional[float], nsample: Optional[int], in_channel: int, mlp: list, group_all: bool):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = xyz.device
        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, C, device=device)
            grouped_xyz = xyz.view(B, 1, N, C)
            if points is not None:
                new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)
            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 2, 1) # [B, C, nsample, npoint]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1) # [B, npoint, C]
        return new_xyz, new_points

class MultiScaleSurfacePointNet2Encoder(nn.Module):
    """
    PointNet++ surface encoder exposing multi-scale intermediate tokens:
    SA1: 1024, SA2: 256, SA3: 64, and global pooled vector (1024).
    """
    def __init__(self, in_channel: int = 3, out_dim: int = 1024):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.8, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 512, out_dim], group_all=True)

    def forward(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        return {
            "l1_xyz": l1_xyz, "l1_feat": l1_points,
            "l2_xyz": l2_xyz, "l2_feat": l2_points,
            "l3_xyz": l3_xyz, "l3_feat": l3_points,
            "global_feat": l4_points.view(xyz.shape[0], -1)
        }

class TargetQueryTransformerDecoder(nn.Module):
    """
    Target-Query Cross-Attention Decoder:
    117 learned anatomical queries attend to 320 multi-scale surface tokens (256 SA2 + 64 SA3)
    and predict residual 3D offsets from the canonical training atlas.
    """
    def __init__(
        self,
        atlas_coords: torch.Tensor,
        num_organs: int = 117,
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

        self.register_buffer("atlas_coords", atlas_coords.clone().float())
        self.encoder = MultiScaleSurfacePointNet2Encoder(in_channel=3, out_dim=1024)

        if not global_only:
            self.proj_l2 = nn.Linear(256, d_model)
            self.proj_l3 = nn.Linear(512, d_model)
            self.scale_embed = nn.Embedding(2, d_model)
        else:
            self.proj_global = nn.Linear(1024, d_model)

        self.pe_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.target_embed = nn.Embedding(num_organs, d_model)
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

        if use_geo_bias:
            self.geo_bias_mlp = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, nhead)
            )

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

        self.coord_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(
        self,
        pts: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        query_perm: Optional[torch.Tensor] = None,
        token_shuffle: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = pts.shape[0]
        enc_out = self.encoder(pts)

        if self.global_only:
            mem_tokens = self.proj_global(enc_out["global_feat"]).unsqueeze(1)
            mem_xyz = torch.zeros(B, 1, 3, device=pts.device)
        else:
            l2_xyz, l2_f = enc_out["l2_xyz"], enc_out["l2_feat"]
            l3_xyz, l3_f = enc_out["l3_xyz"], enc_out["l3_feat"]

            t_l2 = self.proj_l2(l2_f) + self.scale_embed.weight[0]
            t_l3 = self.proj_l3(l3_f) + self.scale_embed.weight[1]

            mem_xyz = torch.cat([l2_xyz, l3_xyz], dim=1)
            mem_f = torch.cat([t_l2, t_l3], dim=1)
            mem_tokens = mem_f + self.pe_mlp(mem_xyz)

        if token_shuffle:
            perm = torch.randperm(B, device=pts.device)
            mem_tokens = mem_tokens[perm]
            mem_xyz = mem_xyz[perm]

        q_idx = torch.arange(self.num_organs, device=pts.device)
        if query_perm is not None:
            q_idx = q_idx[query_perm]

        target_emb = self.target_embed(q_idx)
        atlas_emb = self.atlas_pe_mlp(self.atlas_coords.to(pts.device))
        queries = (target_emb + atlas_emb).unsqueeze(0).expand(B, -1, -1)

        if self.use_metadata and metadata is not None:
            m_ctx = self.meta_mlp(metadata).unsqueeze(1)
            queries = queries + m_ctx

        attn_bias = None
        if self.use_geo_bias and not self.global_only:
            diff = mem_xyz.unsqueeze(1) - self.atlas_coords.unsqueeze(0).unsqueeze(2)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            r_feat = torch.cat([diff, dist], dim=-1)
            geo_b = self.geo_bias_mlp(r_feat)
            attn_bias = geo_b.permute(0, 3, 1, 2).reshape(B * self.nhead, self.num_organs, -1)

        last_attn_weights = None
        for i in range(self.num_layers):
            if self.use_self_attn:
                q_self, _ = self.self_attns[i](queries, queries, queries)
                queries = self.norms_self[i](queries + q_self)

            q_cross, attn_w = self.cross_attns[i](
                queries, mem_tokens, mem_tokens,
                attn_mask=attn_bias,
                need_weights=True
            )
            queries = self.norms1[i](queries + q_cross)
            queries = self.norms2[i](queries + self.ffns[i](queries))
            last_attn_weights = attn_w

        delta_p = self.coord_head(queries)
        pred_centroids = self.atlas_coords.unsqueeze(0).to(pts.device) + delta_p

        return pred_centroids, last_attn_weights
