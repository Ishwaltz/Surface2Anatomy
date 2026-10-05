import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

try:
    from sharon.labels import (
        NUM_ORGANS, NUM_REGIONS, ORGAN_NAMES,
        get_sex_organ_mask, get_organ_region_indices
    )
except ImportError:
    from labels import (
        NUM_ORGANS, NUM_REGIONS, ORGAN_NAMES,
        get_sex_organ_mask, get_organ_region_indices
    )


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
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
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = torch.sort(group_idx, dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstraction(nn.Module):
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
        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3, device=xyz.device)
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            if points is not None:
                new_points = torch.cat([grouped_xyz, points.view(points.shape[0], 1, points.shape[1], -1)], dim=-1)
            else:
                new_points = grouped_xyz
            new_points = new_points.permute(0, 3, 2, 1)
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz_norm = grouped_xyz - new_xyz.view(new_xyz.shape[0], self.npoint, 1, 3)
            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz_norm
            new_points = new_points.permute(0, 3, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xyz, new_points


class PointNet2Encoder(nn.Module):
    def __init__(self, in_channel: int = 3, out_dim: int = 1024):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.8, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 512, out_dim], group_all=True)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        return l4_points.view(xyz.shape[0], -1)


class SexEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, sex_prior: torch.Tensor) -> torch.Tensor:
        if sex_prior.dim() == 1:
            if sex_prior.numel() == 2:
                sex_prior = sex_prior.unsqueeze(0)
            else:
                s_val = (sex_prior > 0.5).long()
                sex_prior = F.one_hot(s_val, num_classes=2).float()
        elif sex_prior.shape[-1] == 1:
            s_val = (sex_prior.squeeze(-1) > 0.5).long()
            sex_prior = F.one_hot(s_val, num_classes=2).float()
        return self.net(sex_prior.float())


class DynamicMaskedGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, K, D = h.shape
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)

        # Dynamic Masking: (B, K, 1) * (B, 1, K) = (B, K, K)
        pair_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))
        scores = scores.masked_fill(pair_mask < 0.5, -1e4)

        attn = F.softmax(scores, dim=-1)
        attn = attn * pair_mask
        row_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = attn / row_sum

        msg = torch.matmul(attn, v)
        h = self.norm1(h + msg)
        h = self.norm2(h + self.mlp(h))
        return h * mask.unsqueeze(-1)


class HierarchicalSexConditionedGNN(nn.Module):
    """
    Two-Stage Hierarchical Coarse-to-Fine Point Cloud Regression Architecture.
    Stage 1: Predicts 4 functional anatomical regional anchors in [-1, 1]^3.
    Stage 2: Regresses fine local offsets delta_y_i relative to regional anchors
             using dynamic masked inter-organ message passing.
    """
    def __init__(
        self,
        num_organs: int = 121,
        num_regions: int = 4,
        pointnet_feat_dim: int = 1024,
        sex_emb_dim: int = 128,
        hidden_dim: int = 256,
        gnn_layers: int = 4,
    ):
        super().__init__()
        self.num_organs = num_organs
        self.num_regions = num_regions

        # Encoders
        self.point_encoder = PointNet2Encoder(out_dim=pointnet_feat_dim)
        self.sex_encoder = SexEncoder(out_dim=sex_emb_dim)

        # Patient Latent Fusion
        self.fusion = nn.Sequential(
            nn.Linear(pointnet_feat_dim + sex_emb_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
        )

        # Stage 1: Regional Anchor Head -> (B, 4, 3) in [-1, 1]
        self.regional_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_regions * 3),
            nn.Tanh(),  # strictly bounded in [-1.0, 1.0]
        )

        # Stage 2: Organ Query Embeddings
        self.organ_queries = nn.Parameter(torch.randn(num_organs, hidden_dim) * 0.02)
        self.register_buffer("region_indices", get_organ_region_indices())

        # Region & Anchor Projection
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Dynamic Masked GNN
        self.gnn_layers = nn.ModuleList([DynamicMaskedGNNLayer(hidden_dim=hidden_dim) for _ in range(gnn_layers)])

        # Fine Local Offset Head -> (B, 121, 3)
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        points: torch.Tensor,
        sex_prior: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_regional: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = points.shape[0]
        device = points.device

        if mask is None:
            mask = get_sex_organ_mask(sex_prior, num_organs=self.num_organs, device=device)

        # 1. Encoders
        f_geo = self.point_encoder(points)
        f_sex = self.sex_encoder(sex_prior)
        z = self.fusion(torch.cat([f_geo, f_sex], dim=-1)) # (B, 512)

        # 2. Stage 1: Coarse Regional Anchor Prediction
        pred_regions = self.regional_head(z).view(B, self.num_regions, 3) # (B, 4, 3)

        # 3. Gather regional anchor per organ
        reg_idx = self.region_indices.unsqueeze(0).expand(B, -1) # (B, 121)
        # Gather (B, 121, 3) from (B, 4, 3)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.num_organs)
        organ_anchors = pred_regions[batch_idx, reg_idx] # (B, 121, 3)

        # 4. Stage 2: Initialize Organ Queries with Regional Anchors + Patient Context
        q_exp = self.organ_queries.unsqueeze(0).expand(B, -1, -1) # (B, 121, hidden_dim)
        z_exp = z.unsqueeze(1).expand(-1, self.num_organs, -1)    # (B, 121, 512)
        h = self.query_proj(torch.cat([q_exp, organ_anchors, z_exp], dim=-1)) # (B, 121, hidden_dim)

        # 5. Dynamic Masked GNN Message Passing
        for layer in self.gnn_layers:
            h = layer(h, mask)

        # 6. Fine Local Offsets
        offsets = self.offset_head(h) # (B, 121, 3)

        # 7. Final Hierarchical Reconstruction
        pred_centroids = organ_anchors + offsets
        pred_centroids = pred_centroids * mask.unsqueeze(-1)

        if return_regional:
            return pred_centroids, pred_regions
        return pred_centroids


# Alias for backward compatibility
SexConditionedOrganGNN = HierarchicalSexConditionedGNN
