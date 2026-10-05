import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sharon.model_gnn import PointNet2Encoder
    from sharon.labels import NUM_ORGANS
except ImportError:
    from model_gnn import PointNet2Encoder
    from labels import NUM_ORGANS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline 1: Standard PointNet++ Direct Coordinate Regressor
# ─────────────────────────────────────────────────────────────────────────────

class PointNet2DirectRegressor(nn.Module):
    """
    PointNet++ Direct Coordinate Regressor:
    Extracts global geometric feature f_geo in R^1024 from (B, 4096, 3) point cloud,
    and directly regresses 121 3D organ centroids via an MLP (no sex conditioning, no GNN).
    """
    def __init__(self, num_organs: int = NUM_ORGANS, dropout: float = 0.2):
        super().__init__()
        self.num_organs = num_organs
        self.encoder = PointNet2Encoder(in_channel=3, out_dim=1024)

        self.regressor = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_organs * 3),
        )

    def forward(
        self,
        points: torch.Tensor,                      # (B, 4096, 3)
        sex_prior: Optional[torch.Tensor] = None,   # Ignored by unconditioned baseline
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = points.shape[0]
        f_geo = self.encoder(points)                # (B, 1024)
        out = self.regressor(f_geo)                 # (B, num_organs * 3)
        pred_centroids = out.view(B, self.num_organs, 3)

        if mask is not None:
            pred_centroids = pred_centroids * mask.unsqueeze(-1)
        return pred_centroids


# ─────────────────────────────────────────────────────────────────────────────
# 2. Baseline 2: Dynamic Graph CNN (DGCNN) Regressor
# ─────────────────────────────────────────────────────────────────────────────

def knn_feature(x: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    Constructs k-NN graph in feature space and extracts edge features.
    x: (B, C, N)
    Output: (B, 2*C, N, k) edge features [x_j - x_i || x_i]
    """
    B, C, N = x.shape
    # Pairwise inner product: (B, N, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_dist = -xx - inner - xx.transpose(2, 1)

    # Top-k nearest neighbors
    idx = pairwise_dist.topk(k=k, dim=-1)[1]     # (B, N, k)

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x_flat = x.transpose(2, 1).contiguous().view(B * N, C)
    feature = x_flat[idx, :].view(B, N, k, C)
    x_expanded = x.transpose(2, 1).view(B, N, 1, C).repeat(1, 1, k, 1)

    # Concatenate relative displacement and source feature
    edge_feature = torch.cat((feature - x_expanded, x_expanded), dim=-1).permute(0, 3, 1, 2) # (B, 2*C, N, k)
    return edge_feature


class EdgeConvBlock(nn.Module):
    """
    EdgeConv Layer from DGCNN (Wang et al., 2019):
    Computes dynamical k-NN graphs in feature space and applies shared Conv2d + max pooling.
    """
    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, N)
        Output: (B, out_channels, N)
        """
        edge_feat = knn_feature(x, k=self.k)       # (B, 2*C, N, k)
        out = self.conv(edge_feat)                 # (B, out_channels, N, k)
        out = out.max(dim=-1, keepdim=False)[0]    # (B, out_channels, N)
        return out


class DGCNNRegressor(nn.Module):
    """
    Dynamic Graph CNN (DGCNN) Organ Centroid Regressor:
    Extracts multi-scale EdgeConv dynamic graph features from (B, 4096, 3),
    and regresses 121 3D organ centroids (no sex conditioning, no dynamic biological masking).
    """
    def __init__(self, num_organs: int = NUM_ORGANS, k: int = 20, dropout: float = 0.2):
        super().__init__()
        self.num_organs = num_organs
        self.k = k

        # EdgeConv Hierarchy
        self.conv1 = EdgeConvBlock(in_channels=3, out_channels=64, k=k)
        self.conv2 = EdgeConvBlock(in_channels=64, out_channels=64, k=k)
        self.conv3 = EdgeConvBlock(in_channels=64, out_channels=128, k=k)
        self.conv4 = EdgeConvBlock(in_channels=128, out_channels=256, k=k)

        # Multi-scale aggregation (64 + 64 + 128 + 256 = 512)
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Coordinate Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(1024 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_organs * 3),
        )

    def forward(
        self,
        points: torch.Tensor,                      # (B, 4096, 3)
        sex_prior: Optional[torch.Tensor] = None,   # Ignored
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = points.shape
        x = points.transpose(2, 1)                  # (B, 3, N)

        # Multi-scale EdgeConvs
        x1 = self.conv1(x)                          # (B, 64, N)
        x2 = self.conv2(x1)                         # (B, 64, N)
        x3 = self.conv3(x2)                         # (B, 128, N)
        x4 = self.conv4(x3)                         # (B, 256, N)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        x5 = self.conv5(x_cat)                      # (B, 1024, N)

        # Global pooling (Max + Avg)
        x_max = F.adaptive_max_pool1d(x5, 1).view(B, -1)  # (B, 1024)
        x_avg = F.adaptive_avg_pool1d(x5, 1).view(B, -1)  # (B, 1024)
        x_global = torch.cat((x_max, x_avg), dim=1)       # (B, 2048)

        out = self.regressor(x_global)                    # (B, num_organs * 3)
        pred_centroids = out.view(B, self.num_organs, 3)

        if mask is not None:
            pred_centroids = pred_centroids * mask.unsqueeze(-1)
        return pred_centroids
