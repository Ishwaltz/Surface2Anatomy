import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class LocalPatchRefiner3D(nn.Module):
    """
    Stage 2 Local Patch Refiner:
    Takes a 3D CT patch (1, D_p, H_p, W_p) centered around Stage 1 coarse prediction
    along with organ class embedding, and regresses physical sub-voxel delta offset (Δx, Δy, Δz) in mm.
    """
    def __init__(self, num_organs: int = 117, patch_size: int = 32, embed_dim: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.organ_embedding = nn.Embedding(num_organs, embed_dim)

        self.init_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # (32 -> 16)
        )

        self.layer1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResBlock3D(64),
            nn.MaxPool3d(2), # (16 -> 8)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            ResBlock3D(128),
            nn.AdaptiveAvgPool3d((1, 1, 1)), # (128, 1, 1, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3), # (Δz, Δy, Δx) normalized delta in range [-0.2, 0.2]
        )

    def forward(self, patch: torch.Tensor, organ_ids: torch.Tensor) -> torch.Tensor:
        """
        patch: (B, 1, D_p, H_p, W_p)
        organ_ids: (B,) int
        returns: (B, 3) delta offsets in normalized voxel coordinate space
        """
        x = self.init_conv(patch)
        x = self.layer1(x)
        feat = self.layer2(x).flatten(1) # (B, 128)

        emb = self.organ_embedding(organ_ids) # (B, embed_dim)
        concat = torch.cat([feat, emb], dim=1) # (B, 128 + embed_dim)

        delta = self.fc(concat) # (B, 3)
        return torch.tanh(delta) * 0.20 # bounded sub-voxel offset range [-0.20, +0.20]
