import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
 
 
from labels import NUM_ORGANS
 
 
class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
 
 
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBnRelu(out_ch, out_ch)
        self.drop = nn.Dropout3d(dropout)
 
    def forward(self, x):
        return self.drop(self.conv(self.up(x)))
 
 
class HeatmapHead(nn.Module):
  
    def __init__(self, in_channels, num_organs=NUM_ORGANS,
                 heatmap_size=(64, 64, 64), dropout=0.2):
        super().__init__()
        feat_size = 128 // 32          # 8  (DenseNet121 stride = 32)
        num_ups   = int(round(math.log2(heatmap_size[0] / feat_size)))  # 3
 
        layers, ch = [], in_channels
        for _ in range(num_ups):
            out_ch = max(ch // 2, 64)
            layers.append(DecoderBlock(ch, out_ch, dropout))
            ch = out_ch
 
        self.decoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, num_organs, 1),
        )
 
    def forward(self, x):
        return torch.sigmoid(self.head(self.decoder(x)))
 
 
class OrganLocalizationNet(nn.Module):
    def __init__(self, num_organs, heatmap_size=(64,64,64),
                 freeze_blocks=2, dropout=0.2):
        super().__init__()
        _dn = DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_organs)
        self.features = _dn.features
        self.decoder  = HeatmapHead(1024, num_organs, heatmap_size, dropout)
        self._freeze_blocks(freeze_blocks)
 
    def _freeze_blocks(self, n):
        if n <= 0:
            return
        targets = [self.features.conv0, self.features.norm0]
        for i in range(1, n + 1):
            targets.append(getattr(self.features, f"denseblock{i}"))
            t = getattr(self.features, f"transition{i}", None)
            if t and i < n:
                targets.append(t)
        for m in targets:
            for p in m.parameters():
                p.requires_grad = False
 
    def forward(self, x):
        feats = F.relu(self.features(x), inplace=True)  
        return self.decoder(feats)