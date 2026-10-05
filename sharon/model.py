import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai.networks.nets as nets

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
    def __init__(self, in_channels=1024, num_organs=NUM_ORGANS,
                 heatmap_size=(64, 64, 64), dropout=0.2):
        super().__init__()
        feat_size = 128 // 32          # 4
        num_ups   = int(round(math.log2(heatmap_size[0] / feat_size)))

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


class AdaptiveBottleneck(nn.Module):
    def __init__(self, in_channels, target_channels=1024, target_spatial=(4, 4, 4)):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d(target_spatial)
        self.channel_proj = nn.Sequential(
            nn.Conv3d(in_channels, target_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(target_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.spatial_pool(x)
        return self.channel_proj(x)


class Modular3DOrganPredictor(nn.Module):
    def __init__(
        self,
        num_organs=NUM_ORGANS,
        heatmap_size=(64, 64, 64),
        freeze_blocks=0,
        dropout=0.2,
        backbone="swin_unetr",
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.num_organs = num_organs
        self.heatmap_size = heatmap_size

        if self.backbone_name == "densenet121":
            dn = nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_organs)
            self.encoder = dn.features
            encoder_channels = 1024
            if freeze_blocks > 0:
                self._freeze_densenet(self.encoder, freeze_blocks)

        elif self.backbone_name == "resnet50":
            self.encoder = nets.resnet50(spatial_dims=3, n_input_channels=1, feed_forward=False)
            encoder_channels = 2048

        elif self.backbone_name == "efficientnet_b0":
            self.encoder = nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=1)
            encoder_channels = 1280

        elif self.backbone_name == "swin_unetr":
            swin = nets.SwinUNETR(
                in_channels=1,
                out_channels=num_organs,
                feature_size=48,
                spatial_dims=3,
                use_checkpoint=True,
            )
            self.encoder = swin.swinViT
            encoder_channels = 768

        else:
            raise ValueError(
                f"Unsupported backbone: '{backbone}'. Supported options: ['densenet121', 'resnet50', 'efficientnet_b0', 'swin_unetr']"
            )

        self.bottleneck = AdaptiveBottleneck(encoder_channels, target_channels=1024, target_spatial=(4, 4, 4))
        self.decoder = HeatmapHead(in_channels=1024, num_organs=num_organs, heatmap_size=heatmap_size, dropout=dropout)

    def _freeze_densenet(self, features, n):
        targets = [features.conv0, features.norm0]
        for i in range(1, n + 1):
            targets.append(getattr(features, f"denseblock{i}"))
            t = getattr(features, f"transition{i}", None)
            if t and i < n:
                targets.append(t)
        for m in targets:
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        if self.backbone_name == "densenet121":
            feats = F.relu(self.encoder(x), inplace=True)
        elif self.backbone_name == "resnet50":
            x_rn = self.encoder.conv1(x)
            x_rn = self.encoder.bn1(x_rn)
            x_rn = self.encoder.act(x_rn)
            if hasattr(self.encoder, "maxpool"):
                x_rn = self.encoder.maxpool(x_rn)
            x_rn = self.encoder.layer1(x_rn)
            x_rn = self.encoder.layer2(x_rn)
            x_rn = self.encoder.layer3(x_rn)
            feats = self.encoder.layer4(x_rn)
        elif self.backbone_name == "efficientnet_b0":
            x_eff = self.encoder._conv_stem(x)
            if hasattr(self.encoder, "_bn0"):
                x_eff = self.encoder._bn0(x_eff)
            x_eff = self.encoder._swish(x_eff)
            for b in self.encoder._blocks:
                x_eff = b(x_eff)
            x_eff = self.encoder._conv_head(x_eff)
            x_eff = self.encoder._bn1(x_eff)
            feats = self.encoder._swish(x_eff)
        elif self.backbone_name == "swin_unetr":
            swin_outs = self.encoder(x)
            feats = swin_outs[-1]

        bottleneck_feats = self.bottleneck(feats)
        return self.decoder(bottleneck_feats)


class OrganLocalizationNet(Modular3DOrganPredictor):
    def __init__(self, num_organs=NUM_ORGANS, heatmap_size=(64, 64, 64), freeze_blocks=0, dropout=0.2, backbone="swin_unetr"):
        super().__init__(
            num_organs=num_organs,
            heatmap_size=heatmap_size,
            freeze_blocks=freeze_blocks,
            dropout=dropout,
            backbone=backbone,
        )