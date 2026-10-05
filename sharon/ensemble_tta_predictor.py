import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

SHARON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARON_DIR))

from model import Modular3DOrganPredictor
from losses import soft_argmax
from labels import ORGAN_NAMES, NUM_ORGANS


class EnsembleTTAPredictor:
    """
    Multi-Backbone Ensemble Predictor with 8-Fold Test-Time Augmentation (TTA)
    and Organ Field-of-View (FOV) / Presence Confidence Filter.
    """
    # 8 Spatial 3D Reflection Combinations over (Z, Y, X) axes
    TTA_FLIP_DIMS = [
        (),          # 1. Identity
        (2,),        # 2. Flip Z (axial)
        (3,),        # 3. Flip Y (coronal)
        (4,),        # 4. Flip X (sagittal)
        (2, 3),      # 5. Flip Z, Y
        (2, 4),      # 6. Flip Z, X
        (3, 4),      # 7. Flip Y, X
        (2, 3, 4),   # 8. Flip Z, Y, X
    ]

    def __init__(
        self,
        checkpoint_paths: Dict[str, str],
        weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
        num_organs: int = NUM_ORGANS,
        heatmap_size: Tuple[int, int, int] = (64, 64, 64),
        presence_threshold: float = 0.25,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_organs = num_organs
        self.heatmap_size = heatmap_size
        self.presence_threshold = presence_threshold

        # Default inverse-error weights
        default_weights = {"resnet50": 0.50, "densenet121": 0.35, "swin_unetr": 0.15}
        self.weights = weights or default_weights

        # Normalize weights
        total_w = sum(self.weights.get(m, 0.0) for m in checkpoint_paths.keys())
        self.norm_weights = {m: self.weights.get(m, 0.0) / total_w for m in checkpoint_paths.keys()}

        self.models = {}
        for backbone_name, ckpt_path in checkpoint_paths.items():
            if not os.path.exists(ckpt_path):
                print(f"[Ensemble Warning] Checkpoint not found: {ckpt_path}, skipping {backbone_name}")
                continue

            print(f"[Ensemble] Loading {backbone_name} from {ckpt_path}...")
            model = Modular3DOrganPredictor(
                num_organs=num_organs,
                heatmap_size=heatmap_size,
                freeze_blocks=0,
                dropout=0.0,
                backbone=backbone_name,
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models[backbone_name] = model

        print(f"[Ensemble] Initialized with {len(self.models)} active models: {list(self.models.keys())}")

    def _predict_single_model_tta(self, model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
        """
        Runs 8-fold test-time spatial reflection augmentation and averages heatmaps.
        """
        fused_heatmaps = torch.zeros(
            (img.shape[0], self.num_organs, *self.heatmap_size),
            device=self.device,
            dtype=torch.float32,
        )

        for flip_axes in self.TTA_FLIP_DIMS:
            x_aug = torch.flip(img, dims=flip_axes) if flip_axes else img
            pred_hm = model(x_aug)
            pred_hm_inv = torch.flip(pred_hm, dims=flip_axes) if flip_axes else pred_hm
            fused_heatmaps += pred_hm_inv

        return fused_heatmaps / len(self.TTA_FLIP_DIMS)

    @torch.inference_mode()
    def predict(
        self,
        img: torch.Tensor,
        use_tta: bool = True,
        temperature: float = 1000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            img: 3D CT tensor of shape (B, 1, D, H, W)
            use_tta: Whether to run 8-fold spatial reflection TTA
            temperature: Inverse temperature for soft-argmax

        Returns:
            pred_centroids_norm: (B, num_organs, 3) normalized coordinate centers (z, y, x)
            fused_heatmaps: (B, num_organs, D_hm, H_hm, W_hm) fused Gaussian confidence fields
            confidence_scores: (B, num_organs) peak confidence score per organ in [0, 1]
        """
        img = img.to(self.device)
        B = img.shape[0]

        fused_heatmaps = torch.zeros(
            (B, self.num_organs, *self.heatmap_size),
            device=self.device,
            dtype=torch.float32,
        )

        for name, model in self.models.items():
            w = self.norm_weights[name]
            if use_tta:
                hm = self._predict_single_model_tta(model, img)
            else:
                hm = model(img)
            fused_heatmaps += w * hm

        # Compute peak confidence score per organ (maximum heatmap response)
        # Shape: (B, num_organs)
        confidence_scores = fused_heatmaps.amax(dim=(-3, -2, -1))

        # Extract continuous centroids via differentiable soft-argmax
        pred_centroids_norm = soft_argmax(fused_heatmaps, temperature=temperature)

        return pred_centroids_norm, fused_heatmaps, confidence_scores
