"""
Canonical Body-Frame Geometric Alignment.
Extracts 105 external-only morphology features and computes the canonical body center
via the frozen V3 Ridge regressor (canonical_alignment_v3_ridge.joblib).
Strictly adheres to tools/dataset_v3/apply_canonical_alignment.py.
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
import joblib

from surface2anatomy.exceptions import AlignmentError

def extract_alignment_features(pts_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 105 geometric surface features from external body point cloud ONLY.
    Strictly forbids any CT voxels, internal organ masks, or target ground truth.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (feature_vector_105, bounding_box_midpoint)
    """
    if len(pts_world) < 64:
        raise AlignmentError("Point cloud too sparse to extract canonical morphology features.")

    p_min = pts_world.min(axis=0)
    p_max = pts_world.max(axis=0)
    span = p_max - p_min
    mid = 0.5 * (p_min + p_max)
    mean = pts_world.mean(axis=0)
    std = pts_world.std(axis=0)

    # 12 Normalized axial height slice features
    z_norm = (pts_world[:, 2] - p_min[2]) / (span[2] + 1e-6)
    slice_features = []
    for b in range(12):
        in_b = pts_world[(z_norm >= b / 12.0) & (z_norm < (b + 1) / 12.0)]
        if len(in_b) >= 10:
            w = np.percentile(in_b[:, 0], 98) - np.percentile(in_b[:, 0], 2)
            d = np.percentile(in_b[:, 1], 98) - np.percentile(in_b[:, 1], 2)
            asp = w / (d + 1e-6)
            xm = 0.5 * (np.percentile(in_b[:, 0], 98) + np.percentile(in_b[:, 0], 2))
            ym = 0.5 * (np.percentile(in_b[:, 1], 98) + np.percentile(in_b[:, 1], 2))
            slice_features.extend([w, d, asp, xm - mid[0], ym - mid[1]])
        else:
            slice_features.extend([span[0], span[1], span[0] / (span[1] + 1e-6), 0.0, 0.0])

    # Surface coordinate percentiles relative to bounding box midpoint
    px = np.percentile(pts_world[:, 0], [1, 5, 25, 50, 75, 95, 99])
    py = np.percentile(pts_world[:, 1], [1, 5, 25, 50, 75, 95, 99])
    pz = np.percentile(pts_world[:, 2], [1, 5, 25, 50, 75, 95, 99])

    feat = np.concatenate([
        span,
        mean - mid,
        std,
        np.array(slice_features, dtype=np.float32),
        px - mid[0],
        py - mid[1],
        pz - mid[2]
    ])
    return feat.astype(np.float32), mid.astype(np.float32)

class CanonicalAlignmentPredictor:
    """Encapsulates the frozen Ridge canonical alignment model."""
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Ridge alignment model not found at {self.model_path}")
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            raise AlignmentError(f"Failed to deserialize Ridge alignment model: {e}")

    def compute_canonical_center(self, pts_world: np.ndarray) -> np.ndarray:
        """
        Computes canonical coordinate origin c_external in world space:
        c_external = mid + Ridge.predict(feat)
        """
        feat, mid = extract_alignment_features(pts_world)
        try:
            pred_offset = self.model.predict(feat.reshape(1, -1))[0].astype(np.float32)
            c_external = (mid + pred_offset).astype(np.float32)
            return c_external
        except Exception as e:
            raise AlignmentError(f"Failed to predict canonical alignment offset: {e}")
