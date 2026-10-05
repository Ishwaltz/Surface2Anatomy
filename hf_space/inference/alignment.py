import numpy as np
import joblib
from pathlib import Path

space_root = Path(__file__).resolve().parent.parent
repo_root = space_root.parent
local_ridge = space_root / "checkpoints" / "canonical_alignment_v3_ridge.joblib"
repo_ridge = repo_root / "experiments" / "phase10R" / "checkpoints" / "canonical_alignment_v3_ridge.joblib"
RIDGE_PATH = local_ridge if local_ridge.exists() else repo_ridge

_align_model = None

def get_alignment_model():
    global _align_model
    if _align_model is None:
        if not RIDGE_PATH.exists():
            raise FileNotFoundError(f"Ridge canonical alignment model not found at {RIDGE_PATH}")
        _align_model = joblib.load(RIDGE_PATH)
    return _align_model

def extract_alignment_features(pts_w: np.ndarray):
    """
    Extracts geometric features from external surface point cloud ONLY.
    Strictly follows tools/dataset_v3/apply_canonical_alignment.py.
    """
    p_min = pts_w.min(axis=0)
    p_max = pts_w.max(axis=0)
    span = p_max - p_min
    mid = 0.5 * (p_min + p_max)
    mean = pts_w.mean(axis=0)
    std = pts_w.std(axis=0)
    
    # 12 Normalized axial height slice features
    z_norm = (pts_w[:, 2] - p_min[2]) / (span[2] + 1e-6)
    slice_features = []
    for b in range(12):
        in_b = pts_w[(z_norm >= b / 12.0) & (z_norm < (b + 1) / 12.0)]
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
    px = np.percentile(pts_w[:, 0], [1, 5, 25, 50, 75, 95, 99])
    py = np.percentile(pts_w[:, 1], [1, 5, 25, 50, 75, 95, 99])
    pz = np.percentile(pts_w[:, 2], [1, 5, 25, 50, 75, 95, 99])
    
    feat = np.concatenate([
        span,
        mean - mid,
        std,
        np.array(slice_features, dtype=np.float32),
        px - mid[0],
        py - mid[1],
        pz - mid[2]
    ])
    return feat, mid

def compute_canonical_center(pts_world: np.ndarray) -> np.ndarray:
    """
    Computes canonical center c_external using the frozen Ridge predictor.
    """
    model = get_alignment_model()
    feat, mid = extract_alignment_features(pts_world)
    pred_offset = model.predict(feat.reshape(1, -1))[0].astype(np.float32)
    c_external = (mid + pred_offset).astype(np.float32)
    return c_external
