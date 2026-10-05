import time
import numpy as np
from typing import Dict, Any, Tuple
from api.inference.pointcloud import load_surface_bytes, detect_and_normalize_units, sample_4096_points
from api.inference.alignment import compute_canonical_center

S_GLOBAL_MM = 500.0

def preprocess_surface_bytes(filename: str, content: bytes) -> Dict[str, Any]:
    """
    Executes complete physical and canonical preprocessing:
    1. Parse arbitrary 3D surface formats (.PLY, .PCD, .OBJ, .STL, .XYZ, .NPY)
    2. Check finite coordinates, reject NaN/Inf
    3. Auto-detect physical units (meters/centimeters/millimeters) -> normalize to mm
    4. Deterministically sample exactly 4,096 points
    5. Compute canonical external Ridge alignment anchor c_external
    6. Center surface (pts - c_external) and scale by / 500.0
    
    Returns dictionary with tensors, metadata, and timing.
    """
    t0 = time.time()
    
    # 1. Parse surface
    pts_raw = load_surface_bytes(filename, content)
    
    # 2. Validation
    if len(pts_raw) < 64:
        raise ValueError(f"Geometry contains too few points ({len(pts_raw)}). Minimum required is 64 points.")
    if not np.all(np.isfinite(pts_raw)):
        raise ValueError("Uploaded geometry contains invalid non-finite coordinates (NaN or Inf).")
        
    # 3. Unit normalization to millimeters
    pts_mm, detected_unit = detect_and_normalize_units(pts_raw)
    
    # Check physical plausibility (human torso bounds)
    bbox_min = pts_mm.min(axis=0)
    bbox_max = pts_mm.max(axis=0)
    dims_mm = bbox_max - bbox_min
    
    if np.max(dims_mm) < 50.0 or np.max(dims_mm) > 2500.0:
        raise ValueError(f"Abnormal physical scale detected (max dimension: {np.max(dims_mm):.1f} mm). Expected human-scale geometry between 50 mm and 2500 mm.")
        
    # 4. 4096-point sampling
    pts_4096_world = sample_4096_points(pts_mm, seed=42)
    
    # 5. Canonical alignment with frozen Ridge model
    c_external = compute_canonical_center(pts_4096_world)
    
    # 6. Centered & normalized model input
    pts_centered = pts_4096_world - c_external
    pts_norm = (pts_centered / S_GLOBAL_MM).astype(np.float32)
    
    dt_ms = (time.time() - t0) * 1000.0
    
    return {
        "pts_norm": pts_norm,
        "pts_centered": pts_centered,
        "pts_world": pts_4096_world,
        "c_external": c_external,
        "detected_unit": detected_unit,
        "raw_point_count": len(pts_raw),
        "body_dimensions_mm": dims_mm.tolist(),
        "bbox_min_mm": bbox_min.tolist(),
        "bbox_max_mm": bbox_max.tolist(),
        "preprocessing_latency_ms": round(dt_ms, 2)
    }
