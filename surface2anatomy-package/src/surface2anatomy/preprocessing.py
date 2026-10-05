"""
Complete Physical & Geometric Surface Preprocessing Pipeline.
Handles unit normalization (mm, cm, m, auto), 4096-point sampling,
canonical alignment, centering, and scale normalization.
"""

import time
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import trimesh

from surface2anatomy.constants import S_GLOBAL_MM, NUM_POINTS
from surface2anatomy.exceptions import (
    AmbiguousUnitsError, InvalidSurfaceError, PreprocessingError
)
from surface2anatomy.sampling import sample_4096_points
from surface2anatomy.alignment import CanonicalAlignmentPredictor
from surface2anatomy.results import PreprocessingMetadata

def normalize_units(pts: np.ndarray, units: str = "auto") -> Tuple[np.ndarray, str]:
    """
    Normalizes coordinates to millimeters based on explicit or auto-detected units.
    
    Parameters
    ----------
    pts : np.ndarray
        Raw coordinate array of shape (N, 3).
    units : str
        'mm', 'cm', 'm', or 'auto'.
        
    Returns
    -------
    Tuple[np.ndarray, str]
        (points_in_mm, resolved_unit_string)
    """
    u = units.lower().strip()
    if u in ("mm", "millimeter", "millimeters"):
        return pts.astype(np.float32), "millimeters"
    elif u in ("cm", "centimeter", "centimeters"):
        return (pts * 10.0).astype(np.float32), "centimeters (converted to mm)"
    elif u in ("m", "meter", "meters"):
        return (pts * 1000.0).astype(np.float32), "meters (converted to mm)"
    elif u != "auto":
        raise ValueError(f"Unknown units '{units}'. Must be 'mm', 'cm', 'm', or 'auto'.")

    # Conservative auto-detection
    bbox_span = np.ptp(pts, axis=0)
    max_dim = float(np.max(bbox_span))

    if 0.25 <= max_dim <= 2.5:
        # Patient torso or full body in meters (e.g. 0.85 m torso span)
        return (pts * 1000.0).astype(np.float32), "meters (auto-detected)"
    elif 25.0 <= max_dim <= 250.0:
        # Torso or body in centimeters (e.g. 85.0 cm torso span)
        return (pts * 10.0).astype(np.float32), "centimeters (auto-detected)"
    elif 250.0 < max_dim <= 2500.0:
        # Torso or body in millimeters (e.g. 850.0 mm torso span)
        return pts.astype(np.float32), "millimeters (auto-detected)"
    else:
        raise AmbiguousUnitsError(
            "Surface geometry scale cannot be unambiguously identified as human body anatomy.",
            max_dimension=max_dim
        )

def preprocess_surface(
    points_raw: np.ndarray,
    mesh: Optional[trimesh.Trimesh],
    alignment_predictor: CanonicalAlignmentPredictor,
    units: str = "auto",
    sampling_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, PreprocessingMetadata]:
    """
    Executes complete physical and canonical preprocessing:
    1. Normalize physical scale to millimeters.
    2. Sample exactly 4,096 points.
    3. Predict canonical anchor c_external from external geometry.
    4. Center and scale by / 500.0 mm.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, PreprocessingMetadata]
        (pts_norm_4096, c_external_mm, metadata)
    """
    t0 = time.time()

    # 1. Scale normalization
    pts_mm, resolved_unit = normalize_units(points_raw, units=units)
    orig_count = len(points_raw)

    # Validate physical human torso span
    p_min = pts_mm.min(axis=0)
    p_max = pts_mm.max(axis=0)
    dims_mm = p_max - p_min
    if np.max(dims_mm) < 100.0 or np.max(dims_mm) > 2500.0:
        raise InvalidSurfaceError(
            f"Abnormal physical body span detected ({np.max(dims_mm):.1f} mm). "
            "Expected adult human torso geometry between 100 mm and 2500 mm."
        )

    # 2. 4096-point sampling
    if mesh is not None and len(mesh.faces) > 0:
        # Scale mesh vertices if unit conversion occurred
        scale_factor = pts_mm[0, 0] / (points_raw[0, 0] + 1e-12) if points_raw[0, 0] != 0 else 1.0
        if abs(scale_factor - 1.0) > 1e-3:
            mesh_scaled = mesh.copy()
            mesh_scaled.apply_scale(scale_factor)
            pts_4096_world = sample_4096_points(mesh_scaled, seed=sampling_seed)
        else:
            pts_4096_world = sample_4096_points(mesh, seed=sampling_seed)
    else:
        pts_4096_world = sample_4096_points(pts_mm, seed=sampling_seed)

    # 3. Canonical alignment with frozen Ridge model
    c_external = alignment_predictor.compute_canonical_center(pts_4096_world)

    # 4. Centering and normalized model input
    pts_centered = pts_4096_world - c_external
    pts_norm = (pts_centered / S_GLOBAL_MM).astype(np.float32)

    dt_ms = (time.time() - t0) * 1000.0

    meta = PreprocessingMetadata(
        original_point_count=orig_count,
        sampled_point_count=NUM_POINTS,
        units=resolved_unit,
        body_width_mm=float(round(dims_mm[0], 2)),
        body_depth_mm=float(round(dims_mm[1], 2)),
        body_height_mm=float(round(dims_mm[2], 2)),
        canonical_center_mm=(
            float(round(c_external[0], 2)),
            float(round(c_external[1], 2)),
            float(round(c_external[2], 2))
        ),
        preprocessing_latency_ms=float(round(dt_ms, 2))
    )

    return pts_norm, c_external, meta
