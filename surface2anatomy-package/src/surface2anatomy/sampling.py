"""
Deterministic 4,096-Point Surface Geometry Sampling.
Strictly adheres to the sampling policies established in the frozen experimental benchmark.
"""

from typing import Optional, Union, Tuple
import numpy as np
import trimesh

from surface2anatomy.constants import NUM_POINTS, DEFAULT_SAMPLING_SEED
from surface2anatomy.exceptions import TooFewPointsError

def sample_4096_points(
    points_or_mesh: Union[np.ndarray, trimesh.Trimesh],
    seed: int = DEFAULT_SAMPLING_SEED
) -> np.ndarray:
    """
    Deterministically samples exactly 4,096 points from a 3D point cloud or surface mesh.
    
    Parameters
    ----------
    points_or_mesh : np.ndarray or trimesh.Trimesh
        Input coordinates (N, 3) or watertight/open surface mesh.
    seed : int
        Deterministic random generator seed (default: 42).
        
    Returns
    -------
    np.ndarray
        Array of shape (4096, 3) in float32 precision.
    """
    if isinstance(points_or_mesh, trimesh.Trimesh):
        mesh = points_or_mesh
        if len(mesh.faces) > 0:
            # Uniform area-weighted surface sampling
            pts, _ = trimesh.sample.sample_surface(mesh, count=NUM_POINTS, seed=seed)
            return pts.astype(np.float32)
        pts = np.asarray(mesh.vertices, dtype=np.float32)
    else:
        pts = np.asarray(points_or_mesh, dtype=np.float32)

    if len(pts) < 64:
        raise TooFewPointsError(len(pts), minimum=64)

    rng = np.random.RandomState(seed)
    N = len(pts)

    if N == NUM_POINTS:
        return pts.copy().astype(np.float32)
    elif N > NUM_POINTS:
        indices = rng.choice(N, size=NUM_POINTS, replace=False)
        return pts[indices].astype(np.float32)
    else:
        indices = rng.choice(N, size=NUM_POINTS, replace=True)
        return pts[indices].astype(np.float32)
