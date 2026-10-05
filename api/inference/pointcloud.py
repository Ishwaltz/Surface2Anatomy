import io
import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple

NUM_POINTS = 4096

def parse_pcd_bytes(content: bytes) -> np.ndarray:
    """
    Lightweight PCD parser for ASCII and binary point clouds.
    """
    lines = content.split(b"\n")
    data_idx = 0
    is_ascii = True
    n_points = 0
    
    for i, line in enumerate(lines):
        line_str = line.decode("ascii", errors="ignore").strip()
        if line_str.startswith("POINTS"):
            n_points = int(line_str.split()[1])
        elif line_str.startswith("DATA"):
            if "binary" in line_str:
                is_ascii = False
            data_idx = i + 1
            break
            
    if is_ascii:
        pts = []
        for line in lines[data_idx:]:
            parts = line.decode("ascii", errors="ignore").strip().split()
            if len(parts) >= 3:
                try:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
        return np.array(pts, dtype=np.float32)
    else:
        # Binary data starts after header
        header_end = content.find(b"DATA binary\n") + len(b"DATA binary\n")
        raw_data = content[header_end:]
        # Expect float32 XYZ
        pts = np.frombuffer(raw_data[:n_points * 12], dtype=np.float32).reshape(-1, 3)
        return pts.astype(np.float32)

def load_surface_bytes(filename: str, content: bytes) -> np.ndarray:
    """
    Loads 3D coordinates from arbitrary supported surface files.
    Formats: .PLY, .PCD, .OBJ, .STL, .XYZ, .NPY
    """
    ext = Path(filename).suffix.lower()
    
    if ext == ".npy":
        arr = np.load(io.BytesIO(content))
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"NPY array must have shape (N, 3), got {arr.shape}")
        return arr[:, :3].astype(np.float32)
        
    elif ext == ".pcd":
        return parse_pcd_bytes(content)
        
    elif ext in [".ply", ".obj", ".stl"]:
        mesh_or_scene = trimesh.load(io.BytesIO(content), file_type=ext.replace(".", ""))
        if isinstance(mesh_or_scene, trimesh.Trimesh):
            if len(mesh_or_scene.vertices) >= NUM_POINTS and len(mesh_or_scene.faces) == 0:
                # Point cloud
                return np.array(mesh_or_scene.vertices, dtype=np.float32)
            elif len(mesh_or_scene.faces) > 0:
                # 3D mesh: sample points uniformly across surface
                pts, _ = trimesh.sample.sample_surface(mesh_or_scene, count=max(NUM_POINTS, len(mesh_or_scene.vertices)))
                return pts.astype(np.float32)
            else:
                return np.array(mesh_or_scene.vertices, dtype=np.float32)
        elif isinstance(mesh_or_scene, trimesh.PointCloud):
            return np.array(mesh_or_scene.vertices, dtype=np.float32)
        elif isinstance(mesh_or_scene, trimesh.Scene):
            geoms = list(mesh_or_scene.geometry.values())
            if len(geoms) > 0:
                pts = np.concatenate([np.array(g.vertices, dtype=np.float32) for g in geoms if hasattr(g, "vertices")], axis=0)
                return pts
            raise ValueError("Scene contains no valid geometry")
            
    elif ext == ".xyz" or ext == ".txt":
        pts = np.loadtxt(io.BytesIO(content))
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"XYZ file must have shape (N, 3), got {pts.shape}")
        return pts[:, :3].astype(np.float32)
        
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .PLY, .PCD, .OBJ, .STL, .XYZ, .NPY")

def detect_and_normalize_units(pts: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Detects whether geometry is in meters, centimeters, or millimeters based on span.
    Standard human torso dimensions: Width ~ 350 mm, Depth ~ 250 mm, Height ~ 400-600 mm.
    """
    span = pts.max(axis=0) - pts.min(axis=0)
    max_span = float(np.max(span))
    
    if max_span < 3.0:
        # Scale is in meters (e.g. 0.45 m)
        return (pts * 1000.0).astype(np.float32), "meters (converted to mm)"
    elif max_span < 150.0:
        # Scale is in centimeters (e.g. 45.0 cm)
        return (pts * 10.0).astype(np.float32), "centimeters (converted to mm)"
    else:
        # Already in millimeters (e.g. 450.0 mm)
        return pts.astype(np.float32), "millimeters"

def sample_4096_points(pts: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Uniformly resamples point cloud to exactly 4096 points with fixed seed for reproducibility.
    """
    np.random.seed(seed)
    N = len(pts)
    if N == NUM_POINTS:
        return pts.astype(np.float32)
    elif N > NUM_POINTS:
        idx = np.random.choice(N, size=NUM_POINTS, replace=False)
        return pts[idx].astype(np.float32)
    else:
        idx = np.random.choice(N, size=NUM_POINTS, replace=True)
        return pts[idx].astype(np.float32)
