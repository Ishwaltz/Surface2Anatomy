"""
Multi-Format 3D Geometry Ingestion and Strict Input Validation.
Supports .PLY, .PCD, .OBJ, .STL, .XYZ, .TXT, and .NPY.
Explicitly rejects 2D RGB photographic images.
"""

import io
import os
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import trimesh

from surface2anatomy.exceptions import (
    UnsupportedFormatError, InvalidSurfaceError, TooFewPointsError
)

REJECTED_2D_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
SUPPORTED_3D_EXTENSIONS = {".ply", ".pcd", ".obj", ".stl", ".xyz", ".txt", ".npy"}

def parse_pcd_bytes(content: bytes) -> np.ndarray:
    """Lightweight ASCII and binary PCD parser."""
    lines = content.split(b"\n")
    data_idx = 0
    is_ascii = True
    n_points = 0

    for i, line in enumerate(lines):
        line_str = line.decode("ascii", errors="ignore").strip()
        if line_str.startswith("POINTS"):
            parts = line_str.split()
            if len(parts) > 1:
                n_points = int(parts[1])
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
        if not pts:
            raise InvalidSurfaceError("PCD file contained zero valid 3D coordinate rows.")
        return np.array(pts, dtype=np.float32)
    else:
        header_end = content.find(b"DATA binary\n") + len(b"DATA binary\n")
        raw_data = content[header_end:]
        pts = np.frombuffer(raw_data[:n_points * 12], dtype=np.float32).reshape(-1, 3)
        return pts.astype(np.float32)

def load_surface(
    input_data: Union[str, Path, bytes, np.ndarray],
    filename_hint: Optional[str] = None
) -> Tuple[np.ndarray, Optional[trimesh.Trimesh]]:
    """
    Loads 3D surface geometry from a file path, raw bytes, or NumPy array.
    
    Returns
    -------
    Tuple[np.ndarray, Optional[trimesh.Trimesh]]
        (vertices_array, loaded_mesh_or_None)
    """
    # 1. Handle NumPy array input directly
    if isinstance(input_data, np.ndarray):
        arr = input_data
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise InvalidSurfaceError(
                f"NumPy input array must have shape (N, >=3), got {arr.shape}."
            )
        pts = arr[:, :3].astype(np.float32)
        if not np.all(np.isfinite(pts)):
            raise InvalidSurfaceError("Input coordinate array contains non-finite values (NaN or Inf).")
        if len(pts) < 64:
            raise TooFewPointsError(len(pts), minimum=64)
        return pts, None

    # 2. Handle file path or bytes
    if isinstance(input_data, (str, Path)):
        filepath = Path(input_data)
        if not filepath.is_file():
            raise FileNotFoundError(f"Surface file not found: '{filepath}'")
        ext = filepath.suffix.lower()
        filename_hint = filepath.name
        with open(filepath, "rb") as f:
            content = f.read()
    elif isinstance(input_data, bytes):
        content = input_data
        ext = Path(filename_hint).suffix.lower() if filename_hint else ""
    else:
        raise TypeError(f"Unsupported input type '{type(input_data)}'. Expected filepath, bytes, or np.ndarray.")

    # 3. Check for 2D image rejection
    if ext in REJECTED_2D_FORMATS:
        raise UnsupportedFormatError(
            "The current Surface2Anatomy model requires 3D surface geometry or "
            "depth-derived point clouds. A standard 2D RGB photograph is not currently "
            "a supported model input."
        )

    if ext and ext not in SUPPORTED_3D_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported surface format '{ext}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_3D_EXTENSIONS))}."
        )

    # 4. Parse according to extension
    if ext == ".npy":
        arr = np.load(io.BytesIO(content))
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise InvalidSurfaceError(f"NPY array must have shape (N, >=3), got {arr.shape}.")
        pts = arr[:, :3].astype(np.float32)
        mesh = None

    elif ext == ".pcd":
        pts = parse_pcd_bytes(content)
        mesh = None

    elif ext in (".xyz", ".txt"):
        text = content.decode("utf-8", errors="ignore")
        pts = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                try:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
        if not pts:
            raise InvalidSurfaceError(f"ASCII file '{filename_hint}' contained zero valid 3D points.")
        pts = np.array(pts, dtype=np.float32)
        mesh = None

    else:
        # Load mesh or point cloud via Trimesh (.PLY, .OBJ, .STL)
        file_type = ext.lstrip(".") if ext else "ply"
        try:
            loaded = trimesh.load(io.BytesIO(content), file_type=file_type)
        except Exception as e:
            raise InvalidSurfaceError(f"Failed to parse 3D geometry from {filename_hint or 'input'}: {e}")

        if isinstance(loaded, trimesh.Scene):
            # Combine all geometries in scene
            geoms = [g for g in loaded.geometry.values() if hasattr(g, "vertices")]
            if not geoms:
                raise InvalidSurfaceError("Trimesh Scene contained no geometric vertex data.")
            pts = np.vstack([g.vertices for g in geoms]).astype(np.float32)
            mesh = None
        elif isinstance(loaded, trimesh.Trimesh):
            pts = np.asarray(loaded.vertices, dtype=np.float32)
            mesh = loaded
        elif hasattr(loaded, "vertices"):
            pts = np.asarray(loaded.vertices, dtype=np.float32)
            mesh = None
        else:
            raise InvalidSurfaceError(f"Unable to extract vertex coordinates from {filename_hint or 'file'}.")

    # 5. Coordinate integrity validation
    if not np.all(np.isfinite(pts)):
        raise InvalidSurfaceError("Uploaded surface contains invalid non-finite coordinates (NaN or Inf).")
    if len(pts) < 64:
        raise TooFewPointsError(len(pts), minimum=64)

    return pts, mesh
