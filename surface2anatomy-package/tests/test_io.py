"""Tests 3D surface loading, format support, and 2D RGB image rejection."""

import pytest
import numpy as np
from pathlib import Path
from surface2anatomy.io import load_surface
from surface2anatomy.exceptions import (
    UnsupportedFormatError, InvalidSurfaceError, TooFewPointsError
)

def test_load_numpy():
    arr = np.random.randn(500, 3).astype(np.float32)
    pts, mesh = load_surface(arr)
    assert pts.shape == (500, 3)
    assert mesh is None

def test_load_numpy_with_extra_columns():
    arr = np.random.randn(500, 6).astype(np.float32) # XYZ + Normals
    pts, mesh = load_surface(arr)
    assert pts.shape == (500, 3)

def test_reject_2d_images():
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        with pytest.raises(UnsupportedFormatError) as exc_info:
            load_surface(b"fake_image_bytes", filename_hint=f"patient{ext}")
        assert "standard 2D RGB photograph is not currently a supported model input" in str(exc_info.value)

def test_reject_few_points():
    arr = np.random.randn(30, 3).astype(np.float32)
    with pytest.raises(TooFewPointsError):
        load_surface(arr)

def test_reject_non_finite():
    arr = np.random.randn(100, 3).astype(np.float32)
    arr[10, 1] = np.nan
    with pytest.raises(InvalidSurfaceError):
        load_surface(arr)
