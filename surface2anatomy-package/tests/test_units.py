"""Tests unit normalization and ambiguous scale detection."""

import pytest
import numpy as np
from surface2anatomy.preprocessing import normalize_units
from surface2anatomy.exceptions import AmbiguousUnitsError

def test_explicit_units():
    pts_m = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.8]], dtype=np.float32)
    pts_mm, u = normalize_units(pts_m, units="m")
    assert np.allclose(pts_mm[1], [500.0, 300.0, 800.0])
    assert "meters" in u

    pts_cm = np.array([[0.0, 0.0, 0.0], [50.0, 30.0, 80.0]], dtype=np.float32)
    pts_mm, u = normalize_units(pts_cm, units="cm")
    assert np.allclose(pts_mm[1], [500.0, 300.0, 800.0])

def test_auto_units():
    # Meter scale torso (~0.85m span)
    pts_m = np.random.uniform(0.0, 0.85, (200, 3)).astype(np.float32)
    pts_mm, u = normalize_units(pts_m, units="auto")
    assert "meters (auto-detected)" in u
    assert pts_mm.max() > 100.0

    # Millimeter scale torso (~850mm span)
    pts_mm_in = np.random.uniform(0.0, 850.0, (200, 3)).astype(np.float32)
    pts_mm, u = normalize_units(pts_mm_in, units="auto")
    assert "millimeters (auto-detected)" in u

def test_ambiguous_scale_error():
    # Microscopic or celestial scale (e.g. max span = 10,000)
    pts_huge = np.random.uniform(0.0, 10000.0, (200, 3)).astype(np.float32)
    with pytest.raises(AmbiguousUnitsError):
        normalize_units(pts_huge, units="auto")
