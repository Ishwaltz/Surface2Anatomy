"""Tests deterministic point sampling."""

import numpy as np
from surface2anatomy.sampling import sample_4096_points
from surface2anatomy.constants import NUM_POINTS

def test_sampling_shape():
    raw_pts = np.random.randn(8000, 3).astype(np.float32)
    sampled = sample_4096_points(raw_pts, seed=42)
    assert sampled.shape == (NUM_POINTS, 3)

def test_sampling_determinism():
    raw_pts = np.random.randn(6000, 3).astype(np.float32)
    s1 = sample_4096_points(raw_pts, seed=42)
    s2 = sample_4096_points(raw_pts, seed=42)
    assert np.allclose(s1, s2)

def test_sampling_upsample():
    raw_pts = np.random.randn(500, 3).astype(np.float32)
    sampled = sample_4096_points(raw_pts, seed=42)
    assert sampled.shape == (NUM_POINTS, 3)
