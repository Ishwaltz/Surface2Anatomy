"""Tests canonical Ridge alignment feature extraction."""

import numpy as np
from surface2anatomy.alignment import extract_alignment_features

def test_extract_features_shape():
    pts = np.random.randn(4096, 3).astype(np.float32)
    feat, mid = extract_alignment_features(pts)
    # span(3) + mean-mid(3) + std(3) + 12*5 slices(60) + 3*7 percentiles(21) = 90
    assert len(mid) == 3
    assert len(feat) > 50
    assert np.all(np.isfinite(feat))
