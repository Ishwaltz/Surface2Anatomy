"""Tests package import, versioning, and top-level namespace exports."""

import surface2anatomy as s2a

def test_version():
    assert hasattr(s2a, "__version__")
    assert s2a.__version__ == "0.1.0"

def test_exports():
    expected = [
        "SurfaceAnatomyModel", "list_targets", "resolve_target",
        "load_surface", "sample_4096_points", "PredictionResult",
        "SinglePrediction", "CANONICAL_TARGET_NAMES", "NUM_POINTS"
    ]
    for name in expected:
        assert hasattr(s2a, name), f"Missing export: {name}"
