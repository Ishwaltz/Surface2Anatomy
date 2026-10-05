"""Tests single and multi-target forward predictions."""

import numpy as np
from surface2anatomy import SurfaceAnatomyModel

def test_predict_synthetic():
    model = SurfaceAnatomyModel.from_pretrained(model_variant="phase16_brain", device="cpu")
    # Synthetic patient torso points in mm (~400mm width, 250mm depth, 700mm height)
    pts = np.random.uniform(-200.0, 200.0, (1000, 3)).astype(np.float32)
    pts[:, 2] = np.random.uniform(-350.0, 350.0, 1000)

    res = model.predict(pts, target="spleen", units="mm")
    assert "spleen" in res
    assert len(res.centroid_mm) == 3
    assert res["spleen"].ensemble_disagreement_mm >= 0.0

def test_predict_multiple():
    model = SurfaceAnatomyModel.from_pretrained(model_variant="phase16_brain", device="cpu")
    pts = np.random.uniform(-200.0, 200.0, (1000, 3)).astype(np.float32)
    pts[:, 2] = np.random.uniform(-350.0, 350.0, 1000)

    res = model.predict_multiple(pts, targets=["liver", "spleen", "kidney_left"], units="mm")
    assert len(res) == 3
    assert "liver" in res
    assert "spleen" in res
    assert "kidney_left" in res
