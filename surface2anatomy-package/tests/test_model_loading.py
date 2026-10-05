"""Tests pretrained model loading in local environment."""

from surface2anatomy import SurfaceAnatomyModel

def test_from_pretrained_cpu():
    model = SurfaceAnatomyModel.from_pretrained(model_variant="phase16_brain", device="cpu")
    assert len(model.models) == 3
    assert model.model_variant == "phase16_brain"
