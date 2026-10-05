# Quickstart Guide

Install `surface2anatomy`:

```bash
pip install surface2anatomy
```

### Basic Python Usage

```python
from surface2anatomy import SurfaceAnatomyModel

# Initialize frozen ensemble
model = SurfaceAnatomyModel.from_pretrained()

# Predict internal organ centroid
result = model.predict("patient_surface.ply", target="spleen", units="mm")

print("Centroid (mm):", result.centroid_mm)
print("Uncertainty (mm):", result["spleen"].ensemble_disagreement_mm)
```
