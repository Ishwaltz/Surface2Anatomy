# Surface2Anatomy

> **Target-conditioned localization of hidden internal anatomy from external 3D body-surface geometry.**

[![PyPI Version](https://img.shields.io/pypi/v/surface2anatomy.svg?color=465133)](https://pypi.org/project/surface2anatomy/)
[![License](https://img.shields.io/badge/License-Apache_2.0-olive.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Live Web Demo](https://img.shields.io/badge/Live%20Demo-Vercel-success)](https://web-self-theta-51.vercel.app)

---

## Overview

**Surface2Anatomy** solves the inverse anatomical localization problem: predicting the exact 3D spatial coordinates (centroids) of internal organs, vertebrae, and great vessels directly from a patient's **external 3D body surface scan** without requiring pre-operative or intra-operative CT or MRI radiation.

> **Scientific Grounding**:  
> *CT/MRI-derived annotations were used offline for supervision; no CT/MRI voxel data is required as model input at inference.*

---

## Installation

```bash
pip install surface2anatomy
```

For visualization extensions:
```bash
pip install "surface2anatomy[viz]"
```

---

## Quick Start

```python
from surface2anatomy import SurfaceAnatomyModel

# 1. Load the frozen 3-seed neural ensemble
model = SurfaceAnatomyModel.from_pretrained()

# 2. Predict 3D centroid from an external surface scan
result = model.predict(
    "patient_surface.ply",
    target="spleen",
    units="mm"
)

# 3. Access millimeter coordinates and uncertainty
print(f"Spleen Centroid (mm): {result.centroid_mm}")
# Conceptual output: (-90.26, 23.26, -55.14)

print(f"Ensemble Disagreement: {result['spleen'].ensemble_disagreement_mm} mm")
```

---

## Supported Input Formats

Surface2Anatomy accepts external 3D geometry from optical surface scanners, depth cameras (Intel RealSense, LiDAR), and photogrammetric reconstructions:

| Category | File Formats | Details |
|---|---|---|
| **Point Clouds** | `.ply`, `.pcd`, `.xyz`, `.txt`, `.npy` | Arbitrary point density; uniformly sampled to 4,096 points |
| **Surface Meshes** | `.ply`, `.obj`, `.stl` | Area-weighted surface sampling across mesh faces |
| **In-Memory** | `np.ndarray` | Direct $(N, 3)$ or $(N, \ge 3)$ coordinate arrays |

> ⚠️ **2D Photographs**: Standard 2D RGB photographs (`.jpg`, `.jpeg`, `.png`) are **not** 3D surfaces and are rejected with an explicit diagnostic message.

---

## Python API

### 1. Simultaneous Multi-Organ Query (Single Forward Pass)

Query multiple internal structures in a single neural network forward pass:

```python
result = model.predict_multiple(
    "patient_surface.ply",
    targets=["liver", "spleen", "kidney_left", "kidney_right", "heart"],
    units="mm"
)

for target_name, pred in result.items():
    print(f"{target_name:12s} -> {pred.centroid_mm} mm (±{pred.ensemble_disagreement_mm} mm)")
```

### 2. Predict All 121 Anatomical Targets

```python
result = model.predict_all("patient_surface.ply")
df = result.to_dataframe()
print(df.head())
```

### 3. In-Memory NumPy Coordinate Prediction

```python
import numpy as np

# Coordinates in millimeters (N x 3)
pts = np.random.uniform(-180, 180, (5000, 3))
result = model.predict(pts, target="liver", units="mm")
print("Liver centroid:", result.centroid_mm)
```

### 4. High-Throughput Cohort Batch Processing

```python
results = model.predict_batch(
    ["patient_01.ply", "patient_02.ply", "patient_03.ply"],
    targets=["liver", "spleen"]
)
```

---

## Command Line Interface (CLI)

The package installs the `surface2anatomy` command-line tool:

```bash
# Display model specifications and cache paths
surface2anatomy info

# List supported anatomical landmarks
surface2anatomy list-targets --category "Abdominal & Visceral"

# Validate a 3D surface scan
surface2anatomy validate-surface patient.ply

# Predict internal organ position
surface2anatomy predict patient.ply --target spleen

# Predict multiple organs and export to JSON or CSV
surface2anatomy predict patient.ply \
    --targets liver spleen kidney_left kidney_right \
    --units mm \
    --output result.json

# Batch process an entire patient folder
surface2anatomy batch ./patients/ --targets liver spleen --output cohort.csv
```

### Example CLI Output

```text
Surface2Anatomy 0.1.0

Target: spleen

Predicted centroid:
X =   -90.26 mm
Y =   +23.26 mm
Z =   -55.14 mm

Ensemble disagreement:
4.30 mm

Input:
4096 sampled surface points

Runtime imaging:
External surface only
```

---

## Supported Anatomy (121 Targets)

Surface2Anatomy features a comprehensive anatomical ontology mapped to the TotalSegmentator v2, AMOS22, and CT-ORG registries:

- **Abdominal & Visceral**: Liver, spleen, pancreas, gallbladder, stomach, duodenum, small bowel, colon, kidneys (left/right), adrenal glands.
- **Thoracic & Respiratory**: Heart, trachea, esophagus, thyroid gland, lung lobes (5 lobes).
- **Vascular**: Aorta, superior/inferior vena cava, pulmonary veins, brachiocephalic trunk, carotid & subclavian arteries, iliac vessels.
- **Spine & Ribs**: Vertebrae C1–L5, sacrum, spinal cord, ribs 1–12 (bilateral), sternum.
- **Pelvic & Reproductive**: Urinary bladder, prostate, uterus, ovaries, vagina, hips.
- **Head**: Brain, skull.

View the full list at any time:
```python
import surface2anatomy as s2a
print(s2a.list_targets())
```

---

## Architecture

The system consists of two synergistic deep learning components:

```text
EXTERNAL 3D BODY SURFACE (N points)
               ↓
4096 XYZ Normalized Surface Points
               ↓
Frozen External-Geometry Canonical Alignment (c_external)
               ↓
Multi-Scale PointNet++ Surface Encoder
(SA1: 1024 tokens | SA2: 256 tokens | SA3: 64 tokens | Global: 1024)
               ↓
320 Hierarchical Surface Memory Tokens (256 local + 64 coarse)
               ↓
Target-Query Transformer Decoder (4 Layers, 8 Heads, d=256)
               ↓
Predicted Anatomical Centroids (117 Query Slots)
               ↓
3-Seed Frozen Ensemble Average (Seeds 42, 43, 44)
```

---

## Preprocessing & Canonical Alignment

To ensure physical consistency across diverse patient body sizes, the pipeline employs a **purely external canonical alignment model**:
1. **Physical Scale Verification**: Resolves millimeters, centimeters, and meters.
2. **Deterministic Point Sampling**: Samples exactly 4,096 points with reproducible seed control.
3. **Canonical Alignment ($c_{\text{external}}$)**: Evaluates 105 external-only morphology features (axial slice aspect ratios, curvature proxies, bounding box percentiles) with a frozen Ridge regressor. **No internal CT landmarks or organ masks are used at inference.**
4. **Centering & Normalization**: Normalizes coordinates relative to $c_{\text{external}}$ by $S_{\text{global}} = 500.0\text{ mm}$.

---

## Model Weights & Caching

The wheel package is ultra-lightweight (< 100 KB) and contains no heavy binary checkpoint blobs. Pretrained weights are hosted externally and downloaded on first call:
- Cached locally in `~/.cache/surface2anatomy/` (using `platformdirs`).
- Every downloaded checkpoint is cryptographically verified against its frozen SHA256 checksum.
- Offline environments are supported via `SurfaceAnatomyModel.from_pretrained(local_files_only=True)`.

---

## Validation & Accuracy

Evaluated across the locked held-out test cohort ($N=168$ subjects, Dataset V3):

| Metric | Proposed Transformer Ensemble |
|---|:---:|
| **Macro Mean Radial Error (MRE)** | **23.22 mm (~0.91 inches)** |
| **Micro Mean Radial Error** | **22.84 mm** |
| **Median Error** | **18.79 mm** |
| **Kidneys Error** | **< 12.0 mm** |
| **Heart Error** | **< 16.0 mm** |
| **Inference Latency (GPU)** | **~25 ms** |
| **Inference Latency (CPU)** | **~280 ms** |

---

## Research Disclaimer

> **RESEARCH PROTOTYPE**: Surface2Anatomy is an open scientific software package intended strictly for academic research, pre-clinical computational simulation, and anatomical exploration. It is **not** approved by regulatory agencies (e.g., FDA, CE) for standalone clinical diagnosis, surgical navigation, or autonomous procedural intervention.

## License

Surface2Anatomy is released under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

