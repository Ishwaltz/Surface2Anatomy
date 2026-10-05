# Included trained checkpoints

This repository contains the existing trained checkpoint files directly in Git.
Cloning the repository downloads the weights; Git LFS and separate model downloads
are not required. The files are unchanged from the source project.

| Model | Location | Files |
| --- | --- | --- |
| Phase 10R ensemble | `experiments/phase10R/checkpoints/` | `C4_Proposed_seed42.pt`, `C4_Proposed_seed43.pt`, `C4_Proposed_seed44.pt` |
| Brain-aware ensemble | `experiments/phase16_brain/checkpoints/` | `BrainAware_seed42.pt`, `BrainAware_seed43.pt`, `BrainAware_seed44.pt` |
| Sex-aware ensemble | `experiments/sex_aware/checkpoints/` | `S2A_SexAware_seed42.pt`, `S2A_SexAware_seed43.pt`, `S2A_SexAware_seed44.pt` |
| Sex classifier | `experiments/sex_aware/checkpoints/` | `S2A_SexClassifier.pt` |
| Canonical alignment | `experiments/phase10R/checkpoints/` | `canonical_alignment_v3_ridge.joblib` |

SHA-256 checksums for all eleven files are recorded in `CHECKPOINTS.sha256`.
The six Phase 10R / brain-aware weights and alignment artifact match the
existing Python package manifest. The API and Python package already discover
the Phase 10R / brain-aware weights at these paths.

## Local package use

Run from the repository root after installing the package:

```bash
python -m pip install -e ./surface2anatomy-package
```

```python
from surface2anatomy import SurfaceAnatomyModel

model = SurfaceAnatomyModel.from_pretrained(
    model_variant="phase10r", device="cpu", local_files_only=True
)
```

## Dataset exclusions

Raw datasets, medical scans, preprocessed point-cloud datasets, external
validation surfaces, dataset array archives, case-level coordinate/prediction
exports, demo patient inputs, local environments, and intermediate training
outputs are excluded. Dataset preparation and training code, project
documentation, benchmark reports, figures, and the checkpoints above are included.
Older C0/C1/C5 `.pt` files hold cached benchmark predictions rather than trained
model state and are excluded.

Training, evaluation, and demo tests that reference omitted data require that
data to be supplied locally. Checkpoints retain their original training metadata.
This upload verifies file integrity; it does not retrain models or revalidate
the performance claims in the existing reports.
