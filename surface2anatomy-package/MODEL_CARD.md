# Model Card: Surface2Anatomy

## Model Details
- **Architecture**: Multi-Scale PointNet++ Encoder + 4-Layer Target-Query Transformer Decoder.
- **Task**: 3D Internal Anatomy Localization from External Body Geometry.
- **Model Version**: 0.1.0 (Frozen Seeds 42, 43, 44).
- **Global Normalization Scale**: $S_{\text{global}} = 500.0\text{ mm}$.
- **Input Representation**: 4,096 external 3D surface points.
- **Runtime Input**: External 3D surface geometry only. Zero CT/MRI voxels required at inference.
- **Offline Supervision**: CT/MRI-derived annotations were used offline for supervision; no CT/MRI voxel data is required as model input at inference.

## Intended Use
- Non-invasive surgical navigation and robotic probe placement research.
- Anthropometric anatomical localization studies.

## Training Data & Benchmarks
- Dataset V3 ($N=1,668$ subjects), split into 1,334 train, 166 val, 168 locked test.
- Tested across 104 primary benchmark targets and 121 extended structures.
