# AMOS-22 External Body Surface Extraction & Normalization Pipeline

## 1. Pipeline Overview & Surface Blindness Invariant
In strict accordance with Phase 11 invariants, the external body surface point cloud is extracted exclusively from raw CT and MRI volumetric voxel intensities, ensuring absolute zero leakage from internal organ ground truth masks.

### Stage 1: Volumetric Canonical Reorientation
- Each native NIfTI volume (`.nii.gz`) is mapped to the canonical RAS coordinate frame (+X Right, +Y Anterior, +Z Superior) using `nibabel.as_closest_canonical`.
- Spatial voxel zooms, affine transforms, and direction cosines are strictly preserved.

### Stage 2: Intensity-Only External Surface Segmentation
- **CT Scans:** Thresholding at $\text{HU} > -500$, followed by 3D connected-component analysis to isolate the primary human body envelope, slice-by-slice axial hole filling, and elimination of scanner table artifacts.
- **MRI Scans:** Background noise estimation from 3D corner subvolumes, foreground thresholding at $\mu_{\text{bg}} + 1.5\sigma_{\text{bg}}$, 3D morphological closing, and slice-by-slice hole filling.

### Stage 3: Isosurface Extraction & Metric Mapping
- Triangular surface meshing via marching cubes at level set 0.5.
- Voxel vertex coordinates $(i, j, k)$ are mapped directly to continuous Euclidean world coordinates in millimeters:
  $$\mathbf{x}_{\text{world}} = \mathbf{A} \begin{bmatrix} i \\ j \\ k \\ 1 \end{bmatrix}$$
  where $\mathbf{A}$ is the canonical $4 \times 4$ affine transformation matrix.

### Stage 4: Body Centering & Scale Normalization
- Bounding box midpoint is computed:
  $$\mathbf{c}_{\text{body}} = \frac{\min(\mathbf{x}_{\text{world}}) + \max(\mathbf{x}_{\text{world}})}{2}$$
- Exactly 4,096 points are sampled uniformly over the surface mesh.
- Centered and normalized into frozen model space:
  $$\mathbf{p}_{\text{norm}} = \frac{\mathbf{p}_{\text{world}} - \mathbf{c}_{\text{body}}}{S_{\text{global}}}, \quad S_{\text{global}} = 500.0\text{ mm}$$

## 2. Verification & Integrity
- 100% of 259 evaluated scans underwent end-to-end mesh reconstruction.
- Zero non-finite coordinates (0 NaNs, 0 Infs).
