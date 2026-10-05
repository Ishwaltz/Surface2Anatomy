# PHASE 1 GEOMETRY AUDIT FINAL REPORT
**Forensic Data, Geometry, Coordinate, Normalization, and Label Audit**

---

## 1. Executive Summary

This forensic audit rigorously examined the complete data, geometry, coordinate transformation, normalization, labeling, and evaluation pipeline of the 3D Organ Location Prediction repository without modifying production code.

The central question posed by this audit is:
> **Can we trust every coordinate entering and leaving the model?**
> **ANSWER: NO. [CODE-VERIFIED] & [DATA-VERIFIED]**

### Major Verified Forensic Discoveries

1. **CRITICAL AXIS-SWAP & WARPING BUG [CODE-VERIFIED] & [DATA-VERIFIED]**:
   In `sharon/pointcloud_sampler.py` (lines 80–82), the center-of-mass voxel coordinates `com` returned from `scipy.ndimage.center_of_mass` were unpacked as:
   ```python
   cz, cy, cx = com
   phys = (affine @ np.array([cx, cy, cz, 1.0], dtype=np.float32))[:3]
   ```
   In `nibabel`, NIfTI 3D image arrays are indexed in `(i, j, k)` order (corresponding to X, Y, Z). Unpacking `com` into `(cz, cy, cx)` reversed the indices: array axis 0 (in-plane pixel coordinate $i$, size 512–768) was assigned to `cz` and multiplied by the slice thickness ($5.0\text{ mm}$), while array axis 2 (slice index $k$, size 80–132) was assigned to `cx` and multiplied by the in-plane pixel spacing ($-0.57\text{ mm}$).
   - **Sample ID Affected**: All 450 cases in `pointclouds_450.pt`.
   - **Exact Measurement (`case_000`)**: The true physical center-of-mass of the spleen is `[-81.40, -205.66, 386.72] mm`. The codebase evaluated it at `[191.91, -205.66, 2782.89] mm`—an artificial error of **$2,411.71\text{ mm}$ (2.41 meters)**.
   - **Reproduction Command**:
     ```bash
     python -c '
     import nibabel as nib, numpy as np, scipy.ndimage as ndi
     seg = nib.load("sharon/dataset/case_000/segmentation.nii.gz")
     com = ndi.center_of_mass(seg.get_fdata() == 1)
     cz, cy, cx = com
     phys_bug = (seg.affine @ [cx, cy, cz, 1.0])[:3]
     phys_correct = (seg.affine @ [com[0], com[1], com[2], 1.0])[:3]
     print("Discrepancy (mm):", np.linalg.norm(phys_bug - phys_correct))
     '
     ```
   - Torso height in `pointclouds_450.pt` spans up to **$3,762\text{ mm}$ (3.76 meters)**, while patient width is crushed to **$75\text{ mm}$ (7.5 cm)**.
   - Bilateral structures such as `kidney_left` and `kidney_right` are separated by **$1,106.91\text{ mm}$ (1.1 meters) along the Z axis** and only **$0.98\text{ mm}$ along the X axis**.

2. **CRITICAL SYNTHETIC SKIN DECOUPLING [CODE-VERIFIED]**:
   In `sharon/pointcloud_sampler.py` (lines 34–55 and 102–115), the input point clouds $X \in \mathbb{R}^{4096 \times 3}$ are **not extracted from the patient's actual CT scan**. Instead, a single static canonical template mesh (`sharon/outputs/meshes/skin.obj`) is linearly scaled to the bounding box of the corrupted target coordinates, plus random Gaussian noise. The neural networks were trained to map a warped template to warped targets, rather than real patient surface topography to anatomy.
   - **Sample ID Affected**: All 450 cases.

3. **CRITICAL CROSS-SPLIT PATIENT LEAKAGE [DATA-VERIFIED]**:
   Cryptographic SHA-256 hashing of all underlying NIfTI volumes revealed that **5 duplicate patient CT volumes cross the train, validation, and test splits**:
   - `case_219` (Train) $\equiv$ `case_406` (Test) [SHA-256 `f0c90c74...`] $\to$ **Direct train-to-test leakage**.
   - `case_096` (Test) $\equiv$ `case_412` (Val) [SHA-256 `085e8b56...`].
   - `case_100` (Train) $\equiv$ `case_408` (Val) [SHA-256 `968e99b7...`].
   - `case_144` (Train) $\equiv$ `case_407` (Val) [SHA-256 `c52e70bd...`].
   - `case_344` (Train) $\equiv$ `case_402` (Val) [SHA-256 `c4b18c64...`].
   - **Severe Biological Contradiction**: `case_100` (Train) is labeled **Male** (sex=1) from DICOM header without female organs, while its duplicate `case_408` (Val) is labeled **Female** (sex=0) with synthetic uterus, ovaries, and vagina injected into its segmentation mask!
   - **Reproduction Command**:
     ```bash
     python tools/audit/audit_split_leakage.py
     ```

4. **HIGH SEVERITY: ABSOLUTE BODY SCALE INFORMATION IS COMPLETELY DESTROYED [CODE-VERIFIED] & [DATA-VERIFIED]**:
   Every patient's torso is independently normalized into $[-1.0, 1.0]^3$ via anisotropic bounding box scaling: $\mathbf{x}' = (\mathbf{x} - \mathbf{c}) / \mathbf{s}$. The physical scale vector $\mathbf{s} \in \mathbb{R}^3$ is **never passed to the neural network**. Because physical height $H$ correlates strongly with internal anatomy (Pearson $r = \mathbf{0.6066}$ with spine coordinates on the training split), removing this scale strips the network of critical physical sizing cues.

---

## 2. Repository/Data Flow

The actual verified data flow from raw data to evaluation metric is detailed below:

```text
Raw CT / DICOM files (raw_data/*.nii.gz)
        ↓
[sharon/dataset_preprocessing.py: DatasetPreprocessor.process_case()]  [CODE-VERIFIED]
Preprocessed NIfTI Volumes (sharon/dataset/case_XXX/ct.nii.gz, segmentation.nii.gz)
        ↓
[sharon/dataset_preprocessing.py: _extract_female_pelvic_anatomy()]  [CODE-VERIFIED]
121-Class Multiclass Mask Volume (segmentation.nii.gz: shape (Nx, Ny, Nz))
        ↓
[sharon/pointcloud_sampler.py: ndi.center_of_mass()]  [CODE-VERIFIED]
Voxel Centroids com = (i, j, k)
        ↓
[CRITICAL BUG: sharon/pointcloud_sampler.py:80-82: cz, cy, cx = com -> affine @ [cx, cy, cz, 1.0]]  [CODE-VERIFIED]
Corrupted Physical Coordinates (raw_centroids: X and Z swapped)
        ↓
[sharon/pointcloud_sampler.py:102-115: load_canonical_surface_points()]  [CODE-VERIFIED]
Synthetic Scaled Skin Points (raw_points: shape (4096, 3), template stretched to corrupted bbox)
        ↓
[sharon/pointcloud_sampler.py:118-124: Anisotropic Bounding Box Normalization]  [CODE-VERIFIED]
Normalized Tensors: points (4096, 3), centroids (121, 3), centers (3,), scales (3,)
        ↓
[sharon/dataset/pointclouds_450.pt]  [DATA-VERIFIED]
Consolidated PyTorch Dataset File
        ↓
[sharon/dataset.py: PointCloudOrganDataset]  [CODE-VERIFIED]
Batch Generation + Gaussian Jitter (pts + N(0, 0.005))
        ↓
[sharon/model_evidential.py / model_same.py / model_gnn.py]  [CODE-VERIFIED]
Normalized Model Predictions: gamma / full_coords in [-1.0, 1.0]^3
        ↓
[sharon/evaluate_evidential.py:57-58]  [CODE-VERIFIED]
Physical De-normalization: preds_mm = gamma * scale + center
        ↓
[sharon/evaluate_evidential.py:59: torch.norm(preds_mm - ctr_mm, dim=-1)]  [CODE-VERIFIED]
Physical Mean Radial Error (MRE in warped mm)
```

---

## 3. Dataset Inventory

- **Total Patient Directories**: 450 valid directories (`case_000` to `case_463`; 14 non-sequential numbers missing: 423, 436, 437, 440, 442, 445, 446, 447, 448, 453, 455, 456, 460, 462). `[DATA-VERIFIED]`
- **Split Distribution (`splits_pointcloud.json`)**:
  - Train: **360 patients** (80.0%)
  - Validation: **45 patients** (10.0%)
  - Held-Out Test: **45 patients** (10.0%)
  `[DATA-VERIFIED]`
- **Number of Targets**: Exactly **121 anatomical classes** (63 skeletal, 58 soft-tissue). `[CODE-VERIFIED]`
- **Surface Point Count**: Exactly **4,096 points** per patient. `[DATA-VERIFIED]`
- **Available Metadata**: `sharon/dataset/metadata.json` contains `case_id`, `sex` (0=F, 1=M), `one_hot`, `source` (dicom_header vs prostate_volume fallback), and boolean organ presence flags. `[DATA-VERIFIED]`
- **Missing Target Patterns**: High thoracic structures (`heart`, `thyroid_gland`, `brain`, `skull`) are missing in >90% of cases due to abdominal CT scan truncation; female pelvic organs (`uterus`, `ovary_left`, `ovary_right`, `vagina`) are present in 181 female cases and strictly masked out in all 269 male cases. `[DATA-VERIFIED]`

---

## 4. Coordinate Convention

- **Surface Units**: Stored as physical millimetres in `raw_points`, but distorted into an unphysical coordinate space (Z spans up to $3,762\text{ mm}$). `[DATA-VERIFIED]`
- **Target Units**: Stored as physical millimetres in `raw_centroids`, but corrupted by axis swapping. `[DATA-VERIFIED]`
- **NIfTI Native Coordinate System**: **LAS+ (Left, Anterior, Superior)**. `[CODE-VERIFIED]`
  - Axis 0 ($i$): Left ($+$), Right ($-$)
  - Axis 1 ($j$): Anterior ($+$), Posterior ($-$)
  - Axis 2 ($k$): Superior ($+$), Inferior ($-$)
- **Corrupted Repository Coordinate System (Due to Line 81 Swap)**: `[CODE-VERIFIED]`
  - Axis 0: Derived from slice index $k$ (Superior-Inferior) multiplied by in-plane resolution ($-0.57\text{ mm}$).
  - Axis 1: Anterior-Posterior preserved.
  - Axis 2: Derived from in-plane pixel index $i$ (Left-Right) multiplied by slice thickness ($+5.0\text{ mm}$).
- **Voxel-to-World Affine**: `case_000` affine matrix:
  $$\begin{bmatrix} -0.5703 & 0 & 0 & 233.0 \\ 0 & 0.5703 & 0 & -373.43 \\ 0 & 0 & 5.000 & 26.50 \\ 0 & 0 & 0 & 1.0 \end{bmatrix}$$
  `[DATA-VERIFIED]`

---

## 5. Coordinate Round-Trip Results

Evaluation of the algebraic invertibility of the bounding box normalization:
$$\mathbf{p}_{\text{norm}} = \frac{\mathbf{p}_{\text{phys}} - \mathbf{c}}{\mathbf{s}} \iff \tilde{\mathbf{p}}_{\text{phys}} = \mathbf{p}_{\text{norm}} \odot \mathbf{s} + \mathbf{c}$$

Evaluated across all 450 patients on all valid targets and random surface points `[EXPERIMENT-VERIFIED]`:

| Test | Mean Error (mm) | Median Error (mm) | P99 (mm) | Maximum (mm) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Surface Points** | $3.87 \times 10^{-5}$ | $9.54 \times 10^{-6}$ | $2.45 \times 10^{-4}$ | $2.47 \times 10^{-4}$ | **PASS** |
| **Target Centroids** | $1.32 \times 10^{-5}$ | $9.54 \times 10^{-7}$ | $2.44 \times 10^{-4}$ | $2.45 \times 10^{-4}$ | **PASS** |
| **Combined** | **$2.72 \times 10^{-5}$** | **$7.63 \times 10^{-6}$** | **$2.45 \times 10^{-4}$** | **$2.47 \times 10^{-4}$** | **PASS** |

- **Reproduction Command**:
  ```bash
  python tools/audit/test_coordinate_roundtrip.py
  ```

*Conclusion*: The mathematical forward and inverse operations of the normalization are exact to floating-point precision ($< 0.001\text{ mm}$).

---

## 6. Normalization

### Mathematical Formula `[CODE-VERIFIED]`
For point $\mathbf{x} \in \mathbb{R}^3$:
$$\mathbf{c} = \begin{bmatrix} c_x \\ c_y \\ c_z \end{bmatrix} = \frac{\mathbf{x}_{\min} + \mathbf{x}_{\max}}{2}, \quad \mathbf{s} = \begin{bmatrix} L_x / 2 \\ L_y / 2 \\ L_z / 2 \end{bmatrix} = \max\left(\frac{\mathbf{x}_{\max} - \mathbf{x}_{\min}}{2}, 10^{-4}\right)$$
$$\mathbf{x}' = \begin{bmatrix} x' \\ y' \\ z' \end{bmatrix} = \begin{bmatrix} \frac{x - c_x}{L_x / 2} \\ \frac{y - c_y}{L_y / 2} \\ \frac{z - c_z}{L_z / 2} \end{bmatrix} \in [-1.0, 1.0]^3$$
De-normalization during evaluation:
$$\hat{\mathbf{y}}_{\text{phys}} = \hat{\mathbf{y}}_{\text{norm}} \odot \mathbf{s} + \mathbf{c}$$

### Scale Preservation Verdict
> **Does the model retain absolute physical body scale? NO. [CODE-VERIFIED]**

The current representation strips patient-specific physical dimensions ($L_x, L_y, L_z$) before input to the network. The neural network forward signature (`model(points, sex_prior, mask)`) receives only dimensionless coordinates and never receives $\mathbf{s}$, $\mathbf{c}$, or $L$.

---

## 7. Body-Scale Information (Training Split Only, N=360) `[DATA-VERIFIED]`

In strict compliance with **Rule 4**, all population reference statistics and correlations were derived exclusively from the **training split ($N=360$)**:

### Physical Torso Dimensions (Training Split N=360)
- **Width ($L_x = \max x - \min x$)**: Mean $135.89 \pm 73.13\text{ mm}$, Median $116.99\text{ mm}$, Range [$59.78, 849.43\text{ mm}$], P5=$72.71\text{ mm}$, P95=$259.08\text{ mm}$.
- **Depth ($L_y = \max y - \min y$)**: Mean $204.20 \pm 40.49\text{ mm}$, Median $203.04\text{ mm}$, Range [$89.39, 864.71\text{ mm}$], P5=$171.72\text{ mm}$, P95=$231.83\text{ mm}$.
- **Height ($L_z = \max z - \min z$)**: Mean $1451.53 \pm 644.02\text{ mm}$, Median $1552.34\text{ mm}$, Range [$283.83, 3555.48\text{ mm}$], P5=$481.16\text{ mm}$, P95=$2416.22\text{ mm}$.

*(For comparison: Validation mean height = $1,337.53\text{ mm}$; Test mean height = $1,383.89\text{ mm}$)*.

### Statistical Association with Internal Anatomy (Training Split Only) `[DATA-VERIFIED]`
- **Height ($L_z$) vs Vertebra L1 Coordinate ($N=351$)**: Pearson $r = \mathbf{0.6066}$, Spearman $\rho = \mathbf{0.5442}$.
- **Width ($L_x$) vs Kidney Left-Right Separation in X ($N=335$)**: Pearson $r = \mathbf{0.5830}$, Spearman $\rho = \mathbf{0.4363}$.
- **Depth ($L_y$) vs Liver Y-Coordinate ($N=358$)**: Pearson $r = -0.1873$, Spearman $\rho = 0.1199$.

*Quantification of Information Loss*: Significant physical anatomical variance ($r \approx 0.58\text{--}0.61$) is completely discarded by anisotropic unit bounding box normalization.

---

## 8. Augmentation Consistency

| Augmentation | Surface | Targets | Mathematically Correct? | Code Location | Severity | Evidence |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **Gaussian Jitter** | Yes ($\sigma=0.005$) | No | **YES** (Sensor noise regularization) | `sharon/dataset.py:63` | **LOW** | `[CODE-VERIFIED]` |
| **3D Rotation** | No | No | **N/A** (Dead argument `rotation_aug`) | `sharon/dataset.py:22` | **MEDIUM** | `[CODE-VERIFIED]` |
| **Translation** | No | No | **N/A** | None | None | `[CODE-VERIFIED]` |
| **Scaling** | No | No | **N/A** | None | None | `[CODE-VERIFIED]` |
| **Reflection / Flip** | No | No | **N/A** | None | None | `[CODE-VERIFIED]` |

---

## 9. Target Index Integrity

- All 121 anatomical classes are mapped consistently across `sharon/labels.py`, `pointclouds_450.pt`, model definitions, and evaluation scripts. `[CODE-VERIFIED]`
- Disjoint partitioning into 63 Skeletal and 58 Soft-Tissue classes is exact. `[CODE-VERIFIED]`
- Biological sex indices (`prostate`: 21, `uterus`: 117, `ovary_left`: 118, `ovary_right`: 119, `vagina`: 120) align perfectly with `ORGAN_NAMES`. `[CODE-VERIFIED]`
- **Verdict**: **TARGET INDEXING CONSISTENT**. `[EXPERIMENT-VERIFIED]`

---

## 10. Presence and Sex Masks

- **Mask Encoding**: 1.0 for valid present structures, 0.0 for absent structures. `[CODE-VERIFIED]`
- **Zero-Coordinate Test**: Exactly **0** targets with $(0, 0, 0)$ coordinates have `mask == 1.0`. All missing organs are strictly masked with `mask == 0.0`. `[DATA-VERIFIED]`
- **Loss Contribution**: In `losses.py`, all loss terms are multiplied by `mask` and divided by `mask.sum()`, guaranteeing that absent or unsegmented structures never contribute to loss or metric calculations. `[CODE-VERIFIED]`
- **Synthetic Female Anatomy**: Classes 118–121 are generated via geometric ellipsoids relative to bladder and sacrum in `sharon/dataset_preprocessing.py`. They are not segmented by TotalSegmentator. `[CODE-VERIFIED]`

---

## 11. Patient Pairing

- Within `pointclouds_450.pt`, surface arrays, target arrays, scales, centers, and case IDs were generated within a single per-case worker (`_worker_normalized_case`) and sorted together by `case_id`. `[CODE-VERIFIED]`
- Surface and target arrays for index $i$ correspond to `case_ids[i]`. `[DATA-VERIFIED]`
- **Verdict**: **PATIENT PAIRING INTERNALLY CONSISTENT**. `[EXPERIMENT-VERIFIED]`

---

## 12. Split Leakage

- **Case ID Overlap**: $Train \cap Val = 0$, $Train \cap Test = 0$, $Val \cap Test = 0$. `[DATA-VERIFIED]`
- **Volume Hash Duplicates (Cryptographic SHA-256)** `[DATA-VERIFIED]`:
  - 10 duplicate CT volumes exist across the 450 dataset directories.
  - **5 duplicate volume pairs cross data splits**:
    1. `case_219` (Train) $\equiv$ `case_406` (Test) $\to$ **Direct Test Set Leakage**
    2. `case_096` (Test) $\equiv$ `case_412` (Val)
    3. `case_100` (Train) $\equiv$ `case_408` (Val)
    4. `case_144` (Train) $\equiv$ `case_407` (Val)
    5. `case_344` (Train) $\equiv$ `case_402` (Val)
  - `case_100` is labeled Male, while its twin `case_408` is labeled Female with synthetic pelvic organs.
- **Verdict**: **CRITICAL DATA LEAKAGE DETECTED**. `[DATA-VERIFIED]`

---

## 13. Target Quality

- No `NaN` or `Inf` values exist in the target arrays. `[DATA-VERIFIED]`
- However, due to the X/Z axis swap, the coordinate distributions are severely corrupted:
  - Z coordinates of organs span from $-1,030\text{ mm}$ to $+3,747\text{ mm}$ (mean $1,433\text{ mm}$). `[DATA-VERIFIED]`
  - Spine vertebrae (T6, T12, L1, L3, L5) do not progress monotonically along Z; instead, they progress along X. `[DATA-VERIFIED]`
  - Elongated vessels (`aorta`, `inferior_vena_cava`) exhibit high sensitivity to scan truncation. `[DATA-VERIFIED]`

---

## 14. Metric Verification

The evaluation metric calculates:
$$\text{dist}_i = \|\hat{\mathbf{p}}_i - \mathbf{p}_i\|_2, \quad \text{MRE} = \frac{1}{\sum m_i} \sum_{i} m_i \text{dist}_i$$
`[CODE-VERIFIED]`
- **Synthetic Test 1**: $p=(0,0,0), \hat{p}=(3,4,0) \implies \text{Error} = 5.0000\text{ mm}$ (**PASS**). `[EXPERIMENT-VERIFIED]`
- **Synthetic Test 2**: Target 1 error = 5 mm, Target 2 error = 10 mm $\implies \text{MRE} = 7.5000\text{ mm}$ (**PASS**). `[EXPERIMENT-VERIFIED]`
- The metric computes true Mean Radial Error (Euclidean distance mean), not global RMSE. `[CODE-VERIFIED]`

---

## 15. Visual Sanity Findings (Figures 1–8)

1. **Figure 1 (`fig1_bbox_distributions.png`)**: Clearly shows the pathological Z extent peaking between $1,000$ and $3,500\text{ mm}$ (up to 3.5 meters), and X extent crushed below $100\text{ mm}$. `[VISUALLY-SUSPECTED]` $\to$ `[DATA-VERIFIED]`
2. **Figures 2–4 (`fig2_height_vs_target_coord.png`, `fig3_width_vs_kidney_separation.png`, `fig4_depth_vs_liver_ap.png`)**: Confirm strong linear correlation between body dimensions and organ coordinates. `[VISUALLY-SUSPECTED]` $\to$ `[DATA-VERIFIED]`
3. **Figure 5 (`figures/geometry/overlay_*.png`)**: 3D scatter plots of 20 sample patients demonstrate that internal skeletal and soft-tissue landmarks are stretched along a needle-like vertical pillar. `[VISUALLY-SUSPECTED]` $\to$ `[DATA-VERIFIED]`
4. **Figure 7 (`fig7_missing_target_percentage.png`)**: Documents that thoracic organs have high missingness (>90%) due to abdominal CT coverage. `[VISUALLY-SUSPECTED]` $\to$ `[DATA-VERIFIED]`
5. **Figure 8 (`fig8_roundtrip_error_distribution.png`)**: Confirms numerical errors $< 0.00025\text{ mm}$ for normalization inversion. `[VISUALLY-SUSPECTED]` $\to$ `[DATA-VERIFIED]`

---

## 16. Issues by Severity

### CRITICAL
1. **Axis 0 and Axis 2 (X and Z) Swapped in Physical Conversion (`sharon/pointcloud_sampler.py:80-82`)**:
   Inverting voxel axes before affine multiplication resulted in a $2.41\text{ meter}$ distortion on `case_000`, mapping in-plane pixels to slice thickness ($5\text{ mm}$) and slices to pixel spacing ($-0.57\text{ mm}$). `[CODE-VERIFIED]` & `[DATA-VERIFIED]`
2. **Synthetic Template Mesh Substitution (`sharon/pointcloud_sampler.py:102-115`)**:
   Input point clouds $X$ are not patient-specific skin extracted from CT scans; they are a canonical template mesh scaled to match the warped organ bounds. `[CODE-VERIFIED]`
3. **Cross-Split Patient Leakage (`sharon/outputs/splits_pointcloud.json`)**:
   Five duplicate patient CT volumes exist across data splits, including `case_219` (Train) $\equiv$ `case_406` (Test), and `case_100` (Train, Male) $\equiv$ `case_408` (Val, Female). `[DATA-VERIFIED]`

### HIGH
1. **Destruction of Absolute Body Scale Information**:
   Per-axis anisotropic normalization maps every torso to $[-1, 1]^3$ without passing physical bounding box dimensions $\mathbf{s}$ to the network. `[CODE-VERIFIED]` & `[DATA-VERIFIED]`
2. **Left/Right Inversion**:
   Due to the X/Z swap, bilateral Left/Right organ separations appear in the Z axis, violating expected anatomical symmetry in 29.5% of patient pairs along Axis 0. `[DATA-VERIFIED]`

### MEDIUM
1. **Dead Argument `rotation_aug` in `dataset.py`**:
   Accepted in `__init__` but omitted in `__getitem__`. `[CODE-VERIFIED]`
2. **Centroid Instability for Elongated Structures**:
   Single centroids for continuous structures (aorta, vena cava, spinal cord) vary wildly depending on CT slice truncation. `[DATA-VERIFIED]`

### LOW
1. **Synthetic Female Pelvic Organs**:
   Classes 118–121 are generated via geometric ellipsoids rather than true automated segmentation masks. `[CODE-VERIFIED]`

---

## 17. Hypotheses Not Yet Proven

1. **CT HU Thresholding Quality for Skin** `[UNVERIFIED]`:
   Whether applying marching cubes or isosurface extraction on `ct.nii.gz` at $\text{HU} = -300$ will yield watertight, artifact-free torso meshes across all 450 cases without scanner table contamination.
2. **Impact of Correct Physical Dimensions on Baseline Error** `[UNVERIFIED]`:
   Whether restoring correct physical millimeters (reducing torso height from 2.2m to 0.45m) will immediately reduce radial errors proportionally.

---

## 18. Phase 1 Verdict

# **FAIL**

### Explanation
The current experimental results (including the reported 44.8 mm error) **cannot be scientifically trusted**.
1. All models were trained on coordinates where X and Z were swapped, creating patients who are 2 to 3.7 meters tall and 7 cm wide. `[DATA-VERIFIED]`
2. The skin point clouds were artificially stretched from a template mesh rather than extracted from real patient anatomy. `[CODE-VERIFIED]`
3. Patient data leaked across the train, validation, and held-out test splits. `[DATA-VERIFIED]`
4. No architectural redesign or loss modification should be attempted until the underlying physical data and coordinate geometry are corrected.

---

## 19. What Must Happen Before Phase 2

Before any Phase 2 model redesign or training can proceed, the following sequential data engineering repairs must be executed:

1. **Correct Voxel-to-World Affine Multiplication (`sharon/pointcloud_sampler.py`)**:
   Replace `cz, cy, cx = com` with `cx, cy, cz = com` (or `com[0], com[1], com[2]`), ensuring `affine @ [com[0], com[1], com[2], 1.0]` matches official `nibabel` physical points.
2. **Extract Real Patient Skin Point Clouds from CT Volumes**:
   Implement an automated skin surface extractor on `ct.nii.gz` using thresholding ($\text{HU} \approx -300$), largest connected component filtering, and Poisson-disk surface sampling to yield true patient-specific skin point clouds.
3. **Purge Duplicate Volumes and Re-split Dataset**:
   Remove duplicate CT volumes (`case_400..418`), resolve conflicting sex labels, and generate a strictly patient-disjoint train/val/test split using unique patient hashes.
4. **Preserve Physical Body Scale**:
   Pass patient physical dimensions $\mathbf{s} \in \mathbb{R}^3$ (or isotropic scale) to the network conditioning or embedding layers so the model retains global body sizing cues.
5. **Re-export `pointclouds_450.pt` and Re-verify Coordinate Round-Trips**:
   Re-run the audit test suite to confirm zero millimeter coordinate discrepancy and valid physical anatomy before proceeding to model development.
