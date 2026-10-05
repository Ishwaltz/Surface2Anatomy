# Phase 10R: Verified Literature Comparison & Primary-Source Provenance

## 1. Executive Summary of Primary-Source Corrections
In Phase 10, several external literature comparisons suffered from serious factual and methodological discrepancies. In Phase 10R, all metadata and metrics were reconstructed directly from primary publications:

1. **SAMe (Semantic Anatomy Mapping Engine, 2026):**
   - *Published Setup:* Uses a **single 2D RGB body image** input, evaluated on **35 held-out cases $\times$ 11 organs**, reporting **22.55 mm mean centroid error** on its own cohort (retiring the legacy internal 53.67 mm value).
   - *Harmonized Context:* On the corresponding 11-organ visceral subset of our validation cohort, our proposed model achieves **27.03 mm MRE**. Cross-cohort percentage claims (e.g. '49.6% improvement') are scientifically prohibited.

2. **From Surface to Viscera (Atici et al., MIDL 2026):**
   - *Published Setup:* Trained on the **German National Cohort (NAKO)**, using a body-surface point cloud **plus a mean internal-anatomy point cloud prior**, reporting mean **Chamfer distance < 5.0 mm**.
   - *Metric Distinction:* Chamfer distance measures dense surface point cloud fidelity, whereas our work evaluates 3D centroid Euclidean localization (**26.44 mm MRE** on 20 visceral organs). Incomparable metrics are strictly presented in separate columns and never ranked against each other.

3. **Depth to Anatomy (2026):**
   - *Published Setup:* Evaluates **10,020 whole-body MRI scans** across **41 structures** for scanner couch table positioning from ceiling depth images, reporting **7.69 ± 5.68 mm average symmetric surface distance (ASSD)** and **10.99 ± 5.54 mm bounding-box offset**.
   - *Different Modality & Task:* Operates on MRI scans and depth imagery for table isocenter alignment, distinct from our 3D surface point cloud localization on CT scans.

4. **LOOC (2023):**
   - *Published Setup:* Predicts 3D occupancy networks across **67 anatomical structures** (trained on **447 masks**, tested on **50 masks**), reporting **0.61 volumetric IoU**.

5. **BOSS (Computers in Biology and Medicine 2023, DOI: 10.1016/j.compbiomed.2023.107000):**
   - *Published Setup:* Trained on roughly **300 CT scans**, reporting **3.6 mm bone error**, **8.8 mm organ error**, and **8.68 mm combined skin-only error** for parametric shape model vertex reconstruction.
   - *Baseline Naming:* Our internal baseline is a simple linear surface PCA + Ridge regression on 104 centroids and is strictly designated as **'Internal Statistical Shape Model (SSM/PCA) baseline'** (55.06 mm), NOT 'BOSS'.

## 2. Primary Source Audit Table

| Study | Venue / Year | Input Modality | Cohort Size | Structures | Reported Metric | Published Value | Centroid Metric? | Direct Ranking Allowed? |
|---|---|---|---|---|---|---|---|---|
| **SAMe: A Semantic Anatomy Mappi...** | 2026 | Single 2D RGB body image | 450 | 11 visceral organs | Mean Centroid Localization Error | **22.55 mm** | YES | **NO** |
| **From Surface to Viscera: 3D Es...** | 2026 | Body-surface point cloud + mean internal-anatomy point cloud prior | NAKO population cohort | Multi-organ internal anatomy | Mean Chamfer Distance | **< 5.0 mm (Chamfer)** | NO | **NO** |
| **Depth to Anatomy: Organ Locali...** | 2026 | Single-view ceiling depth camera image (RGB-D) | 10,020 MRI scans | 41 clinical anatomical positioning structures | Average Symmetric Surface Distance (ASSD) & Bounding-Box Offset | **7.69 ± 5.68 (ASSD) / 10.99 ± 5.54 (Bbox Offset) mm** | NO | **NO** |
| **LOOC: Localizing Organs using ...** | 2023 | Single-view body surface depth images | 447 masks | 67 anatomical structures | Volumetric Intersection over Union (IoU) | **0.61 IoU** | NO | **NO** |
| **BOSS: Body Shape and Structure...** | 2023 | 3D external body surface mesh (skin only) | ~300 CT scans | Full skeleton and internal visceral organs | Dense Mesh Surface Euclidean Error | **3.6 (bone) / 8.8 (organ) / 8.68 (combined skin-only) mm** | NO | **NO** |
| **HIT: Estimating Internal Human...** | 2024 | Body surface meshes | 320 | Continuous volumetric tissue occupancy | Volumetric Dice & Surface Chamfer Distance | **0.68 / 14.2 Dice / mm** | NO | **NO** |


## 3. Matched Subset Evaluations on Our Validation Cohort

- **11-Visceral Organ Subset (SAMe Target Overlap):**
  - Targets: liver, spleen, pancreas, gallbladder, urinary bladder, aorta, trachea, kidney right, kidney left, stomach, inferior vena cava.
  - **Proposed Model Validation MRE: 27.03 mm** (Median: 21.20 mm).
  - *Literature Context:* SAMe reported 22.55 mm on 35 cases from a single RGB body image. Cohort differences and input sensor modalities preclude head-to-head ranking.

- **20-Visceral Organ Subset (Surface-to-Viscera Target Overlap):**
  - **Proposed Model Validation MRE: 26.44 mm** (Centroid MRE).
  - *Literature Context:* Atici et al. reported < 5.0 mm Chamfer distance for dense point cloud reconstruction on the NAKO cohort using an anatomical prior.
