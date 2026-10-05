# Phase 10R: Verified Literature Comparison & Primary-Source Audit (Version 2)

> [!IMPORTANT]
> **STRICT GOVERNANCE RULE ON LITERATURE COMPARISONS**
> Direct numerical performance rankings (e.g., "our model outperforms Paper X by Y%") are **strictly prohibited** unless an identical test dataset, identical input sensor modality, identical target definitions, and identical evaluation metrics are evaluated under an identical protocol.
> All external literature numbers are presented strictly as contextual benchmarks of related clinical and computer-vision methodologies.

---

## 1. Verified Literature Comparison Table

| Paper | Year & Venue | Input Modality | Cohort Size (Train / Test) | Target Count | Output Type | Reported Metric & Value | Centroid Metric? | Same Metric as Ours? | Same Dataset? | Direct Ranking Allowed? | Primary Source Citation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **SAMe** | 2026 (arXiv) | Single 2D RGB image | 450 / 35 | 11 visceral organs | 3D bounding boxes & centroids | **22.55 mm** (Mean Centroid Error) | **YES** | **YES** | **NO** | **NO** | [arXiv:2604.25646](https://arxiv.org/abs/2604.25646), Table 2 |
| **From Surface to Viscera** | 2026 (MIDL/PMLR) | Surface point cloud + internal prior | NAKO population cohort | 20 structures | Dense 3D point clouds | **< 5.0 mm** (Chamfer Distance) | **NO** | **NO** | **NO** | **NO** | [MIDL 2026](https://proceedings.mlr.press/v315/atici26a.html), Table 1 |
| **Depth to Anatomy** | 2026 (arXiv) | Ceiling depth camera (RGB-D) | 10,020 MRI scans | 41 structures | Bounding boxes & surfaces | **7.69 ± 5.68 mm** (ASSD) / **10.99 ± 5.54 mm** (Bbox Offset) | **NO** | **NO** | **NO** | **NO** | [arXiv:2601.18260](https://arxiv.org/abs/2601.18260), Table 2 |
| **LOOC** | 2023 (Conf) | Surface depth images | 447 / 50 CT masks | 67 structures | 3D occupancy fields | **0.61 IoU** (Volumetric IoU) | **NO** | **NO** | **NO** | **NO** | Official Manuscript, Table 2 |
| **BOSS** | 2023 (CMPB) | 3D body surface mesh (skin only) | ~300 CT scans | Skeleton & visceral organs | Parametric statistical shape model | **3.6 mm** (bone) / **8.8 mm** (organ) / **8.68 mm** (skin) | **NO** | **NO** | **NO** | **NO** | [DOI: 10.1016/j.compbiomed.2023.107000](https://doi.org/10.1016/j.compbiomed.2023.107000), Table 3 |
| **HIT** | 2024 (CVPR) | Body surface meshes | 320 / 64 CT scans | Tissue classes | Implicit continuous occupancy | **0.68 Dice** / **14.2 mm** (Chamfer) | **NO** | **NO** | **NO** | **NO** | CVPR 2024, Table 1 |

---

## 2. Detailed Primary-Source Synthesis & Contextualization

### 1. SAMe (arXiv:2604.25646, 2026)
- **Primary Source Finding:** SAMe introduces a semantic anatomy mapping engine for robotic ultrasound, mapping a single 2D RGB camera image of the human body to 3D internal organ positions. It evaluates on 35 held-out subjects across 11 organs, reporting **22.55 mm mean centroid localization error**.
- **Contextual Alignment:** On our Dataset V3 validation cohort, evaluating the identical 11-organ subset (liver, spleen, pancreas, gallbladder, urinary bladder, aorta, trachea, kidney right, kidney left, stomach, IVC), our proposed model achieves **27.03 mm MRE** (Median: 21.20 mm).
- **Publication Phrasing Standard:**  
  *"Our model achieved 27.03 mm MRE on the corresponding 11-target visceral subset of our frozen validation cohort. SAMe reported 22.55 mm on its independent 35-subject evaluation cohort from single RGB images. Variations in patient cohort, sensor hardware, and acquisition protocols preclude direct numerical ranking."*

### 2. From Surface to Viscera (Atici et al., MIDL / PMLR 2026)
- **Primary Source Finding:** Evaluates dense surface reconstruction of 20 internal anatomical structures on the German National Cohort (NAKO) using surface point clouds conditioned on a mean internal-anatomy template, reporting bidirectional Chamfer distance **< 5.0 mm**.
- **Contextual Alignment:** On our validation cohort across 20 visceral organs, our model achieves **26.44 mm centroid MRE**.
- **Publication Phrasing Standard:**  
  *"Our centroid localization MRE is mathematically distinct from and not directly comparable to the dense-shape Chamfer distance reported by Atici et al."*

### 3. Depth to Anatomy (arXiv:2601.18260, 2026)
- **Primary Source Finding:** Utilizes a single ceiling-mounted depth camera to estimate patient internal anatomy for scanner couch table positioning across 10,020 whole-body MRI scans. Reports **7.69 ± 5.68 mm average symmetric surface distance (ASSD)** and **10.99 ± 5.54 mm bounding-box center offset** across 41 structures.
- **Publication Phrasing Standard:**  
  *"Our 3D landmark centroid MRE cannot be directly compared to the bounding-box offset or surface distance metrics reported for automated MRI scanner positioning."*

### 4. LOOC (2023)
- **Primary Source Finding:** Employs continuous 3D occupancy networks from depth images across 67 anatomical structures (trained on 447 CT masks, tested on 50), reporting **0.61 volumetric IoU**.
- **Publication Phrasing Standard:**  
  *"Volumetric occupancy IoU evaluates spatial volumetric overlap and does not provide physical-space Euclidean point localization errors."*

### 5. BOSS (Computers in Biology and Medicine, 2023)
- **Primary Source Finding:** Parametric statistical shape model of skeleton and organs fit to skin meshes on ~300 CT scans, reporting **3.6 mm bone error**, **8.8 mm organ error**, and **8.68 mm combined surface error**.
- **Baseline Naming Rule:** In our benchmark, our internal statistical baseline is a simple linear surface PCA (64 components) + Ridge regressor fit to 104 centroids (achieving **52.56 mm validation MRE**). It is strictly designated **'Internal Statistical Shape Model (SSM/PCA) baseline'** and never referred to as 'BOSS'.

### 6. HIT (CVPR 2024)
- **Primary Source Finding:** Continuous implicit neural representation for volumetric tissue classification from surface meshes on 320 CT scans, reporting **0.68 Dice** and **14.2 mm surface Chamfer distance**.
- **Publication Phrasing Standard:**  
  *"HIT models continuous volumetric tissue classes rather than discrete anatomical landmark centroids."*

---

## 3. Governance Summary

- **Total External Studies Audited:** 6
- **Directly Comparable on Same Dataset:** 0 (None)
- **Permitted Cross-Paper Numerical Superiority Claims:** **NONE**
- **Approved Manuscript Language:** All external literature comparisons must remain purely qualitative and contextual.
