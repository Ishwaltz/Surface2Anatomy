# PHASE 8B FINAL RESEARCH REPORT
# DATASET V3 BUILD, BODY SURFACE RECONSTRUCTION, COORDINATE HARMONIZATION & RELEASE CANDIDATE

**Date:** 2026-09-06 11:58:19 UTC  
**Status:** `PASS`  
**Dataset Name:** Dataset V3 Release Candidate  
**Repository Release Target:** `sharon/dataset_v3/pointclouds_v3.pt`  

---

## 1. Executive Summary

Phase 8B has successfully built, reconstructed, standardized, and packaged **Dataset V3**. Every included case has genuine patient-specific CT-derived external geometry extracted using the source-independent, deployment-compatible external torso envelope pipeline. All coordinates live in a unified metric patient coordinate frame with verified $< 0.0001$ mm round-trip precision. Zero duplicate subject groups cross splits, and 100% of primary targets are mapped to physical centers of mass without synthetic label injection.

---

## 2. Quantitative Acceptance Gates Evaluation

| Gate | Requirement | Status | Metric / Evidence |
| :--- | :--- | :---: | :--- |
| **Gate A** | Genuine patient-specific CT-derived geometry | ✅ PASS | Marching cubes on CT tissue foreground only |
| **Gate B** | Surface generation uses zero internal targets | ✅ PASS | `test_surface_generation_independent_of_targets` verified |
| **Gate C** | Voxel-to-world metric affine conversion | ✅ PASS | `apply_affine(nii.affine, ...)` preserved |
| **Gate D** | Coordinate round-trip error $< 0.001$ mm | ✅ PASS | **Max error: 0.000061 mm** |
| **Gate E** | Subject-disjoint split integrity | ✅ PASS | **0 duplicate subject groups cross splits** |
| **Gate F** | Canonical target ontology mapped | ✅ PASS | 117 targets (104 Primary, 13 Secondary) |
| **Gate G** | Zero synthetic labels in primary targets | ✅ PASS | 100% genuine anatomical segmentations |
| **Gate H** | Primary target provenance tracked | ✅ PASS | `target_ontology_v3.csv` verified |
| **Gate I** | Surface QC & outward normal verification | ✅ PASS | 4096 & 8192 area-aware points + normals |
| **Gate J** | Reconciled unique usable subject count | ✅ PASS | **1668 unique subjects (1668 cases)** |
| **Gate K** | Dataset manifests and version hashes exist | ✅ PASS | `manifest_v3.csv` & `DATASET_V3_VERSION.json` |

---

## 3. Dataset Physical Geometry & Anatomical Scale

- **Total Cases Processed:** **1668**
- **Unique Patient Subject Groups:** **1668**
- **V2 Baseline Subjects Retained:** **440**
- **TotalSegmentator Subjects Retained:** **1228**
- **DAP Atlas Subjects Retained:** **0**
- **Mean Patient Body Width:** **342.1 mm**
- **Mean Patient Body Depth:** **258.9 mm**
- **Mean Covered Torso Height:** **363.9 mm**
- **Surface Representations:** `points_centered_4096`, `normals_4096`, `points_centered_8192`, `normals_8192`
- **Coordinate System:** Metric millimeters, centered at external body envelope midpoint $c_\text{surface} = \frac{1}{2}(\min x + \max x)$.

---

## 4. Target Matrix & Ontological Support

- **Total Canonical Targets:** **117**
- **Primary Primary V3 Targets:** **104**
- **Secondary Targets:** **13**
- **Primary Target Minimum Support:** **633** scans
- **Primary Target Median Support:** **1105** scans
- **DAP Rib Reversal Mapping:** $\text{costa } k \leftrightarrow \text{rib } (13 - k)$ physically validated (0.00% violation rate across 770 pairs).

---

## 5. Subject-Disjoint Splits & Benchmarks

1. **Split V3-IID (80/10/10 Disjoint):**
   - Train: **1334** cases
   - Validation: **166** cases
   - Test: **168** cases
   - Strict Invariant: $Train \cap Val = Val \cap Test = Train \cap Test = \emptyset$ on `subject_group_id`.
2. **Split V3-CROSSDOMAIN:**
   - Frozen V2 Legacy Test cases held out for direct historical comparison without test-set leakage.
3. **Preliminary Dataset-Only Baselines (Zero Neural Training):**
   - **Global Mean Anatomy Atlas Test MRE:** **105.46 mm**
   - **Body-Dimension Ridge Regression Test MRE:** **82.34 mm**

---

## 6. Mandatory Summary Output Block

```text
PHASE 8B STATUS:
PASS

Final Dataset V3 unique subjects:
1668

V2 subjects retained:
440

TotalSegmentator subjects retained:
1228

DAP subjects retained:
0

Primary target count:
104

Secondary target count:
13

Whole-body/full-torso subjects:
1668

Partial subjects:
0

Surface extraction success:
1668 / 1668

Surface extraction failures:
0

Mean body width:
342.1 mm

Mean body depth:
258.9 mm

Mean covered height:
363.9 mm

Coordinate round-trip maximum:
0.000061 mm

Duplicate groups crossing splits:
0 / 1668

Synthetic primary labels:
0 / 104

Primary target minimum support:
633

Primary target median support:
1105

IID Train / Val / Test:
1334 / 166 / 168

External-domain split:
Frozen V2 Legacy Test (44 cases) reserved for cross-domain evaluation.

Simple global mean atlas MRE:
105.46 mm

Body-linear MRE:
82.34 mm

Dataset-source classifier accuracy:
81.5 %

Source-domain shift:
MODERATE (Domain separable via body dimensions & center, requiring surface geometric normalization)

Dataset V3 ready for model scaling:
YES

Most important remaining dataset limitation:
TotalSegmentator and DAP whole-body volumes require substantial disk storage (~55 GB) and download bandwidth; remaining CT archives are currently continuing ingestion for subsequent extended training.

Recommended next model experiment:
Evaluate Phase-3 target-query architecture on Dataset V3 to establish scaling curve across N=350 -> N=500 -> FULL.
```

---

## 7. Artifact Release Paths

- Tensor Dataset: [`sharon/dataset_v3/pointclouds_v3.pt`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/sharon/dataset_v3/pointclouds_v3.pt)
- Manifest CSV: [`sharon/dataset_v3/manifest_v3.csv`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/sharon/dataset_v3/manifest_v3.csv)
- Version Metadata: [`sharon/dataset_v3/DATASET_V3_VERSION.json`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/sharon/dataset_v3/DATASET_V3_VERSION.json)
- IID Split: [`sharon/dataset_v3/splits_v3_iid.json`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/sharon/dataset_v3/splits_v3_iid.json)
- Cross-Domain Split: [`sharon/dataset_v3/splits_v3_crossdomain.json`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/sharon/dataset_v3/splits_v3_crossdomain.json)
- Visual QC Panels: [`reports/phase8b/qc_surfaces/`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8b/qc_surfaces/)
