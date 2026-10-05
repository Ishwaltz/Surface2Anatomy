# PHASE 8A FINAL CORRECTION GATE & AUDIT ADDENDUM

**Date:** 2026-09-06  
**Status:** `CORRECTION GATE COMPLETE — PHASE 8B AUTHORIZED`  
**Auditor:** Senior Medical-Imaging Data Engineer, Computational Anatomy Researcher, Scientific Dataset Auditor  
**Artifact:** `reports/phase8a/PHASE_8A_CORRECTION_ADDENDUM.md`  

---

## 1. Reconciled Cohort Counts & Mathematical Discrepancy Resolution

### 1.1 Programmatic Explanation of the Initial Discrepancy
In the initial preliminary summary, candidate counts were reported as 2,202 scans and 2,151 subject groups instead of 2,201 and 2,150. 

A programmatic audit of the source discovery mechanisms revealed the root cause:
- In the local `dataset/` directory, **441 folders** matching `case_*` existed.
- Case `case_413` is an exact duplicate of `case_411` from the raw AMOS 2022 cohort. During the Phase 1R dataset reconstruction and audit, `case_413` was formally identified and permanently excluded from Dataset V2, leaving exactly **440 canonical cases** in the frozen [`sharon/dataset_v2/manifest_v2.csv`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/sharon/dataset_v2/manifest_v2.csv).
- The initial V2 adapter discovery logic utilized a raw directory glob `sorted(self.root_dir.glob("case_*"))`, which inadvertently picked up the excluded physical folder `case_413`.
- We updated [`v2_adapter.py`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/sharon/dataset_v3/adapters/v2_adapter.py) to bind strictly to the authoritative `manifest_v2.csv` (filtering out `case_413`).

### 1.2 Exact Counts Derived Directly from Preprocessing Manifest
Counts generated programmatically and directly from [`sharon/dataset_v3/manifest_preprocessing_v3.csv`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/sharon/dataset_v3/manifest_preprocessing_v3.csv):

| Source Dataset | Source Name | Candidate Scans | Unique Subject Groups | Repeat Longitudinal Scans |
| :--- | :--- | :---: | :---: | :---: |
| **Source A** | Dataset V2 (AMOS CT) | 440 | 440 | 0 |
| **Source B** | TotalSegmentator CT v2 | 1,228 | 1,228 | 0 |
| **Source C** | DAP Atlas (AutoPET) | 533 | 482 | 51 |
| **TOTALS** | **Combined Harmonized Manifest** | **2,201** | **2,150** | **51** |

### 1.3 Mathematical Reconciliation Proof
$$\text{Total Candidate Scans} = 440 + 1,228 + 533 = \mathbf{2,201}$$
$$\text{Unique Subject Groups} = 440 + 1,228 + 482 = \mathbf{2,150}$$
All counts reconcile mathematically with zero discrepancies. No manual entries exist.

---

## 2. Multi-Layer Provenance and Granular License Record

Rather than applying a blanket license across third-party assets, the distinct licenses, versions, DOIs, and access terms are separated below:

### 2.1 TotalSegmentator CT
- **Exact Release / Version:** TotalSegmentator CT Dataset **v2.0.1** (1,228 clinical CT scans, 117 anatomical classes).
- **Exact Persistent DOI:** [`10.5281/zenodo.10047292`](https://doi.org/10.5281/zenodo.10047292) (Zenodo Record 10047292).
- **Primary Publication:** J. Wasserthal et al., *"TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images"*, *Radiology: Artificial Intelligence*, 5(5):e230024, 2023. DOI: [10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024).
- **Dataset License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**). Allows academic and commercial reuse and redistribution with attribution.
- **Software / Model Code License:** **Apache License 2.0** ([GitHub Repository](https://github.com/wasserth/TotalSegmentator/blob/master/LICENSE)).

### 2.2 DAP Atlas (Dense Anatomical Prediction)
- **Atlas Segmentation Masks:**
  - **Asset:** DAP Atlas Annotation Masks v1 (533 whole-body CT multi-organ masks covering 142 structures).
  - **Distribution:** Google Drive (`1ex0a9eQULLvKPDwijmijX2h49A-ockNy`) and Synapse (`syn52287632.1`).
  - **License / Terms:** Open research dataset release by Karlsruhe Institute of Technology (KIT) with attribution to Jaus et al. (arXiv:2307.13375).
- **Atlas Processing Code:**
  - **Asset:** `alexanderjaus/AtlasDataset` repository.
  - **License:** **Apache License 2.0** ([GitHub License](https://github.com/alexanderjaus/AtlasDataset/blob/main/LICENSE)).
- **Underlying CT Imaging Data:**
  - **Collection:** The Cancer Imaging Archive (TCIA) **"FDG-PET-CT-Lesions" Version 2**.
  - **Collection Persistent DOI:** [`10.7937/gkr0-xv29`](https://doi.org/10.7937/gkr0-xv29) (Legacy v1 DOI: `10.7937/98TC-K464`).
  - **Primary Publication:** S. Gatidis et al., *"A whole-body FDG-PET/CT Dataset with manually annotated Tumor Lesions"*, *Scientific Data*, 9:601, 2022. DOI: [10.1038/s41597-022-01718-3](https://doi.org/10.1038/s41597-022-01718-3).
  - **CT Access & License Terms:** Creative Commons Attribution 4.0 International (**CC BY 4.0**), governed by TCIA Data Usage Policies and Restrictions (open research access requiring mandatory citation of Gatidis et al. and TCIA).

---

## 3. Cohort Coverage Status: Downgraded to Preliminary

All 2,201 candidate scans in [`manifest_preprocessing_v3.csv`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/sharon/dataset_v3/manifest_preprocessing_v3.csv) are formally downgraded to:
- `coverage_status`: `PRELIMINARY_CANDIDATE`
- `cohort_assignment`: `PENDING_PHASE8B_QC`
- `include_v3_primary`: `CANDIDATE`
- `include_v3_extended`: `CANDIDATE`

### Critical Protocol Clarification
We do **not** claim that all candidate scans provide full-torso or whole-body coverage. Final mutually exclusive assignment into:
1. **PRIMARY COHORT:** Scans whose external body geometry satisfies the strict, deployment-compatible external torso envelope windowing (neck notch to groin bifurcation) with full landmark visibility.
2. **EXTENDED COHORT:** Scans with partial field of view (e.g. abdomen-only or thorax-only) that fail the full torso windowing test but contain $\ge 20$ valid target centroids.
3. **EXCLUDED COHORT:** Scans with severe external clipping, field-of-view truncation, or anatomical corruptions.

**This cohort assignment will occur strictly and exclusively during Phase 8B common external surface coverage QC.**

---

## 4. Physical 3D Validation of DAP Rib Inversion Mapping

### 4.1 Anatomical Rationale & Hypothesized Inversion
In anatomical standard coordinate systems (where $+Z$ is Superior / cranial):
- Standard anatomical Rib 1 is cranial (adjacent to first thoracic vertebra and clavicle).
- Standard anatomical Rib 12 is caudal (inferior-most floating rib).
- DAP's internal post-processing code sorts ribs ascending by Z-coordinate. Therefore, DAP `costa 1` is expected to be caudal (Rib 12) and DAP `costa 12` cranial (Rib 1).
- Canonical mapping rule:
  $$\text{costa } k \longleftrightarrow \text{rib } (13 - k)$$

### 4.2 Quantitative Verification Across 35 Representative DAP Scans
We executed [`tools/dataset_v3/validate_dap_rib_mapping.py`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/tools/dataset_v3/validate_dap_rib_mapping.py) across 35 representative scans evenly sampled across the DAP cohort:
- **Total Adjacent Rib Pairs Tested:** **770** pairs across both Left and Right hemithoraces.
- **Pairs with $Z(\text{costa}_{k+1}) > Z(\text{costa}_k)$:** **770**
- **Monotonicity Agreement Rate:** **100.00%**
- **Physical Inversion / Violation Count:** **0**
- **Physical Violation Rate:** **0.00%**

All 35 scans demonstrate strict, continuous cranial ascent from `costa 1` to `costa 12`. Applying $\text{rib } r = 13 - k$ converts this into a strictly descending cranial-to-caudal sequence matching V2 and TotalSegmentator.

Full numerical records are archived in [`reports/phase8a/dap_rib_physical_validation.csv`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/dap_rib_physical_validation.csv).

### 4.3 Visual QC Artifacts
Visual verification plots depicting physical world $Z$ coordinates (in mm) versus canonical rib indices for 10 representative cases were generated and inspected:
- Summary 10-case panel: [`reports/phase8a/qc_ribs/summary_rib_qc.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/summary_rib_qc.png)
- Individual case artifacts:
  - [`case_01_AutoPET_0011f3de.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/case_01_AutoPET_0011f3de.png)
  - [`case_02_AutoPET_098c4b7b.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/case_02_AutoPET_098c4b7b.png)
  - [`case_03_AutoPET_13d0984c.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/case_03_AutoPET_13d0984c.png)
  - [`case_04_AutoPET_19838cb8.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/case_04_AutoPET_19838cb8.png)
  - [`case_05_AutoPET_21e4ffcb.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/case_05_AutoPET_21e4ffcb.png)
  - [`case_06_AutoPET_27ad42f8.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/case_06_AutoPET_27ad42f8.png)
  - [`case_07_AutoPET_345b1177.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/case_07_AutoPET_345b1177.png)
  - [`case_08_AutoPET_3b2a4af4.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/case_08_AutoPET_3b2a4af4.png)
  - [`case_09_AutoPET_41fadf65.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/case_09_AutoPET_41fadf65.png)
  - [`case_10_AutoPET_48c70dc4.png`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/qc_ribs/case_10_AutoPET_48c70dc4.png)

**Conclusion:** The mapping $\text{costa } k \leftrightarrow \text{rib } (13 - k)$ is physically verified, proven correct, and is now **officially frozen**.

---

## 5. Deterministic Near-Duplicate and Exact Duplicate Audit

We executed [`tools/dataset_v3/near_duplicate_audit.py`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/tools/dataset_v3/near_duplicate_audit.py) to distinguish byte-exact duplicates from near-duplicate candidate scans robust to NIfTI recompression or header modifications:

1. **Exact Duplicate Search:**
   - Cryptographic SHA-256 byte hashes of raw volumes and canonical voxel array buffers were cross-compared across all sources.
   - **Cross-Source Exact Duplicates:** **0**
2. **Near-Duplicate Candidate Search:**
   - Scans were fingerprinted using invariant geometry tuples: matrix dimensions $(N_x, N_y, N_z)$, voxel spacings $(s_x, s_y, s_z)$ rounded to 2 decimal places, and physical bounding FOV extents.
   - Cross-source geometry collisions: **0**
   - **Cross-Source Near-Duplicate Candidates:** **0**

Full audit output is archived in [`reports/phase8a/cross_dataset_near_duplicates.csv`](file:///home/sharon/Desktop/3D-Organ-Location-Prediction-Model/reports/phase8a/cross_dataset_near_duplicates.csv).

---

## 6. Source-Domain Classifier Status

The source classifier evaluation performed in `tests/test_dataset_v3_surface_definition.py` (`test_source_classifier_after_coverage_standardization`) is hereby explicitly designated as an **implementation-level unit test**.

- It validates that external geometry standardization eliminates superficial bounding box cues on synthetic/representative mock profiles.
- **Empirical Scientific Invariant:** The final source-domain classifier accuracy across real patient scans will **NOT** be claimed or reported until all 2,201 V3 external surfaces have been fully extracted and standardized under common coverage in Phase 8B.

---

## 7. Mandatory Summary Status Block

```text
Manifest scan count:
2201

Unique subject groups:
2150

Counts reconcile mathematically:
YES

TotalSegmentator provenance verified:
YES

DAP mask provenance verified:
YES

DAP CT provenance verified:
YES

Coverage classification final:
NO — DEFERRED TO 8B

DAP rib mapping physically validated:
YES

Cross-source exact duplicates:
0

Cross-source near-duplicate candidates:
0

PHASE 8B AUTHORIZED:
YES
```
