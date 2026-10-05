# PHASE 14 CT-ORG EXTERNAL VALIDATION PRE-INFERENCE FREEZE PROTOCOL

## 1. Frozen Model Checkpoints
- **Seed 42**: `experiments/phase10R/checkpoints/C4_Proposed_seed42.pt`
  - SHA256: `760adfee1c47d80fc3f59492971b6c83459fc10ce2166f13fc1d7e4d708cb936`
- **Seed 43**: `experiments/phase10R/checkpoints/C4_Proposed_seed43.pt`
  - SHA256: `3e3c739c992b80f193ec41041e62ff59191436e2e5f1e78bd0a328c8310f7d91`
- **Seed 44**: `experiments/phase10R/checkpoints/C4_Proposed_seed44.pt`
  - SHA256: `28fbc7b30d8539f84c265b4f95f4f73b63978fb5b252f9a3ad20ea4fa1792cdd`

## 2. Frozen Canonical Alignment Model
- **Model**: Scikit-Learn Ridge Regressor (alpha = 50.0) trained on Dataset V3 training set external geometry features only.
- **Path**: `experiments/phase10R/checkpoints/canonical_alignment_v3_ridge.joblib`
  - SHA256: `b9901d72c3aa73e70b1fa77aefdbc1ebf8e682d5f5fa011ee840772827488e26`
- **Reference Script**: `tools/dataset_v3/apply_canonical_alignment.py`
  - SHA256: `4a077cf7af675b434066c26818e9499fd7fd0dfbf267d38b7264c26a6910183b`

## 3. Dataset Inclusion and Exclusion Criteria
- Eligible Cases:
  - Valid CT volume and segmentation label pair (`volume-XX.nii.gz` and `labels-XX.nii.gz`).
  - NIfTI readable with valid affine matrix.
  - Orientation unambiguous and converted to canonical RAS.
  - External body surface extractable via CT voxel intensities alone (HU > -300).
  - Surface point count >= 4096.
  - >= 1 compatible ground-truth target centroid.
  - Confirmed not duplicate of any previously used dataset (V3, TotalSegmentator, FLARE, AMOS).
  - Not marked as possible overlap.
- Excluded Categories:
  - `MISSING_CT`, `MISSING_LABEL`, `CORRUPT_CT`, `CORRUPT_LABEL`, `INVALID_AFFINE`, `ORIENTATION_AMBIGUOUS`, `INSUFFICIENT_SURFACE`, `NO_OVERLAPPING_TARGETS`, `CONFIRMED_DUPLICATE`, `POSSIBLE_OVERLAP`.
- Post-inference error-based exclusions are strictly FORBIDDEN.

## 4. Duplicate Removal Rules
- Any case matching SHA256 or canonical image fingerprint (downsampled RAS volume, clamped HU [-1000, 2000], normalized, quantized) against local cohorts (V3, TotalSegmentator, FLARE22, AMOS) is classified as `CONFIRMED_DUPLICATE` and excluded.
- Any case flagged with near-duplicate perceptual similarity score >= 0.98 is classified as `POSSIBLE_OVERLAP` and excluded from primary evaluation.

## 5. Orientation and Canonicalization Rules
- Convert volume and label to canonical RAS via `nib.as_closest_canonical()`.
- Orientation audit verifies determinant, axis permutations, and left-right consistency.
- Any unresolved orientation ambiguity results in pre-inference exclusion (`ORIENTATION_AMBIGUOUS`).

## 6. Predefined Target Mapping Rules
- `liver` (CT-ORG ID 1) -> `liver` (Slot 4) [EXACT]
- `bladder` (CT-ORG ID 2) -> `urinary_bladder` (Slot 19) [EXACT]
- `brain` (CT-ORG ID 6) -> `brain` (Slot 100) [EXACT, only when brain GT is present in scan]
- `kidneys` (CT-ORG ID 4): Bilateral split via 3D connected components in canonical RAS:
  - Two largest connected components must together account for >= 95% of kidney mask volume.
  - Component with larger physical X (patient Right in RAS) -> `kidney_right` (Slot 1).
  - Component with smaller physical X (patient Left in RAS) -> `kidney_left` (Slot 2).
  - If criteria not met, kidneys marked unavailable for that case.
- Whole `lungs` (ID 3) and `bone` (ID 5) are `INCOMPATIBLE` (no 1-to-1 mapping to our 104-target ontology).

## 7. Canonical Alignment and Point Cloud Normalization
- External surface mesh extracted via Marching Cubes at level 0.5 from CT-only binary mask.
- Deterministic area-weighted surface sampling: 4096 points with random seed 42.
- Compute external geometry features: 12 axial slice widths, depths, aspect ratios, percentile offsets.
- Offset predicted via frozen Ridge model: c_external = midpoint + pred_offset.
- Center surface: P_centered = P_world - c_external.
- Scale: P_norm = P_centered / 500.0.
- Ground truth centroids computed in world coordinates from segmentation labels, centered by subtracting the SAME c_external, and scaled by 500.0 mm.

## 8. Evaluation Metrics and Inference Protocol
- One-shot zero-shot evaluation of frozen Phase 10R ensemble:
  - y_ens = 1/3 * (y_42 + y_43 + y_44).
- Primary Metric: 3-seed ensemble Macro MRE (mm) across valid overlapping targets.
- Secondary Metrics: Micro MRE, Median, P75, P90, P95, SDR@5mm, 10mm, 15mm, 20mm, 25mm, 30mm.
- Bootstrap Confidence Interval: 5,000 patient-level resamples with fixed seed 42.
- Sampling Seed: 42.
- Bootstrap Seed: 42.
