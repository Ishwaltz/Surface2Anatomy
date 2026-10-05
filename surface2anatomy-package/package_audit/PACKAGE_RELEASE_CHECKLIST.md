# Surface2Anatomy Package Release Checklist

## 1. Scientific & Checkpoint Integrity
- [x] Verified frozen checkpoint SHA256 hashes against Phase 10R Locked Test Protocol Freeze (`reports/phase10R/PRE_TEST_FREEZE_FINAL.md`).
- [x] Verified external-only canonical alignment features (105 morphology features) and Ridge regressor (`canonical_alignment_v3_ridge.joblib`).
- [x] Verified 117-target ontology mapping and 104 primary benchmark targets.
- [x] Zero CT/MRI voxels required at inference runtime (supervision offline only).
- [x] Generated deterministic regression reference (`REGRESSION_REFERENCE.json`).

## 2. Package Architecture & Public API
- [x] Standard package namespace: `surface2anatomy`.
- [x] Clean imports: `from surface2anatomy import SurfaceAnatomyModel, list_targets`.
- [x] Unified predictions: `model.predict()`, `model.predict_multiple()`, `model.predict_all()`, `model.predict_batch()`.
- [x] Rich typed results exposing `.centroid_mm`, `.seed_predictions_mm`, `.ensemble_disagreement_mm`.
- [x] Multiple export formats: `.to_dict()`, `.to_json()`, `.to_dataframe()`.

## 3. Data Ingestion & Input Verification
- [x] Supported geometry formats: `.PLY`, `.PCD`, `.OBJ`, `.STL`, `.XYZ`, `.TXT`, `.NPY`.
- [x] Explicit informative rejection of 2D RGB photographs (`.JPG`, `.JPEG`, `.PNG`).
- [x] Strict physical scale management (`units="mm"`, `units="cm"`, `units="m"`, conservative `units="auto"` with `AmbiguousUnitsError`).
- [x] Deterministic 4,096-point sampling with configurable `sampling_seed`.

## 4. Distribution & Lightweight Wheel Compliance
- [x] Python build backend: PEP 517/518 (`hatchling`).
- [x] Checkpoints hosted externally; wheel file size < 100 KB.
- [x] Checksum verification on remote asset downloads (SHA256).
- [x] Offline mode support with `local_files_only=True`.
- [x] Cache management via `platformdirs` (`~/.cache/surface2anatomy/`).

## 5. Testing & CLI
- [x] Full test suite across 13 dedicated test modules in `tests/`.
- [x] CLI entry point: `surface2anatomy` console command.
- [x] All 10 regression landmarks verified to 0.00 mm divergence against locked reference.
