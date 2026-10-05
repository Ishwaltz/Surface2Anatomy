# SURFACE2ANATOMY — IPMI 2027 FINAL EVIDENCE REPORT

Status: STOPPED AT THE REQUIRED ONTOLOGY GATE. This is a blocking evidence report, not completed experimental evidence.

The root and sharon labels.py define 121 named slots. However, four entries (zero-based indices 117–120: uterus, ovary_left, ovary_right, vagina) lack verified genuine ground truth in the reconstructed legacy benchmark. A 121-entry dictionary does not establish a scientifically valid 121-target benchmark.

Evidence inspected locally:

- dataset_preprocessing.py and sharon/dataset_preprocessing.py, lines 125–184: the fallback creates pelvic ellipsoids relative to internal pelvic landmarks, then assigns label IDs 118–121. The optional automated segmentation branch is not evidence of validated annotations.
- reports/phase1r/target_provenance_v2.csv: all four targets are SYNTHETIC, with primary_benchmark_mask=0.
- tools/reconstruct/target_extractor.py, lines 63–69: those four label IDs are explicitly excluded from the primary mask.
- reports/phase1r/target_support_matrix.csv: reported legacy synthetic train/validation/test counts are uterus 140/18/17; left ovary 139/18/17; right ovary 138/18/17; vagina 138/18/17. These are not valid ground-truth support counts and are not reused as new-study support.
- sharon/dataset_v3/target_ontology_v3.csv is a different legacy ordering and must be reconciled by names, not slot number.
- data_external/dap_atlas/raw/label_name.csv includes uterus (21), uterocervix (20), and gonads (18); it does not define ovary_left, ovary_right, or vagina. Neither gonads nor uterocervix can silently substitute for those targets. Presence of a label name alone does not validate annotation provenance or support.
- reports/phase11/mappings/AMOS_target_mapping.csv includes a combined prostate/uterus label. This requires case-level applicability and semantic validation; it does not supply separate ovaries or vagina.

The required stop is triggered by protocol section 1. No model was trained, no final split was sealed, and no new final-test metric was computed. The remaining 117 candidates have not received the required new full data audit either; they are not certified here as ground truth.

Scope and limitations: the audit inventories local source files and statically inspects serialized pickle strings without executing pickle payloads. It is not a tensor/volume validity audit, source-volume hash audit, or exhaustive search for newly obtainable public annotations. Prior exposure, patient disjointness, fresh-test availability, coordinate integrity, and genuine per-target support remain unestablished. Inspection errors are recorded. Legacy performance files were not used to derive any new result. Source hashes are evidence fingerprints, not a FINAL_FREEZE.

To reopen the gate: provide provenance-validated annotations for the missing structures, establish exact semantic mappings and patient identities, then verify all 121 targets with valid masks. Obtain at least 10 valid final-test subjects per target (preferably 20), with additional training/validation support. If changing the target definitions is intended, approve a scientifically justified ontology revision explicitly; this audit does not substitute targets to reach 121. Run the complete exposure and geometry audits before training. Keep final test sealed until validation, independent reviews, repairs, model selection and calibration are complete.

No venue verdict is defensible yet. REDIRECT_VENUE would also be premature: the current failure is a data-evidence gate, not a measured assessment of method quality.

## Requested status fields

```
STUDY_VERSION=Surface2Anatomy-IPMI-121-v1
TARGET_COUNT=121_DECLARED_CANDIDATES
TARGET_COUNT_VERIFIED=NO
TRAIN_N=NOT_ESTABLISHED
VAL_N=NOT_ESTABLISHED
FINAL_TEST_N=NOT_ESTABLISHED
FRESH_FINAL_TEST=UNKNOWN_NOT_AUDITED
FINAL_MODEL=NOT_ESTABLISHED
FINAL_121_MACRO_MRE_MM=NOT_ESTABLISHED
FINAL_121_MACRO_MRE_95CI=NOT_ESTABLISHED
THIS_IS_THE_ONLY_HEADLINE_ERROR=NOT_APPLICABLE_NO_RESULT
STRONGEST_NEURAL_BASELINE=NOT_ESTABLISHED
STRONGEST_NEURAL_BASELINE_MRE=NOT_ESTABLISHED
ABSOLUTE_IMPROVEMENT_MM=NOT_ESTABLISHED
RELATIVE_IMPROVEMENT_PERCENT=NOT_ESTABLISHED
P_VALUE=NOT_ESTABLISHED
EFFECT_SIZE=NOT_ESTABLISHED
FSEC_FRAME_STABILITY_GAIN=NOT_ESTABLISHED
CCTQ_PARTIAL_VIEW_GAIN=NOT_ESTABLISHED
ASOM_COMPLETED=NO
ASOM_DIMENSIONS=NOT_COMPUTED_REQUIRED_121x12
ASOM_CROSS_SEED_STABILITY=NOT_ESTABLISHED
ASOM_CROSS_COHORT_STABILITY=NOT_ESTABLISHED
ELAA_COMPLETED=NO
ELAA_ERROR_CORRELATION_RHO=NOT_ESTABLISHED
ELAA_ERROR_CORRELATION_P=NOT_ESTABLISHED
FEMALE_TARGETS=uterus,ovary_left,ovary_right,vagina (candidate entries; legacy labels synthetic)
UTERUS_MRE=NOT_ESTABLISHED
OVARY_LEFT_MRE=NOT_ESTABLISHED
OVARY_RIGHT_MRE=NOT_ESTABLISHED
FEMALE_TARGET_SUPPORT_OK=NO_VERIFIED_SUPPORT
BRAIN_MRE=NOT_ESTABLISHED
SKULL_MRE=NOT_ESTABLISHED
BRAIN_FOV_RELATIONSHIP=NOT_ESTABLISHED
EXTERNAL_BRAIN_N=NOT_ESTABLISHED
BRAIN_FOV_REPAIR_GAIN=NOT_ESTABLISHED
FLARE_ZERO_SHOT=NOT_ESTABLISHED
AMOS_ZERO_SHOT=NOT_ESTABLISHED
CTORG_ZERO_SHOT=NOT_ESTABLISHED
REAL_SENSOR_N_SUBJECTS=NOT_ESTABLISHED
REAL_SENSOR_N_FRAMES=NOT_ESTABLISHED
REAL_SENSOR_TEMPORAL_STABILITY=NOT_ESTABLISHED
CONFORMAL_90_COVERAGE=NOT_ESTABLISHED
CONFORMAL_90_MEAN_RADIUS_MM=NOT_ESTABLISHED
PATIENT_SHUFFLE_MRE=NOT_ESTABLISHED
QUERY_PERMUTATION_MRE=NOT_ESTABLISHED
GLOBAL_TOKEN_MRE=NOT_ESTABLISHED
NO_CROSS_ATTENTION_MRE=NOT_ESTABLISHED
DATA_AUDIT=NOT_PASSED_ONTOLOGY_GATE_BLOCKED
LEAKAGE_AUDIT=NOT_COMPLETED
STATISTICS_REVIEW=NOT_PERFORMED
ANATOMY_REVIEW=NOT_PERFORMED
HOSTILE_REVIEW=NOT_PERFORMED
POST_TEST_TUNING=NO_FINAL_TEST_NOT_RUN
IPMI_DECISION=DEFERRED_INSUFFICIENT_EVIDENCE
```

TOP_5_DEFENSIBLE_NOVELTY_CLAIMS: None established. FSEC, CCTQ, ASOM, ELAA, and genuine 121-target breadth remain hypotheses to test.

TOP_5_REMAINING_WEAKNESSES:
1. Four legacy female targets are synthetic, not validated patient anatomy.
2. All-121 valid support and new split membership are unestablished.
3. Freshness and full prior-exposure accounting remain unaudited.
4. New geometry, provenance, semantic equivalence, and leakage audits are incomplete.
5. No new matched experiments, uncertainty calibration, independent reviews, or final results exist.
