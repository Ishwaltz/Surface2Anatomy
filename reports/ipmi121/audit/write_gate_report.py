import argparse, json, shutil
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--protocol',type=Path,required=True)
p.add_argument('--deliverables',type=Path,required=True)
a=p.parse_args()
out=a.root/'reports/ipmi121'
out.mkdir(parents=True,exist_ok=True)
report='''# SURFACE2ANATOMY — IPMI 2027 FINAL EVIDENCE REPORT

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
'''
protocol=a.protocol.read_text(encoding='utf-8-sig')
section=protocol.split('45. FINAL RESPONSE FORMAT',1)[1].split('ABSOLUTE RULE:',1)[0]
overrides={'STUDY_VERSION':'Surface2Anatomy-IPMI-121-v1','TARGET_COUNT':'121_DECLARED_CANDIDATES','TARGET_COUNT_VERIFIED':'NO','FRESH_FINAL_TEST':'UNKNOWN_NOT_AUDITED','THIS_IS_THE_ONLY_HEADLINE_ERROR':'NOT_APPLICABLE_NO_RESULT','ASOM_COMPLETED':'NO','ASOM_DIMENSIONS':'NOT_COMPUTED_REQUIRED_121x12','ELAA_COMPLETED':'NO','FEMALE_TARGETS':'uterus,ovary_left,ovary_right,vagina (candidate entries; legacy labels synthetic)','FEMALE_TARGET_SUPPORT_OK':'NO_VERIFIED_SUPPORT','DATA_AUDIT':'NOT_PASSED_ONTOLOGY_GATE_BLOCKED','LEAKAGE_AUDIT':'NOT_COMPLETED','STATISTICS_REVIEW':'NOT_PERFORMED','ANATOMY_REVIEW':'NOT_PERFORMED','HOSTILE_REVIEW':'NOT_PERFORMED','POST_TEST_TUNING':'NO_FINAL_TEST_NOT_RUN','IPMI_DECISION':'DEFERRED_INSUFFICIENT_EVIDENCE'}
for line in section.splitlines():
    if '=' in line and line.split('=',1)[0].strip().replace('_','').isalnum():
        key=line.split('=',1)[0].strip()
        if key.startswith('TOP_5'): continue
        report+=key+'='+overrides.get(key,'NOT_ESTABLISHED')+'\n'
report+='''```

TOP_5_DEFENSIBLE_NOVELTY_CLAIMS: None established. FSEC, CCTQ, ASOM, ELAA, and genuine 121-target breadth remain hypotheses to test.

TOP_5_REMAINING_WEAKNESSES:
1. Four legacy female targets are synthetic, not validated patient anatomy.
2. All-121 valid support and new split membership are unestablished.
3. Freshness and full prior-exposure accounting remain unaudited.
4. New geometry, provenance, semantic equivalence, and leakage audits are incomplete.
5. No new matched experiments, uncertainty calibration, independent reviews, or final results exist.
'''
(out/'FINAL_IPMI_EVIDENCE_REPORT.md').write_text(report,encoding='utf-8')
(out/'00_EXPERIMENT_LINEAGE.md').write_text('''# Surface2Anatomy-IPMI-121-v1 lineage

LEGACY: all existing phase studies, V1/V2/V3 datasets, 104/107 target summaries, 117-slot checkpoints, and earlier 121-slot datasets. Existing evidence is diagnostic only. No legacy performance is promoted into this study.

FINAL IPMI121: newly created candidate ontology and ontology gate audit only. Gate blocked by synthetic female-target provenance. New splits, preprocessing freeze, canonicalizer, model, checkpoints, predictions and results do not exist. No final test has been locked or evaluated. Existing working-tree changes were left intact.
''',encoding='utf-8')
(out/'NEGATIVE_RESULTS_LEDGER.md').write_text('# Negative evidence ledger\n\nOntology gate: 121 declared slots do not establish 121 valid targets; four legacy female labels are synthetic. This is a data-provenance failure, not a negative model-performance result.\n',encoding='utf-8')
shutil.copy2(a.protocol,out/'audit/REQUESTED_PROTOCOL.txt')
shutil.copy2(Path(__file__).with_name('ontology_gate.py'),out/'audit/ontology_gate.py')
shutil.copy2(Path(__file__),out/'audit/write_gate_report.py')
a.deliverables.mkdir(parents=True,exist_ok=True)
shutil.copy2(out/'FINAL_IPMI_EVIDENCE_REPORT.md',a.deliverables/'FINAL_IPMI_EVIDENCE_REPORT.md')
print('Stop report written.')
