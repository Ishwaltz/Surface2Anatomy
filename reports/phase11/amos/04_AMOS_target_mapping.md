# AMOS22 Target Mapping & Semantic Alignment

## 1. Protocol Integrity & Freezing
This mapping was constructed strictly prior to running model inference, based exclusively on official anatomical definitions and nomenclature.
- **SHA-256 Checksum:** `15bb5c66500ab4bb4a2544f8d103869dd59972429780cfe0b0f8cae49d856362`

## 2. Definitive Target Alignment Table

| AMOS ID | AMOS Label | Phase 10R Target | Model Index | Primary V3 Target? | Mapping Status | Sex Specific? |
|---|---|---|---|---|---|---|
| 1 | spleen | `spleen` | 0 | YES | EXACT_MATCH | NO |
| 2 | right kidney | `kidney_right` | 1 | YES | EXACT_MATCH | NO |
| 3 | left kidney | `kidney_left` | 2 | YES | EXACT_MATCH | NO |
| 4 | gallbladder | `gallbladder` | 3 | YES | EXACT_MATCH | NO |
| 5 | esophagus | `esophagus` | 14 | YES | EXACT_MATCH | NO |
| 6 | liver | `liver` | 4 | YES | EXACT_MATCH | NO |
| 7 | stomach | `stomach` | 5 | YES | EXACT_MATCH | NO |
| 8 | aorta | `aorta` | 51 | YES | EXACT_MATCH | NO |
| 9 | inferior vena cava | `inferior_vena_cava` | 62 | YES | EXACT_MATCH | NO |
| 10 | pancreas | `pancreas` | 6 | YES | EXACT_MATCH | NO |
| 11 | right adrenal gland | `adrenal_gland_right` | 7 | YES | EXACT_MATCH | NO |
| 12 | left adrenal gland | `adrenal_gland_left` | 8 | YES | EXACT_MATCH | NO |
| 13 | duodenum | `duodenum` | 18 | YES | EXACT_MATCH | NO |
| 14 | urinary bladder | `urinary_bladder` | 20 | YES | EXACT_MATCH | NO |
| 15 | prostate/uterus | `prostate` | 21 | NO | SEX_SPECIFIC_MATCH | YES |

## 3. Analysis of Target Support
- **Primary Benchmark Intersection:** 14 organs (`spleen`, `kidney_right`, `kidney_left`, `gallbladder`, `esophagus`, `liver`, `stomach`, `aorta`, `inferior_vena_cava`, `pancreas`, `adrenal_gland_right`, `adrenal_gland_left`, `duodenum`, `urinary_bladder`) are members of the primary 104 benchmark target set.
- **Secondary / Auxiliary Targets:** AMOS label 15 (`prostate/uterus`) represents male prostate and female uterus. In Phase 10R, slot 21 corresponds to `prostate` (non-primary target). For female patients, uterus is excluded from model error evaluation as it is not in the 117-target ontology.
- **Common Intersection:** The headline common-target cross-dataset benchmark between Dataset V3 test set and AMOS will evaluate the matched 14 abdominal organ targets.
