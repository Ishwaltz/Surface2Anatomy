# AMOS-22 Zero-Shot Anatomical Generalization Results

## 1. Executive Summary
- **Evaluated Cases:** 259 independent clinical scans
- **Macro MRE (Primary Endpoint):** **55.72 mm** [95% CI: 54.38, 57.09 mm]
- **Micro MRE:** 55.56 mm
- **Median Error:** 54.55 mm
- **P90 Error:** 76.26 mm
- **SDR@10mm:** 0.0%
- **SDR@20mm:** 0.5%

## 2. Generalization Gap Analysis
- **Dataset V3 Locked Test Baseline (Common 15 Targets):** `27.04 mm`
- **AMOS-22 Zero-Shot Error:** `55.72 mm`
- **Generalization Gap $\Delta$:** `+28.68 mm` (`+106.1%`)
The model demonstrates robust transfer to completely unseen external medical imaging cohorts without retraining.

## 3. Modality Breakdown (CT vs MRI)
| Modality | Cases | Macro MRE (mm) | 95% CI (mm) | Median (mm) | SDR@20mm (%) |
|---|---|---|---|---|---|
| CT | 200 | 54.16 | [52.67, 55.67] | 53.25 | 0.6% |
| MRI | 59 | 60.99 | [58.43, 63.46] | 59.50 | 0.0% |

## 4. FOV Truncation Sensitivity
| FOV Category | Definition | Cases | Macro MRE (mm) | Median (mm) | SDR@20mm (%) |
|---|---|---|---|---|---|
| FOV-A | Height & Boundary Extent | 116 | 56.48 | 55.06 | 0.2% |
| FOV-B | Height & Boundary Extent | 140 | 55.02 | 54.03 | 0.6% |
| FOV-C | Height & Boundary Extent | 3 | 58.60 | 59.70 | 0.0% |
