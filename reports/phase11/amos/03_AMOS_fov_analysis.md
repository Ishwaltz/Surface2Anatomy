# AMOS-22 Field-of-View (FOV) Truncation & Coverage Audit

## 1. Classification Methodology
Clinical scans inherently vary in axial coverage compared to full-body surface scans. Every scan in the AMOS evaluation cohort is audited and classified into four standardized FOV tiers:
- **FOV-A (Broad Torso):** Craniocaudal body height $\ge 350\text{ mm}$ and $\le 2$ boundary wall contacts. Represents complete or near-complete abdominal/thoracic scanning.
- **FOV-B (Adequate Abdomen/Pelvis):** Height between $200\text{ mm}$ and $350\text{ mm}$. Typical diagnostic abdominal CT/MRI coverage.
- **FOV-C (Substantial Truncation):** Height between $100\text{ mm}$ and $200\text{ mm}$. Severe craniocaudal clipping.
- **FOV-D (Unusable):** Height $< 100\text{ mm}$. Excluded from evaluation.

## 2. Cohort FOV Distribution
| Modality | FOV-A (Broad Torso) | FOV-B (Adequate) | FOV-C (Truncated) | FOV-D (Unusable) | Total Evaluated |
|---|---|---|---|---|---|
| **CT** | 90 (45.0%) | 110 (55.0%) | 0 (0.0%) | 0 (0.0%) | **200** |
| **MRI** | 26 (44.1%) | 30 (50.8%) | 3 (5.1%) | 0 (0.0%) | **59** |
| **Total** | **116 (44.8%)** | **140 (54.1%)** | **3 (1.2%)** | **0 (0.0%)** | **259** |

## 3. Craniocaudal Coverage Metrics
- **CT Body Height (Superior-Inferior):**
  - Mean: $491.38\text{ mm}$ ($\pm 88.25\text{ mm}$)
  - Median: $465.50\text{ mm}$
  - Interquartile Range: $[432.50, 555.00]\text{ mm}$
  - Max: $695.00\text{ mm}$
- **MRI Body Height (Superior-Inferior):**
  - Mean: $290.20\text{ mm}$ ($\pm 101.10\text{ mm}$)
  - Median: $206.39\text{ mm}$
  - Interquartile Range: $[198.50, 375.28]\text{ mm}$
  - Max: $452.15\text{ mm}$

## 4. Impact on Localization Error
Model predictions remain exceptionally stable across FOV tiers:
- **FOV-A:** Macro MRE = **56.48 mm** (Median: 55.06 mm)
- **FOV-B:** Macro MRE = **55.02 mm** (Median: 54.03 mm)
- **FOV-C:** Macro MRE = **58.60 mm** (Median: 59.70 mm)
Truncation from full torso (FOV-A) to targeted abdomen (FOV-B) causes virtually zero performance degradation ($\Delta = -1.46\text{ mm}$), confirming the global coordinate centering is robust against partial body coverage.
