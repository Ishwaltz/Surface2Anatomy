# AMOS-22 Modality Comparison: CT vs MRI Generalization

## 1. Modality-Stratified Performance
The frozen Phase-10R model was evaluated across both modalities in AMOS-22 without fine-tuning:

| Modality | Cohort Size ($N$) | Macro MRE (mm) | 95% Bootstrap CI (mm) | Median (mm) | SDR@20mm (%) | SDR@40mm (%) |
|---|---|---|---|---|---|---|
| **CT** | 200 | **54.16** | [52.67, 55.67] | **53.25** | 0.6% | 18.2% |
| **MRI** | 59 | **60.99** | [58.43, 63.46] | **59.50** | 0.0% | 8.6% |
| **Overall** | **259** | **55.72** | **[54.38, 57.09]** | **54.55** | **0.5%** | **16.0%** |

## 2. Statistical Analysis of the Modality Gap
- **Modality Discrepancy ($\Delta_{\text{MRI} - \text{CT}}$):** $+6.83\text{ mm}$ ($+12.6\%$)
- **Two-sample Mann-Whitney $U$ test:** $p < 10^{-4}$, confirming a statistically significant difference in error distribution.

## 3. Physical and Clinical Root Causes
1. **Craniocaudal Field-of-View Coverage:**
   - AMOS CT scans exhibit an average body height of $491.4\text{ mm}$, providing extensive thoracic and abdominal coverage.
   - AMOS MRI scans average only $290.2\text{ mm}$ in height (concentrated around specific abdominal regions). The truncated superior/inferior boundary reduces axial contextual geometry for the neural backbone.
2. **Boundary Segmentation Fidelity:**
   - In CT, the exterior human skin interface is delineated by sharp Hounsfield Unit contrasts (air at $\sim -1000\text{ HU}$ vs tissue at $\sim 0\text{ HU}$), yielding crisp isosurfaces.
   - In MRI, RF coil receive inhomogeneities and chemical shift artifacts yield softer intensity boundaries.
3. **Physiological Acquisition Times:**
   - CT scans are acquired in seconds during a single breath-hold.
   - Abdominal MRI sequences span multiple minutes, introducing diaphragmatic motion blur and organ deformation.

Despite these physical imaging differences, both modalities maintain median errors under $60\text{ mm}$ in a completely zero-shot evaluation without sensor-specific adaptation.
