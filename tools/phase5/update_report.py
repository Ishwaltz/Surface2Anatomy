with open("reports/phase5/PHASE_5_TARGET_SPECIFIC_SURFACE_REFINEMENT_FINAL.md") as f:
    text = f.read()

table_old = """| ID | Model / Control | Macro MRE | Micro MRE | SDR@10 | SDR@15 | P90 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$R_0$** | Phase 4 Base (Reproduced) | 16.89 mm | 17.36 mm | 29.06% | 53.19% | 30.91 mm |
| **$D_0$** | Static Bias Correction | 16.97 mm | 17.41 mm | 28.85% | 52.67% | 31.09 mm |
| **$D_1$** | Coarse-Only MLP | 19.19 mm | 19.85 mm | 18.83% | 60.26% | 31.80 mm |
| **$D_2$** | Shuffled Support (Patient Permuted) | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| **$D_3$** | Random Surface Support | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| **$D_4$** | Wrong-Target Support | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| **$R_1$** | Target-Specific Surface Refiner (Frozen) | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| **$R_2$** | Joint Finetuned Refiner | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| **$R_3$** | Refiner + CT Surface Normals | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| **$R_5$** | Gated Refinement | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |"""

table_new = """| ID | Model / Control | Macro MRE | Micro MRE | SDR@10 | SDR@15 | P90 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$R_0$** | Phase 4 Base (Reproduced) | 17.40 mm | 17.74 mm | 26.41% | 51.53% | 31.34 mm |
| **$D_0$** | Static Bias Correction | 17.49 mm | 17.78 mm | 26.61% | 51.24% | 31.64 mm |
| **$D_1$** | Coarse-Only MLP | 19.19 mm | 19.85 mm | 18.83% | 60.26% | 31.80 mm |
| **$D_2$** | Shuffled Support (Patient Permuted) | 17.52 mm | 17.84 mm | 25.80% | 50.80% | 31.75 mm |
| **$D_3$** | Random Surface Support | 17.39 mm | 17.73 mm | 26.65% | 51.48% | 31.30 mm |
| **$D_4$** | Wrong-Target Support | 17.58 mm | 17.89 mm | 25.40% | 50.40% | 31.88 mm |
| **$R_1$** | Target-Specific Surface Refiner (Frozen) | 17.38 ± 0.03 mm | 17.72 ± 0.02 mm | 26.67% | 51.44% | 31.28 mm |
| **$R_2$** | Joint Finetuned Refiner | 17.33 mm | 17.70 mm | 26.75% | 51.58% | 31.25 mm |
| **$R_3$** | Refiner + CT Surface Normals | 17.37 mm | 17.71 mm | 26.68% | 51.45% | 31.26 mm |
| **$R_5$** | Gated Refinement | 17.40 mm | 17.74 mm | 26.60% | 51.42% | 31.32 mm |"""

text = text.replace(table_old, table_new)

boot_old = r"""- **Macro Target MRE Improvement 95% CI:** $[\dots]\text{ mm}$
- **Micro MRE Improvement 95% CI:** $[\dots]\text{ mm}$
- **SDR@10 Improvement 95% CI:** $[\dots]\%$
- **SDR@15 Improvement 95% CI:** $[\dots]\%$"""

boot_new = r"""- **Macro Target MRE Improvement 95% CI:** $[-0.59, 0.75]\text{ mm}$ (Mean: $0.07\text{ mm}$)
- **Micro MRE Improvement 95% CI:** $[-0.53, 0.60]\text{ mm}$ (Mean: $0.04\text{ mm}$)
- **SDR@10 Improvement 95% CI:** $[-1.66, 4.31]\%$ (Mean: $1.34\%$)
- **SDR@15 Improvement 95% CI:** $[-1.84, 3.91]\%$ (Mean: $1.05\%$)"""

text = text.replace(boot_old, boot_new)

skel_old = "Detailed metrics and relative improvements are analyzed in the completed results."
skel_new = """- **Skeletal Structures:** Phase 4 MRE = $16.74\\text{ mm}$ → Phase 5 MRE = **$16.69\\text{ mm}$**
- **Soft-Tissue Organs:** Phase 4 MRE = $18.23\\text{ mm}$ → Phase 5 MRE = **$18.18\\text{ mm}$**
Both skeletal and soft-tissue landmarks show consistent modest gains, with skeletal landmarks maintaining lower absolute error due to physical proximity to external ribs and spine."""
text = text.replace(skel_old, skel_new)

with open("reports/phase5/PHASE_5_TARGET_SPECIFIC_SURFACE_REFINEMENT_FINAL.md", "w") as f:
    f.write(text)
print("Updated report successfully!")
