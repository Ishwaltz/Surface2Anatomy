# Phase 10R: Matched Cross-Domain Transfer Analysis (Final)

All models trained for 65 epochs on identical architectures across seeds 42, 43, 44.

### Domain Transfer Matrix (Macro MRE mm, Mean ± SD across 3 seeds)

| Training Cohort | V2 Validation ($N=41$) | TotalSegmentator Validation ($N=125$) | Full Validation ($N=166$) |
|---|---|---|---|
| **V2 TRAIN** | 26.76 ± 0.98 | 62.60 ± 0.92 | 53.75 |
| **TS TRAIN** | 26.49 ± 0.96 | 28.86 ± 0.23 | 28.27 |
| **POOLED TRAIN** | 19.96 ± 0.35 | 28.05 ± 0.58 | 26.05 |

### Statistical Transfer Evaluation

- **$\Delta_{V2}$ (Pooled vs V2-only on V2 Val):** **-6.80 mm** (95% CI: [-7.85 mm, -5.74 mm], $p < 0.0002$)
- **$\Delta_{TS}$ (Pooled vs TS-only on TS Val):** **-0.80 mm** (95% CI: [-1.42 mm, -0.19 mm], $p = 0.012$)

> [!NOTE]
> **APPROVED PUBLICATION WORDING:**
> *"No negative transfer was observed under the matched experimental protocol; pooled training improved localization on both evaluated source domains."*
