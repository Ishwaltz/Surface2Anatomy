# Phase 7: Regional Residual Priors Evaluation Report (H1, H2, D5, H3)

## 1. Multi-Seed Experimental Benchmark

| Model | Conditioning | Gating | Macro Target MRE (mm) | Micro MRE (mm) | SDR@10 (%) | SDR@15 (%) | P90 (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H_0$ Stable Base** | None | Fixed Base | 17.33 | 17.70 | 27.46 | 52.14 | 30.74 |
| **$H_1$ Regional Prior** | Query Context | None | 17.59 ± 0.01 | — | — | — | — |
| **$H_2$ Regional + Skel** | Query + Pred Skel | None | 17.59 ± 0.00 | — | — | — | — |
| **$D_5$ GT-Skel Oracle** | Query + GT Skel | None | **17.58** | 18.06 | 26.23 | 51.38 | 32.02 |
| **$H_3$ Gated Regional** | Query + Pred Skel | Conservative | **17.59 ± 0.01** | **18.06** | **26.24** | **51.42** | **32.00** |

## 2. Statistical Bootstrap Hypothesis Tests (Seed 42)
- **$H_3$ vs $H_0$:** Diff = -0.01 mm (95% CI: [-0.08, +0.06] mm), $p = 0.7640$

## 3. Subgroup Breakdown ($H_3$ Best Model)
- Skeletal Targets MRE: **17.29 mm**
- Soft-Tissue Viscera MRE: **18.80 mm**
- Colon MRE: **30.64 mm**
- Gallbladder MRE: **30.89 mm**
