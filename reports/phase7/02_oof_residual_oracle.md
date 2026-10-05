# Phase 7: Out-of-Fold Residual Construction & D3 Oracle Report

## 1. 5-Fold OOF Residual Methodology
To prevent in-sample residual leakage, 5-fold cross-validation was conducted across all 352 training patients. For every patient $i$, predictions $P_i^{\text{OOF}}$ were produced by models trained without seeing patient $i$.
- **OOF Generalization Baseline on Train:** **24.45 mm** (SDR@10: 13.98%, P90: 42.29 mm)
- **OOF Residual Vector:** $R_i^{\text{OOF}} = P_i^{\text{gt}} - P_i^{\text{OOF}} \in \mathbb{R}^{107 \times 3}$

## 2. D3 OOF Residual Oracle Performance (Validation Cohort, 44 Patients)

| Latent Dimension ($d_{res}$) | D3 Oracle Macro MRE (mm) | Micro MRE (mm) | SDR@10 (%) | SDR@15 (%) | P90 (mm) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **8**  | 12.21  | 12.24  | 47.22  | 74.53  | 21.68  |
| **12** | 10.74 | 10.90 | 56.34 | 80.30 | 19.15 |
| **16** | **9.74** | **9.95** | **61.47** | **84.23** | **17.63** |
| **24** | 8.87 | 8.96 | 67.85 | 88.37 | 15.87 |
| **32** | 8.06 | 8.09 | 73.62 | 91.64 | 14.17 |

## 3. Success Gate B Verdict
- **Target Threshold:** $D_3 \le 10.0\text{ mm}$
- **Achieved D3 Macro MRE:** **9.74 mm**
- **Verdict:** **GATE B SATISFIED**. The residual oracle reproduces reliably under true out-of-fold methodology (9.74 mm vs Phase 6's 10.05 mm), verifying that model generalization errors possess structured, low-dimensional capacity!
