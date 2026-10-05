# V2 Signed Axis Comparison & Non-Regression Audit

> [!WARNING]
> **AUDIT FINDING: IN-DOMAIN DEGRADATION ON V2 (FAIL A & FAIL B)**
> - **V2 System 0 (Baseline):** **18.68 mm** [95% CI: 16.90, 20.95 mm]
> - **V2 System 1 (New External Frame):** **27.23 mm** [95% CI: 23.93, 30.60 mm] ($\Delta_{\text{frame}} = \mathbf{+8.54\text{ mm}}$, **FAIL A: $> +3.0\text{ mm}$**, Wilcoxon $p = 1.49 \times 10^{-7}$)
> - **V2 System 3 (Combined):** **27.23 mm** [95% CI: 23.88, 30.84 mm] ($\Delta_{\text{combined}} = \mathbf{+8.54\text{ mm}}$, **FAIL B: $> +3.0\text{ mm}$**, Wilcoxon $p = 1.49 \times 10^{-7}$)
> - **Governing Interpretation (Section 11):** Because $\Delta > +5\text{ mm}$, we DO NOT call the new frame universally better. Rather, **"canonicalization trades source-domain performance for external robustness."**
> - **Mechanism:** In-domain V2 originally enjoyed access to the internal CT vertebral column anchor (T12 spine, $C_{\text{body}}$), reaching an exceptional 18.68 mm MRE. When constrained to operate strictly on external surface geometry using a population-average dorsal offset ($-56.60\text{ mm}$), patient-to-patient chest depth variance increases error by $+8.54\text{ mm}$ to $27.23\text{ mm}$—which matches AMOS deployable performance ($27.25\text{ mm}$).

---

## 1. Quantitative System Comparison on V2

| Evaluation Cohort | System Configuration | Macro MRE (mm) | 95% Bootstrap CI (mm) | Median (mm) | P90 (mm) | SDR@20mm (%) | $\Delta$ vs Sys 0 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **V2 Locked Test ($N=41$)** | System 0 (Phase10R Original) | **18.68** | [16.90, 20.95] | 16.43 | 26.35 | 70.7% | 0.00 mm |
| **V2 Locked Test ($N=41$)** | System 1 (New External Frame) | **27.23** | [23.93, 30.60] | 22.94 | 43.73 | 34.1% | **+8.54 mm** |
| **V2 Locked Test ($N=41$)** | System 2 (Phase12-FOV Original) | **18.69** | [16.83, 20.97] | 16.45 | 26.38 | 70.7% | +0.00 mm |
| **V2 Locked Test ($N=41$)** | System 3 (FOV + New Frame) | **27.23** | [23.88, 30.84] | 22.96 | 43.69 | 34.1% | **+8.54 mm** |
| **V2 Combined ($N=104$)** | System 0 (Phase10R Original) | **19.14** | [17.72, 20.73] | 16.83 | 27.68 | 71.2% | 0.00 mm |
| **V2 Combined ($N=104$)** | System 3 (FOV + New Frame) | **27.29** | [25.18, 29.58] | 22.84 | 43.84 | 33.7% | +8.15 mm |

---

## 2. Signed Axis Error Comparison (Did V2 Ever Contain the AMOS Bias?)

| Axis / Coordinate | AMOS Raw (Phase 11) | V2 System 0 (Original) | V2 System 1 (New Frame) | V2 System 3 (Combined) |
|---|:---:|:---:|:---:|:---:|
| **Mean $dx$ (Lateral)** | $+0.02\text{ mm}$ | **+0.89 mm** | **+6.82 mm** | **+6.81 mm** |
| **Mean $dy$ (Anterior)** | **$+46.22\text{ mm}$** | **+0.90 mm** | **+8.56 mm** | **+8.56 mm** |
| **Mean $dz$ (Superior)** | **$-15.40\text{ mm}$** | **+1.36 mm** | **+1.36 mm** | **+1.36 mm** |

### Scientific Verdict:
- V2 System 0 exhibits **near-zero signed bias** across all three axes ($dy = +0.90\text{ mm}, dz = +1.36\text{ mm}$).
- In contrast, AMOS System 0 exhibited a massive **$+46.22\text{ mm}$ anterior** and **$-15.40\text{ mm}$ inferior** bias.
- This confirms beyond all doubt that the $+46.22\text{ mm}$ anterior displacement on AMOS was **NOT** an intrinsic bias of the neural network, but a **preprocessing/canonicalization mismatch** caused by forcing the AMOS bounding box midpoint to zero while training on dorsal-aligned point clouds.
