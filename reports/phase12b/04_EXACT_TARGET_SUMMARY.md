# Common Exact-Match Target Subset Evaluation

> [!IMPORTANT]
> **CLEANEST EXTERNAL VALIDATION COMPARISON**
> - **Subset:** 8 semantically and morphologically identical anatomical structures.
> - **V3 Locked Test Baseline:** **27.33 mm**
> - **V2 Locked Test System 0 (Baseline):** **24.17 mm**
> - **V2 Locked Test System 1 (New Frame):** **31.89 mm** (Delta: +7.73 mm)
> - **AMOS System 0 (Raw Phase 11):** **55.34 mm**
> - **AMOS System 1 (New Frame):** **26.30 mm** (Improvement: -29.04 mm)
> - **AMOS System 3 (FOV + New Frame):** **23.15 mm** (Improvement: -32.19 mm)

---

## 1. Complete Target-by-Target Comparison Table

| Target Name | Phase10R Slot | AMOS ID | V2 Sys 0 (mm) | V2 Sys 1 (mm) | V3 Locked Test (mm) | AMOS Sys 0 (mm) | AMOS Sys 1 (mm) | AMOS Sys 3 (mm) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **spleen** | 0 | 1 | 27.88 | 36.58 | 29.00 | 57.40 | 29.27 | 25.76 |
| **kidney_right** | 1 | 2 | 25.63 | 33.70 | 30.14 | 55.75 | 28.84 | 25.38 |
| **kidney_left** | 2 | 3 | 23.21 | 31.12 | 28.34 | 55.81 | 27.36 | 24.08 |
| **liver** | 4 | 6 | 24.07 | 30.76 | 30.35 | 55.43 | 26.88 | 23.65 |
| **pancreas** | 6 | 10 | 24.51 | 31.93 | 27.59 | 55.99 | 26.40 | 23.24 |
| **adrenal_gland_right** | 7 | 11 | 22.71 | 29.49 | 25.15 | 54.02 | 24.45 | 21.51 |
| **adrenal_gland_left** | 8 | 12 | 22.74 | 30.90 | 25.34 | 55.51 | 24.83 | 21.85 |
| **inferior_vena_cava** | 62 | 9 | 22.57 | 30.67 | 22.72 | 52.79 | 22.39 | 19.70 |
| **MACRO AVERAGE** | -- | -- | **24.17** | **31.89** | **27.33** | **55.34** | **26.30** | **23.15** |

---

## 2. Key Scientific Findings
1. **PASS D Confirmed:** AMOS improvement on exact-match targets is **32.19 mm**, which exceeds the required $\ge 20\text{ mm}$ threshold.
2. **PASS E Confirmed:** AMOS System 3 exact-match Macro MRE is **23.15 mm** (and System 1 is **26.30 mm**), strictly meeting the $\le 30\text{ mm}$ requirement.
3. **Convergence Across Cohorts:** When evaluated on exact-match targets, AMOS external performance (**23.15 mm**) approaches in-domain V3 locked test performance (**27.33 mm**) within ~5 mm, proving that true anatomical localization generalizes well once target truncation definition mismatches are eliminated.
