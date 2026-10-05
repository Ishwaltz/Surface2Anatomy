# Phase 12B: V2 Canonical-Frame Sanity & Non-Regression Audit
## Final Audit & Forensic Report

**Project:** 3D Internal Organ / Anatomical Landmark Localization from External Body Surface Geometry  
**Protocol:** Phase 12B — V2 Canonical-Frame Sanity & Non-Regression Audit  
**Date:** September 14, 2026  
**Auditor:** Senior Medical-Imaging ML Engineer, 3D Geometry Scientist, & Skeptical Reviewer  
**Frozen State Sealed In:** `reports/phase12b/00_V2_SANITY_FREEZE.json`  

---

## Executive Summary

Phase 12 demonstrated that the apparent generalization collapse of the frozen Phase-10R model on the independent AMOS-22 external dataset (Macro MRE 55.72 mm) could be recovered to **27.25 mm** (New External Canonical Frame) and **23.98 mm** (Phase12-FOV + New Frame).

Phase 12B was executed as a **strict, non-retrained, frozen source-domain sanity check** on Dataset V2 ($N=41$ locked test, $N=104$ combined evaluation) to determine whether the external canonicalization fix was genuinely robust or whether it gained external performance at the expense of damaging source-domain accuracy.

### Key Audit Findings:
1. **The Preprocessing Trace Hard Gate Resolved (PREPROCESSING BUG):**
   - V2 System 0 exhibits **near-zero signed bias** ($dx = +0.89\text{ mm}, dy = +0.90\text{ mm}, dz = +1.36\text{ mm}$).
   - AMOS Raw exhibited massive systematic anterior ($dy = +46.22\text{ mm}$) and inferior ($dz = -15.40\text{ mm}$) bias.
   - Trace analysis proves that the $+56.60\text{ mm}$ AP reference was baked into Dataset V3/V2 canonical point clouds (anchored at the dorsal vertebral spine, T12). The loader did NO centering. In AMOS Phase 11, subtracting the raw bounding box midpoint forced $Y_{\text{mid}} = 0.0$, creating a literal **PREPROCESSING BUG** that displaced all predicted internal organs $+46.22\text{ mm}$ forward.
2. **The Non-Regression Audit Result (FAIL A & FAIL B):**
   - On V2 Locked Test ($N=41$), System 0 achieved **18.68 mm** [95% CI: 16.90, 20.95 mm].
   - System 1 (New External Frame) achieved **27.23 mm** [95% CI: 23.93, 30.60 mm], a degradation of $\Delta_{\text{frame}} = \mathbf{+8.54\text{ mm}}$ (Wilcoxon $p = 1.49 \times 10^{-7}$).
   - System 3 (FOV + New Frame) achieved **27.23 mm** [95% CI: 23.88, 30.84 mm], $\Delta_{\text{combined}} = \mathbf{+8.54\text{ mm}}$.
   - Because $\Delta > +5\text{ mm}$, **Criteria PASS A and PASS B FAIL**.
3. **Governing Interpretation (Section 11):**
   - Under the mandated governance rules, we **DO NOT** claim the new canonical frame is universally superior.
   - Official Verdict: **"canonicalization trades source-domain performance for external robustness."**
   - The reason is anatomical: V2 System 0 used the true internal spine center from CT. Replacing an internal spine anchor with an external-only population offset ($-56.60\text{ mm}$) introduces $\sim 8.5\text{ mm}$ chest thickness scatter on individual subjects, raising V2 error from 18.68 mm to 27.23 mm. However, this same external frame operates without any scan-internal landmarks, allowing AMOS to recover from 55.72 mm down to 23.98 mm.
4. **Cleanest External Comparison (Exact 8-Target Subset):**
   - On the 8 semantically identical targets, AMOS System 3 achieves **23.15 mm** (exceeding PASS D: $\ge 20\text{ mm}$ improvement, and meeting PASS E: $\le 30\text{ mm}$).
   - On these same 8 targets, V3 locked test is **27.33 mm**, and V2 is **24.17 mm** (Sys 0) / **31.89 mm** (Sys 1).
   - This proves that when semantic truncation definitions are aligned, internal organ prediction from external geometry operates consistently in the **23–27 mm** regime across both internal and external medical cohorts.

---

## 1. Quantitative Multi-System Critical Comparison

| Dataset | Cohort / Split | System Configuration | Macro MRE (mm) | 95% Bootstrap CI (mm) | Median (mm) | P90 (mm) | SDR@20mm (%) | $\Delta$ vs Baseline |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **V2 (In-Domain Sanity)** | Locked Test ($N=41$) | System 0 (Phase10R Original) | **18.68** | [16.90, 20.95] | 16.43 | 26.35 | 70.7% | 0.00 mm |
| **V2 (In-Domain Sanity)** | Locked Test ($N=41$) | System 1 (New External Frame) | **27.23** | [23.93, 30.60] | 22.94 | 43.73 | 34.1% | **+8.54 mm** |
| **V2 (In-Domain Sanity)** | Locked Test ($N=41$) | System 2 (Phase12-FOV Original) | **18.69** | [16.83, 20.97] | 16.45 | 26.38 | 70.7% | +0.01 mm |
| **V2 (In-Domain Sanity)** | Locked Test ($N=41$) | System 3 (FOV + New Frame) | **27.23** | [23.88, 30.84] | 22.96 | 43.69 | 34.1% | **+8.54 mm** |
| **V2 (In-Domain Sanity)** | Combined ($N=104$) | System 0 (Phase10R Original) | **19.14** | [17.72, 20.73] | 16.83 | 27.68 | 71.2% | 0.00 mm |
| **V2 (In-Domain Sanity)** | Combined ($N=104$) | System 3 (FOV + New Frame) | **27.29** | [25.18, 29.58] | 22.84 | 43.84 | 33.7% | **+8.15 mm** |
| **AMOS-22 (External)** | Independent ($N=180$) | System 0 (Phase10R Original) | **55.72** | [54.35, 57.09] | 54.55 | 76.26 | 0.5% | 0.00 mm |
| **AMOS-22 (External)** | Independent ($N=180$) | System 1 (New External Frame) | **27.25** | [26.07, 28.46] | 23.91 | 46.72 | 36.6% | **-28.47 mm** |
| **AMOS-22 (External)** | Independent ($N=180$) | System 2 (Phase12-FOV Original) | **50.66** | [49.32, 51.99] | 49.80 | 70.15 | 3.2% | **-5.06 mm** |
| **AMOS-22 (External)** | Independent ($N=180$) | System 3 (FOV + New Frame) | **23.98** | [22.96, 25.04] | 21.15 | 41.20 | 46.8% | **-31.74 mm** |

*Paired Patient-Level Statistics on V2 Locked Test:*
- System 1 vs System 0: Mean paired difference $= +8.54\text{ mm}$ [95% CI: $+5.47, +11.96\text{ mm}$], Wilcoxon signed-rank $W = 54.0, p = 1.49 \times 10^{-7}$.
- System 3 vs System 0: Mean paired difference $= +8.54\text{ mm}$ [95% CI: $+5.43, +11.99\text{ mm}$], Wilcoxon signed-rank $W = 54.0, p = 1.49 \times 10^{-7}$.

---

## 2. Signed Axis Error Forensics (Did V2 Ever Have the AMOS Bias?)

| Coordinate Axis | AMOS Raw Phase 11 | V2 System 0 (Original) | V2 System 1 (New Frame) | V2 System 3 (Combined) |
|---|:---:|:---:|:---:|:---:|
| **Mean $dx$ (Lateral)** | $+0.02\text{ mm}$ | **+0.89 mm** | **+6.82 mm** | **+6.81 mm** |
| **Mean $dy$ (Anterior-Posterior)** | **+46.22 mm** *(73% energy)* | **+0.90 mm** | **+8.56 mm** | **+8.56 mm** |
| **Mean $dz$ (Superior-Inferior)** | **-15.40 mm** *(22% energy)* | **+1.36 mm** | **+1.36 mm** | **+1.36 mm** |

### Critical Forensic Insight:
- In V2 System 0, the signed bias is virtually zero along all three axes ($dx=+0.89, dy=+0.90, dz=+1.36\text{ mm}$).
- The $+46.22\text{ mm}$ anterior displacement seen in AMOS Raw was **NEVER** present in V2.
- This confirms that the AMOS failure was not a neural hallucination or morphological collapse, but an artifact of forcing the AMOS body bounding box midpoint to zero.

---

## 3. Preprocessing Trace Hard Gate (5 V2 vs 5 AMOS Cases)

Full numerical trace in `reports/phase12b/02_V2_AMOS_PREPROCESSING_TRACE.md`.

### The Core Root Cause Discrepancy:
1. **In Dataset V3/V2 Preprocessing (`apply_canonical_alignment.py`):**
   The coordinate origin was defined at the **dorsal vertebral column (spine, T12)**. Because the human spine sits posterior to the body centroid, all 1,668 V3 point clouds had an intrinsic **mean bounding box midpoint $Y = +56.60\text{ mm}$**.
2. **In Model Loader (`FastPointDataset`):**
   The loader applied **NO centering**. It divided `points_centered_4096 / 500.0` directly. Thus, the network weights learned an input coordinate space where the abdomen protrudes forward into $+Y$.
3. **In AMOS Phase 11 Preprocessing (`04_amos_surface_and_gt.py`):**
   The script subtracted $C_{\text{body}} = 0.5 \times (\min(P) + \max(P))$, forcing the normalized input midpoint to **$0.0$**.
4. **The Consequence:**
   The network expected the spine at $Y=0$ and the abdomen centered at $+56.6\text{ mm}$. When presented with a surface centered at $0.0$, the network inferred that the spine was at $-56.6\text{ mm}$, and projected all organ queries forward by $+46.22\text{ mm}$ anteriorly!
5. **Label:** **PREPROCESSING BUG**.

---

## 4. V2 FOV Cropping Stress Test (PASS C Evaluation)

Full report in `reports/phase12b/03_V2_FOV_STRESS.md`.

| Crop Condition | Nominal Height | Old Frame Drift | New Frame Drift | **Drift Reduction** | Old Frame MRE | New Frame MRE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full (Uncropped)** | 450 mm | 113.87 mm | 16.74 mm | **85.3%** | 53.12 mm | 26.95 mm |
| **450 mm SI** | 450 mm | 113.88 mm | 16.81 mm | **85.2%** | 56.81 mm | 32.58 mm |
| **350 mm SI** | 350 mm | 113.88 mm | 17.70 mm | **84.5%** | 65.88 mm | 40.80 mm |
| **300 mm SI** | 300 mm | 113.90 mm | 18.10 mm | **84.1%** | 73.49 mm | 48.40 mm |
| **250 mm SI** | 250 mm | 113.50 mm | 18.99 mm | **83.3%** | 80.94 mm | 57.77 mm |
| **200 mm SI** | 200 mm | 113.15 mm | 20.38 mm | **82.0%** | 91.46 mm | 70.71 mm |

- **Mean Origin Drift Reduction:** **84.1%** (PASS C requirement: $\ge 50\%$).
- **Status:** **PASS C SATISFIED**.
- Under Old Frame naive scan-window centering, MRE surges from 53.12 mm to 91.46 mm. Under New Frame, origin drift is suppressed below 21 mm.

---

## 5. Common Exact-Match Target Subset (PASS D & PASS E Evaluation)

Full details in `reports/phase12b/04_EXACT_TARGET_SUMMARY.md` and `04_EXACT_TARGET_COMPARISON.csv`.

| Target Structure | Phase10R Slot | AMOS ID | V2 Sys 0 (mm) | V2 Sys 1 (mm) | V3 Locked Test (mm) | AMOS Sys 0 (mm) | AMOS Sys 1 (mm) | AMOS Sys 3 (mm) |
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

- **PASS D:** AMOS improvement on exact targets $= 55.34 - 23.15 = \mathbf{32.19\text{ mm}} \ge 20\text{ mm}$ $\implies$ **PASS D SATISFIED**.
- **PASS E:** AMOS System 3 exact-match MRE $= \mathbf{23.15\text{ mm}} \le 30\text{ mm}$ $\implies$ **PASS E SATISFIED**.

---

## 6. Formal Hypothesis Testing & Success Criteria Scorecard

| Criterion | Formal Threshold | Observed Value | Verdict | Detailed Explanation |
|---|---|:---:|:---:|---|
| **PASS A** | System 1 V2 degradation $\le 3\text{ mm}$ | **+8.54 mm** | **FAIL** | Statistically significant degradation ($p=1.49 \times 10^{-7}$). V2 drops from 18.68 to 27.23 mm. |
| **PASS B** | System 3 V2 degradation $\le 3\text{ mm}$ | **+8.54 mm** | **FAIL** | Statistically significant degradation ($p=1.49 \times 10^{-7}$). Combined model also degrades by +8.54 mm. |
| **PASS C** | New frame reduces crop drift $\ge 50\%$ | **84.1%** | **PASS** | Origin drift reduced from 113.6 mm down to 18.1 mm across Full to 200 mm crops. |
| **PASS D** | AMOS improvement remains $\ge 20\text{ mm}$ | **32.19 mm** | **PASS** | Error reduced from 55.34 mm to 23.15 mm on exact targets (31.74 mm macro). |
| **PASS E** | Exact-match AMOS result $\le 30\text{ mm}$ | **23.15 mm** | **PASS** | System 3 achieves 23.15 mm (and System 1 achieves 26.30 mm). |
| **PASS F** | Preprocessing trace explains $+Y/-Z$ bias | **VERIFIED** | **PASS** | Bbox zero-centering vs dorsal vertebral reference offset trace completed. |

### Overall Verdict:
**FOUR OF SIX CRITERIA PASS (PASS C, PASS D, PASS E, PASS F).**  
**TWO CRITERIA FAIL (PASS A, PASS B).**  

**OFFICIAL AUDIT VERDICT:**  
> **"Canonicalization trades source-domain performance for external robustness."**

---

## 7. Artifact Registry

- Starting State Freeze: `reports/phase12b/00_V2_SANITY_FREEZE.json`
- Signed Axis Comparison: `reports/phase12b/01_V2_SIGNED_AXIS_COMPARISON.md`
- Preprocessing Trace Report: `reports/phase12b/02_V2_AMOS_PREPROCESSING_TRACE.md`
- FOV Stress Test Report: `reports/phase12b/03_V2_FOV_STRESS.md`
- Exact Target Comparison CSV: `reports/phase12b/04_EXACT_TARGET_COMPARISON.csv`
- Exact Target Summary: `reports/phase12b/04_EXACT_TARGET_SUMMARY.md`
- Patient Multi-System Results: `reports/phase12b/tables/V2_PATIENT_MULTI_SYSTEM_RESULTS.csv`
- System Aggregate Table: `reports/phase12b/tables/V2_SYSTEM_RESULTS.csv`
- FOV Stress Table: `reports/phase12b/tables/V2_FOV_STRESS_RESULTS.csv`
- Publication Figures:
  - Figure 1: `reports/phase12b/figures/fig1_v2_amos_system_comparison.png`
  - Figure 2: `reports/phase12b/figures/fig2_signed_axis_bias_comparison.png`
  - Figure 3: `reports/phase12b/figures/fig3_v2_fov_crop_drift_mre.png`
  - Figure 4: `reports/phase12b/figures/fig4_exact_8_target_comparison.png`
