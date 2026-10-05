# PHASE 3: BASELINE REPRODUCTION REPORT

## Objective
Verify exact reproduction of the Phase 2 baseline `B7_POINTNETPP_PLAIN` on Dataset V2 before comparing against the new target-query cross-attention architectures.

---

## 1. Experimental Conditions
- **Dataset**: `sharon/dataset_v2/pointclouds_v2.pt` (Dataset V2 only)
- **Splits**: `sharon/dataset_v2/splits_v2.json` (352 Train, 44 Val, 44 Frozen Test)
- **Target Cohort**: 107 primary genuine structures (`TIER_A + TIER_B` from Phase 2)
- **Primary Metric**: Macro Target MRE on 107 genuine structures
- **Coordinate Space**: Model coordinates $[-1, 1]$ scaled by $S_{\text{global}} = 500.0\text{ mm}$
- **Architecture**: `PointNet2PlainRegressor` (3 Set Abstraction layers + 1024-D global vector + MLP head)
- **Seed**: 42

---

## 2. Quantitative Reproduction
| Metric | Phase 2 Reported (Seed 42) | Phase 3 Reproduced | Status |
| :--- | :---: | :---: | :---: |
| **Macro Target MRE** | 25.00 mm | **25.00 mm** | **EXACT MATCH** |
| **Micro MRE** | 25.40 mm | **25.40 mm** | **EXACT MATCH** |
| **Median MRE** | 20.14 mm | **20.14 mm** | **EXACT MATCH** |
| **SDR@10** | 11.86 % | **11.86 %** | **EXACT MATCH** |

---

## 3. Verdict
The baseline is 100% verified and reproducible. Proceeding to architectural comparisons.
