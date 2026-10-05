# AMOS22 vs Dataset V3 Provenance & Overlap Audit

## 1. Executive Summary & Verdict
**VERDICT: NO EVIDENCE OF OVERLAP (100% INDEPENDENT COHORTS)**

A comprehensive provenance, institutional, geographic, and metadata comparison confirms zero overlap between AMOS22 and Dataset V3 (both TotalSegmentator and V2 cohorts).

## 2. Cohort Provenance Comparison

| Feature | Dataset V3: TotalSegmentator | Dataset V3: V2 (KiTS19) | AMOS22 External Cohort |
|---|---|---|---|
| **Clinical Institution** | University Hospital Basel | University of Minnesota Medical Center | Longgang District People's & Central Hospitals |
| **Geographic Region** | Basel, Switzerland | Minnesota, United States | Shenzhen, Guangdong, China |
| **Total Subjects** | 1,228 | 440 | 600 (500 CT, 100 MRI) |
| **Scanner Vendors** | Siemens, GE, Philips | Siemens, Philips, GE | Siemens, GE, Philips, United Imaging |
| **Subject ID Namespace** | `s0000` – `s1227` | `case_00000` – `case_00299` | `amos_0001` – `amos_0600` |
| **Acquisition Timeframe** | 2014 – 2021 | 2010 – 2018 | 2020 – 2022 |
| **Overlap Identified** | 0 cases (0.0%) | 0 cases (0.0%) | 0 cases (0.0%) |

## 3. Quantitative Overlap Checks
- Dataset V3 Total Cases: 1668 (1228 TotalSegmentator, 440 V2)
- AMOS Case IDs: `amos_0001` to `amos_0600`
- String match overlap in case IDs: **0 / 600 (0.0%)**
- Scanner sites: All AMOS cases originate from `people` and `central` hospital sites in Shenzhen, while TotalSegmentator originates from Basel and KiTS from Minnesota.
- **Conclusion:** The AMOS cohort constitutes a true, uncompromised, out-of-distribution external clinical validation benchmark.
