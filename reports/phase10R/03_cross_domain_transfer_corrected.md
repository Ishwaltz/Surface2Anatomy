# Phase 10R: Corrected Cross-Domain Transfer Analysis

All domain variants trained under matched 65-epoch protocol across seeds 42, 43, 44.

| Training Cohort | Val V2 (mm) | Val TotalSegmentator (mm) |
|---|---|---|
| **V2-only (N=336)** | 26.76 ± 0.98 | 62.60 ± 0.92 |
| **TotalSegmentator-only (N=998)** | 26.49 ± 0.96 | 28.86 ± 0.23 |
| **Pooled (N=1334)** | **19.96 ± 0.35** | **28.05 ± 0.58** |

- **Delta V2 (Pooled - V2-only):** -6.80 mm
- **Delta TS (Pooled - TS-only):** -0.80 mm
- **Scientific Conclusion:** Pooled training confirms positive transfer without negative transfer degradation on both cohorts.
