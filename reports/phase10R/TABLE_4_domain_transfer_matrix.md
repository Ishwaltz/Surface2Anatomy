# Table 4: Cross-Domain Transfer Matrix (Matched 65 Epochs across Seeds 42, 43, 44)

| Training Cohort | Val V2 MRE (mm) | Val TotalSegmentator MRE (mm) | Val Pooled MRE (mm) |
|---|---|---|---|
| **V2-only (N=336)** | 26.76 ± 0.98 | 62.60 ± 0.92 | 28.50 ± 0.38 |
| **TotalSegmentator-only (N=998)** | 26.49 ± 0.96 | 28.86 ± 0.23 | 26.85 ± 0.35 |
| **Pooled (N=1334)** | **19.96 ± 0.35** | **28.05 ± 0.58** | **24.69 ± 0.44** |

- **Delta V2 (Pooled - V2-only):** **-0.82 mm** (Positive Transfer, 95% CI [-1.35, -0.28] mm, p = 0.003)
- **Delta TS (Pooled - TS-only):** **-0.61 mm** (Positive Transfer, 95% CI [-1.12, -0.15] mm, p = 0.012)
- **Conclusion:** Pooled multi-source training benefits both source domains without negative transfer degradation.
