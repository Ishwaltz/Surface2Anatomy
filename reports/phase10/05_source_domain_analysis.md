# Source-Domain and Cross-Domain Transfer Matrix Analysis

## 1. Domain Transfer Evaluation Matrix

| Training Domain | Training Subjects (N) | Evaluation on V2 Val (mm) | Evaluation on TotalSegmentator Val (mm) | Full Val Macro (mm) |
| :--- | :---: | :---: | :---: | :---: |
| **V2 Only** ($N_{V2}=336$) | 336 | 36.32 | 75.12 | 60.39 |
| **TotalSegmentator Only** ($N_{TS}=998$) | 998 | 30.18 | 32.94 | 31.89 |
| **Pooled Cohort** ($N_{FULL}=1334$) | 1334 | **18.83** | **26.59** | **23.36** |

## 2. Transfer Gap and Negative Transfer Analysis

- **V2 Transfer Delta:** $\Delta_{V2} = \text{Pooled} - \text{In-Domain} = 18.83 - 36.32 = -17.49\text{ mm}$
- **TotalSegmentator Transfer Delta:** $\Delta_{TS} = \text{Pooled} - \text{In-Domain} = 26.59 - 32.94 = -6.35\text{ mm}$
- **Conclusion:** Negative transfer is strictly absent ($\Delta \le 0$). Pooling disparate hospital cohorts yields substantial positive cross-domain synergy.
