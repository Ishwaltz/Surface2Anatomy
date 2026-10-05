# Patient-Level Forensics and Failure Modes Report

## 1. Extremes of Patient Distribution

| Category | Case ID | Source Dataset | Valid Targets | Patient Macro MRE (mm) | Body Width (mm) | Body Depth (mm) | Body Height (mm) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Best | `v3_v2_case_372` | v2 | 101 | 9.72 | 336.6 | 253.5 | 594.0 |
| Best | `v3_totalseg_s1335` | totalsegmentator | 26 | 11.19 | 183.8 | 184.5 | 175.5 |
| Best | `v3_v2_case_163` | v2 | 65 | 11.81 | 319.3 | 205.6 | 387.9 |
| Best | `v3_v2_case_351` | v2 | 65 | 11.98 | 307.1 | 225.7 | 388.7 |
| Best | `v3_v2_case_078` | v2 | 101 | 11.99 | 381.4 | 199.8 | 619.4 |
| Worst | `v3_totalseg_s0233` | totalsegmentator | 42 | 228.69 | 135.0 | 135.0 | 130.5 |
| Worst | `v3_totalseg_s0482` | totalsegmentator | 82 | 94.59 | 359.1 | 359.9 | 235.5 |
| Worst | `v3_totalseg_s0313` | totalsegmentator | 42 | 91.77 | 162.0 | 162.0 | 252.3 |
| Worst | `v3_totalseg_s0036` | totalsegmentator | 34 | 79.05 | 179.8 | 179.4 | 136.5 |
| Worst | `v3_totalseg_s1024` | totalsegmentator | 99 | 65.35 | 378.0 | 496.6 | 583.4 |

## 2. Morphological Correlates of Error

- **Torso Width Correlation:** $r = -0.283$
- **Torso Depth Correlation:** $r = -0.105$
- **Torso Height Correlation:** $r = -0.283$
- **BMI Proxy Correlation:** $r = 0.091$
