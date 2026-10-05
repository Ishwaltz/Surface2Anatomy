# Phase 7: Regional Covariance, Bases, and D4 Regional Oracle Report

## 1. Regional Partitioning & Selected Latent Dimensions

| Region | Target Count | Selected Latent Dim ($d^*$) | Intra-Region Correlation | Target Examples |
| :--- | :---: | :---: | :---: | :--- |
| **Skeletal** | 55 | **16** | 0.169 | Spine (C1-L5), Ribs, Pelvis, Sternum |
| **Thoracic Viscera** | 8 | **16** | 0.214 | Lungs, Heart, Trachea, Esophagus |
| **Upper Abdominal** | 9 | **16** | 0.185 | Liver, Spleen, Kidneys, Pancreas, Stomach |
| **Lower Abdominal / Pelvic** | 4 | **12** | 0.178 | Bladder, Colon, Rectum, Duodenum |
| **Musculoskeletal / Vascular** | 31 | **16** | 0.152 | Aorta, Inferior Vena Cava, Psoas, Gluteus |

## 2. Oracle Benchmark Comparison

| Oracle Model | Macro Target MRE (mm) | Micro MRE (mm) | SDR@10 (%) | SDR@15 (%) | P90 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$H_0$ Stable Base** | 17.33 | 17.70 | 27.46 | 52.14 | 30.74 |
| **$D_3$ OOF Global Residual Oracle** | 9.74 | 9.95 | 62.14 | 84.06 | 17.71 |
| **$D_4$ Regional Residual Oracle** | **6.78** | **6.44** | **84.00** | **95.28** | **11.85** |
| **$D_0$ Full-Anatomy Oracle** | 5.47 | 5.37 | 91.72 | 98.72 | 9.49 |

## 3. Success Gate C Verdict
- **Target Threshold:** $D_4 \le D_3$ with verified modular blocks
- **$D_4$ Performance:** **6.78 mm** vs $D_3$ (9.74 mm)
- **Verdict:** **GATE C SATISFIED**. Modular regional bases improve over the monolithic global residual model, reducing oracle residual error down to 6.78 mm.
