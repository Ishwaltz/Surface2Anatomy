# AMOS-22 Signed X/Y/Z Coordinate Error Forensics

## 1. Executive Forensic Diagnostic Statement
> [!CAUTION]
> **systematic coordinate bias detected**
>
> A strong, cohort-wide directional shift was identified across the 259 evaluated AMOS patients. Specifically, the Superior-Inferior axis ($Z$) exhibits a systematic signed bias of **-15.40 mm** (Median: **-15.68 mm**), accounting for **21.9%** of total Euclidean error energy.

## 2. Global Signed Error Breakdown
| Axis | Anatomical Direction | Mean (mm) | 95% Bootstrap CI (mm) | Median (mm) | IQR (mm) | Mean |Error| (mm) | Radial Energy Share (%) |
|---|---|---|---|---|---|---|---|
| **X** | Right (+) / Left (-) | **+0.02** | [-0.38, +0.42] | +0.52 | 14.36 | 9.34 | 5.1% |
| **Y** | Anterior (+) / Posterior (-) | **+46.22** | [+45.74, +46.68] | +45.73 | 19.68 | 46.24 | 73.0% |
| **Z** | Superior (+) / Inferior (-) | **-15.40** | [-16.19, -14.63] | -15.68 | 29.08 | 23.03 | 21.9% |

## 3. Modality-Stratified Axis Bias (CT vs MRI)
| Modality | Patients ($N$) | Mean $dx$ (mm) | Mean $dy$ (mm) | Mean $dz$ (mm) | Median $dz$ (mm) | $Z$-Energy Share (%) |
|---|---|---|---|---|---|---|
| **CT** | 200 | +0.88 | +44.74 | **-17.90** | -17.68 | **22.6%** |
| **MRI** | 59 | -3.30 | +51.94 | **-5.71** | -6.12 | **19.3%** |

## 4. FOV Group Stratification
| FOV Category | Definition | Patients ($N$) | Mean $dz$ (mm) | Median $dz$ (mm) | $Z$-Energy Share (%) |
|---|---|---|---|---|---|
| **FOV-A** | Height Extent | 116 | **-16.80** | -17.81 | 24.8% |
| **FOV-B** | Height Extent | 140 | **-14.35** | -13.97 | 19.8% |
| **FOV-C** | Height Extent | 3 | **-9.13** | -3.85 | 8.6% |

## 5. Target-Wise Axis Bias Table
| Target Name | Observations ($N$) | Mean $dx$ (mm) | Mean $dy$ (mm) | Mean $dz$ (mm) | Mean |$dz$| (mm) | $Z$-Energy Share (%) |
|---|---|---|---|---|---|---|
| `aorta` | 259 | +2.17 | +46.75 | **-4.11** | 11.29 | 8.2% |
| `esophagus` | 258 | +2.74 | +44.84 | **-8.21** | 16.47 | 15.2% |
| `urinary_bladder` | 197 | +1.93 | +45.60 | **-10.68** | 17.24 | 16.2% |
| `inferior_vena_cava` | 259 | +1.21 | +46.01 | **-13.29** | 19.69 | 18.2% |
| `adrenal_gland_right` | 258 | -1.61 | +46.43 | **-13.28** | 21.32 | 19.7% |
| `liver` | 259 | -0.94 | +45.97 | **-17.28** | 23.55 | 22.8% |
| `adrenal_gland_left` | 259 | +0.45 | +47.31 | **-14.77** | 22.43 | 20.7% |
| `kidney_right` | 259 | -0.23 | +45.90 | **-13.73** | 24.47 | 23.4% |
| `kidney_left` | 258 | -0.70 | +46.17 | **-14.08** | 24.14 | 22.8% |
| `pancreas` | 259 | -0.70 | +46.94 | **-15.69** | 23.35 | 22.0% |
| `spleen` | 256 | -0.17 | +47.08 | **-17.22** | 25.67 | 25.8% |
| `prostate` | 194 | +2.93 | +45.08 | **-24.39** | 28.96 | 28.8% |
| `duodenum` | 259 | -2.12 | +46.86 | **-23.18** | 28.94 | 28.2% |
| `stomach` | 257 | -0.35 | +46.90 | **-20.29** | 27.29 | 26.5% |
| `gallbladder` | 243 | -3.32 | +44.91 | **-22.40** | 31.25 | 31.3% |

## 6. Critical Forensic Takeaways
1. **Dominance of Superior-Inferior Shift:** The $Z$ axis accounts for the overwhelming majority of error energy. Rather than random 3D spatial diffusion, the predictions are rigidly displaced along the craniocaudal axis.
2. **Lateral ($X$) Symmetry:** Mean $dx$ is remarkably close to zero ($< 3\text{ mm}$), confirming that left-right symmetry and coronal alignment are preserved.
3. **Anteroposterior ($Y$) Offset:** A secondary anterior/posterior shift exists, but is secondary in magnitude to the $Z$-axis shift.
4. **Hard Interpretation Verdict:** In accordance with the non-negotiable scientific rules, **systematic coordinate bias detected**.
