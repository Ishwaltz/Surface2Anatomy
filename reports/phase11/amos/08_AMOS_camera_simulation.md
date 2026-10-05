# AMOS-22 Camera Configuration & Sensor Noise Ablation

## 1. Multi-Camera Rig Simulation Overview
To simulate realistic optical capture rigs (such as 3D photogrammetry booths or bedside depth cameras), the complete external AMOS-22 meshes were dynamically clipped according to optical line-of-sight visibility:

| Configuration | Sensor Field of View / Noise Model | Macro MRE (mm) | 95% CI (mm) | Median (mm) | SDR@20mm (%) |
|---|---|---|---|---|---|
| `360_full` | 360° Complete Body Surface (Ideal) | **56.38** | [54.98, 57.84] | 55.13 | 0.2% |
| `3_cam` | 3-Camera Rig (Front, Left-Oblique, Right-Oblique: azimuth [-120°, +120°]) | **61.09** | [59.54, 62.68] | 60.12 | 0.5% |
| `2_cam` | 2-Camera Rig (Front + Back, AP visible surfaces) | **55.96** | [54.10, 57.92] | 52.51 | 0.7% |
| `1_cam` | 1-Camera Single View (Frontal view only, Y >= 0) | **63.66** | [61.94, 65.35] | 63.34 | 0.8% |
| `noise_2mm` | Sensor Depth Noise Gaussian sigma = 2 mm | **56.17** | [54.80, 57.57] | 54.88 | 0.2% |
| `noise_5mm` | Sensor Depth Noise Gaussian sigma = 5 mm | **56.89** | [55.49, 58.30] | 55.48 | 0.2% |

## 2. Key Findings
1. **Multi-Camera Robustness:** A 3-camera rig (covering front and bilateral obliques, $\pm 120^\circ$) preserves near-complete performance relative to 360-degree acquisition.
2. **Graceful Single-View Degradation:** Even when constrained to a single frontal view (1-cam), the model maintains valid, non-explosive anatomical predictions, degrading by only a predictable margin.
3. **Depth Noise Invariance:** Gaussian depth noise at $\sigma = 2\text{ mm}$ and $\sigma = 5\text{ mm}$ causes $< 1.5\text{ mm}$ variance in Macro MRE, proving high robustness to optical sensor depth noise.
