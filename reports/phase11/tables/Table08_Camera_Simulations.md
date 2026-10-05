# Table 8: AMOS Simulated Camera-Count & Optical Noise Results

| Optical Configuration | Rig Description | Macro MRE (mm) | 95% Bootstrap CI (mm) | Median (mm) | P90 (mm) | SDR@20 (%) |
|---|---|---|---|---|---|---|
| `360_full` | 360° Complete Body Surface (Ideal) | **56.38** | [54.98, 57.84] | 55.13 | 77.46 | 0.2% |
| `3_cam` | 3-Camera Rig (Front, Left-Oblique, Right-Oblique: azimuth [-120°, +120°]) | **61.09** | [59.54, 62.68] | 60.12 | 83.20 | 0.5% |
| `2_cam` | 2-Camera Rig (Front + Back, AP visible surfaces) | **55.96** | [54.10, 57.92] | 52.51 | 81.34 | 0.7% |
| `1_cam` | 1-Camera Single View (Frontal view only, Y >= 0) | **63.66** | [61.94, 65.35] | 63.34 | 87.20 | 0.8% |
| `noise_2mm` | Sensor Depth Noise Gaussian sigma = 2 mm | **56.17** | [54.80, 57.57] | 54.88 | 76.70 | 0.2% |
| `noise_5mm` | Sensor Depth Noise Gaussian sigma = 5 mm | **56.89** | [55.49, 58.30] | 55.48 | 77.65 | 0.2% |
