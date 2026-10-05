# Table 7: Virtual Camera Sensing & Sensor Perturbation Robustness

| Sensing Scenario / Perturbation | Macro MRE (mm) | Median Error (mm) | P90 Error (mm) | SDR@10 (%) | SDR@20 (%) |
|---|---|---|---|---|---|
| **Full 360-deg Surface (Reference)** | **24.60** | 19.01 | 42.12 | 15.0% | 53.4% |
| **3-Camera SGRT Ceiling Array** | **29.08** | 23.59 | 49.62 | 8.9% | 38.9% |
| **2-Camera Frontal Oblique Pair** | **30.55** | 23.23 | 56.02 | 9.5% | 40.1% |
| **1-Camera Frontal View** | **34.38** | 27.46 | 60.19 | 5.6% | 28.6% |
| **Point Dropout 10%** | **26.31** | 20.64 | 47.28 | 13.2% | 47.8% |
| **Point Dropout 25%** | **27.49** | 21.46 | 49.21 | 12.2% | 45.4% |
| **Point Dropout 50%** | **29.44** | 23.12 | 54.81 | 10.0% | 41.1% |
| **Depth Noise sigma = 1 mm** | **24.56** | 19.46 | 41.84 | 15.1% | 52.1% |
| **Depth Noise sigma = 3 mm** | **25.40** | 19.59 | 44.26 | 14.4% | 51.2% |
| **Depth Noise sigma = 5 mm** | **26.62** | 20.44 | 47.60 | 12.8% | 48.6% |
| **Scale Error -5%** | **26.85** | 20.87 | 46.24 | 12.3% | 47.1% |
| **Scale Error -2%** | **25.15** | 19.29 | 43.81 | 15.2% | 52.3% |
| **Scale Error +2%** | **24.71** | 19.43 | 42.69 | 14.7% | 52.2% |
| **Scale Error +5%** | **25.96** | 20.38 | 43.37 | 13.1% | 48.7% |
