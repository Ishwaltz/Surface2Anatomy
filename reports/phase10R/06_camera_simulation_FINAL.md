# Phase 10R: Physically Motivated Simulated Optical/Depth Sensing (Final)

> [!IMPORTANT]
> **WORDING GOVERNANCE:** Characterized strictly as *"simulated optical/depth sensing"*, NOT *"real RGB-D validation"*.
> Latency is measured at **88.7 ms (11.3 FPS)** on NVIDIA RTX 4070 Ti SUPER, characterized as *"interactive / near-real-time"*.

## 1. Multi-Camera Rig & Partial Visibility Benchmark

| Simulated Sensing Scenario | Visibility / Artifact Condition | Macro MRE (mm) | Median (mm) | P90 (mm) | SDR@20 (%) |
|---|---|---|---|---|---|
| **Full 360-deg Surface (Reference)** | Realistic ray-tracing / surface occlusion | **24.60** | 19.01 | 42.12 | 53.4% |
| **3-Camera SGRT Ceiling Array** | Realistic ray-tracing / surface occlusion | **29.08** | 23.59 | 49.62 | 38.9% |
| **2-Camera Frontal Oblique Pair** | Realistic ray-tracing / surface occlusion | **30.55** | 23.23 | 56.02 | 40.1% |
| **1-Camera Frontal View** | Realistic ray-tracing / surface occlusion | **34.38** | 27.46 | 60.19 | 28.6% |
| **Point Dropout 10%** | Realistic ray-tracing / surface occlusion | **26.31** | 20.64 | 47.28 | 47.8% |
| **Point Dropout 25%** | Realistic ray-tracing / surface occlusion | **27.49** | 21.46 | 49.21 | 45.4% |
| **Point Dropout 50%** | Realistic ray-tracing / surface occlusion | **29.44** | 23.12 | 54.81 | 41.1% |
| **Depth Noise sigma = 1 mm** | Realistic ray-tracing / surface occlusion | **24.56** | 19.46 | 41.84 | 52.1% |
| **Depth Noise sigma = 3 mm** | Realistic ray-tracing / surface occlusion | **25.40** | 19.59 | 44.26 | 51.2% |
| **Depth Noise sigma = 5 mm** | Realistic ray-tracing / surface occlusion | **26.62** | 20.44 | 47.60 | 48.6% |
| **Scale Error -5%** | Realistic ray-tracing / surface occlusion | **26.85** | 20.87 | 46.24 | 47.1% |
| **Scale Error -2%** | Realistic ray-tracing / surface occlusion | **25.15** | 19.29 | 43.81 | 52.3% |
| **Scale Error +2%** | Realistic ray-tracing / surface occlusion | **24.71** | 19.43 | 42.69 | 52.2% |
| **Scale Error +5%** | Realistic ray-tracing / surface occlusion | **25.96** | 20.38 | 43.37 | 48.7% |

## 2. Latency & Throughput Benchmark

- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM)
- **Batch Size 1 Latency:** **88.7 ms**
- **Throughput:** **11.3 FPS** (interactive / near-real-time)
