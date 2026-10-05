# Real-Sensor End-to-End Computational Latency Benchmark

## 1. Executive Summary & Throughput
- **Hardware Environment:** NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM)
- **Benchmark Cohort:** 200 sequential real iPhone TrueDepth frames
- **Total End-to-End Latency (Median):** **95.15 ms**
- **Total End-to-End Latency (P90 / P95):** **96.98 ms / 97.91 ms**
- **Neural Forward Pass Alone (Median):** **93.67 ms**
- **Effective Real-Sensor Throughput:** **10.5 FPS**

## 2. Granular Stage-by-Stage Latency Breakdown

| Pipeline Stage | Stage Description | Median (ms) | Mean (ms) | P90 (ms) | P95 (ms) | Fraction of Total (%) |
|---|---|---|---|---|---|---|
| `depth_load_ms` | A: Depth I/O (Disk to RAM) | 0.71 | 0.77 | 0.77 | 0.82 | 0.8% |
| `backproject_ms` | B: 3D Camera Back-Projection | 0.27 | 0.28 | 0.31 | 0.32 | 0.3% |
| `filter_ms` | C: Geometric Outlier Rejection | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| `sample_4096_ms` | D: 4096-Point Sampling & Centering | 0.35 | 0.35 | 0.37 | 0.38 | 0.4% |
| `cpu_to_gpu_ms` | E: Host to Device Transfer (PCIe) | 0.06 | 0.07 | 0.08 | 0.09 | 0.1% |
| `model_forward_ms` | F: Neural Forward Pass (PointNet++ + Transformer) | 93.67 | 93.88 | 95.35 | 96.24 | 98.4% |
| `denorm_ms` | G: Metric Denormalization & Output | 0.07 | 0.07 | 0.08 | 0.09 | 0.1% |
| `total_e2e_ms` | **Total End-to-End Latency** | 95.15 | 95.41 | 96.98 | 97.91 | 100.0% |

## 3. Methodological Note on Real-Time Claims
- Earlier preliminary claims stated unmeasured '40 FPS / 25 ms' inference.
- Rigorous real-sensor benchmarking on real hardware verifies that total pipeline latency is **95.2 ms (~10.5 FPS)**, with the deep learning forward pass accounting for **93.7 ms** and CPU point cloud preprocessing accounting for the remaining time.
