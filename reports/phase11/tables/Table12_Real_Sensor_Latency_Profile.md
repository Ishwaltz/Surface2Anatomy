# Table 12: Real-Sensor Computational Latency and Throughput Breakdown

| Pipeline Stage | Operational Description | Median Latency (ms) | Mean Latency (ms) | P90 Latency (ms) | P95 Latency (ms) | Percentage of Total (%) |
|---|---|---|---|---|---|---|
| **A: Depth I/O** | Loading 16-bit depth map from storage | 0.28 | 0.32 | 0.45 | 0.52 | 0.3% |
| **B: Back-Projection** | Unprojecting pixels to metric 3D point cloud | 0.62 | 0.68 | 0.85 | 0.94 | 0.7% |
| **C: Outlier Filter** | Statistical radius outlier removal | 0.35 | 0.39 | 0.51 | 0.58 | 0.4% |
| **D: Sampling & Centering** | 4,096 uniform sampling & scale normalization | 0.18 | 0.21 | 0.28 | 0.32 | 0.2% |
| **E: Host to Device** | PCIe memory transfer to GPU | 0.05 | 0.06 | 0.08 | 0.09 | 0.1% |
| **F: Neural Forward Pass** | PointNet++ backbone + Target Transformer decoder | **93.67** | **94.20** | **95.12** | **95.80** | **98.4%** |
| **G: Denormalization** | Rescaling to patient coordinates | 0.04 | 0.05 | 0.06 | 0.07 | 0.1% |
| **Total End-to-End** | **Full Capture-to-Centroid Pipeline** | **95.15** | **95.91** | **96.98** | **97.91** | **100.0%** |

- **Effective Real-Time Throughput:** **10.51 FPS** on NVIDIA GeForce RTX 4070 Ti SUPER.
