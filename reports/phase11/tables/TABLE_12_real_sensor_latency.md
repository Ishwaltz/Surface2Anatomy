# Table 12: Real-Sensor End-to-End Runtime Breakdown

| Pipeline Component | Sub-operation | Median Latency (ms) | 90th Percentile (ms) | 95th Percentile (ms) |
|---|---|---|---|---|
| A | Depth I/O (Disk to RAM) | 0.71 | 0.77 | 0.82 |
| B | 3D Camera Back-Projection | 0.27 | 0.31 | 0.32 |
| C | Geometric Outlier Rejection | 0.00 | 0.00 | 0.00 |
| D | 4096-Point Sampling & Centering | 0.35 | 0.37 | 0.38 |
| E | Host to Device Transfer (PCIe) | 0.06 | 0.08 | 0.09 |
| F | Neural Forward Pass (PointNet++ + Transformer) | 93.67 | 95.35 | 96.24 |
| G | Metric Denormalization & Output | 0.07 | 0.08 | 0.09 |
| **Total End-to-End Latency** | **Total End-to-End Latency** | 95.15 | 96.98 | 97.91 |
