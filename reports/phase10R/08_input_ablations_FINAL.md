# Phase 10R: Input Point Resolution & Feature Ablations (Final)

All variants trained for matched 65 epochs.

| Input Representation | Description | Macro MRE (mm) | Median (mm) | P90 (mm) | SDR@20 (%) |
|---|---|---|---|---|---|
| **1,024 XYZ Points** | Lightweight subsampling (1k points) | **25.15** | 19.47 | 42.41 | 51.7% |
| **2,048 XYZ Points** | Moderate subsampling (2k points) | **24.33** | 18.88 | 41.72 | 53.7% |
| **4,096 XYZ Points (Standard)** | Canonical baseline resolution (4k points) | **25.09** | 19.47 | 43.61 | 51.7% |
| **8,192 XYZ Points** | Dense surface representation (8k points) | **24.37** | 18.32 | 42.34 | 56.0% |
| **4,096 XYZ + Surface Normals** | 6D input (3D coords + estimated outward surface normals) | **25.46** | 19.73 | 42.87 | 51.1% |
