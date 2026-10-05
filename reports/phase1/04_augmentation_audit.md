# Data Augmentation Audit Report

## Executive Summary
During active training (`train_evidential.py`, `train_same.py`, `train_benchmarks.py`), dataset augmentation is enabled via `augment=True` in `PointCloudOrganDataset`.

## Augmentation Matrix

| Augmentation Name | Applied to Surface? | Applied to Targets? | Mathematically Consistent? | Code Location | Severity if Wrong |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Gaussian Point Jitter** | Yes (`pts + noise * 0.005`) | No | **YES** (Valid feature noise) | `sharon/dataset.py:62-64` | **LOW** |
| **3D Spatial Rotation** | No (Unused parameter) | No | **N/A** (Dead parameter) | `sharon/dataset.py:22,45` | **MEDIUM** (Misleading interface) |
| **Spatial Translation** | No | No | **N/A** | None | None |
| **Isotropic/Anisotropic Scaling** | No | No | **N/A** | None | None |
| **Axis Reflection / Flipping** | No | No | **N/A** | None | None |

## Key Findings
1. **No Spatial Coordinate Transformations in Augmentation**: The only active augmentation is minor independent Gaussian point coordinate jitter (`sigma = 0.005` in normalized cube `[-1, 1]`) applied to surface points.
2. **Targets are Not Jittered**: Because Gaussian jitter represents surface sensor noise rather than patient translation, keeping target organ centroids fixed is mathematically sound.
3. **Dead Argument `rotation_aug`**: `PointCloudOrganDataset.__init__` accepts `rotation_aug: bool = False`, but `__getitem__` never executes any rotation logic. If a developer enabled `rotation_aug=True` expecting rotation, nothing would happen.
