# Storage Preflight Audit & Capacity Budget (Phase 8A)

**Audit Date:** 2026-09-06  
**Auditor:** Senior Medical-Imaging Data Engineer  
**Filesystem:** `/dev/nvme0n1p2` (Root and Repository Mount)  

---

## 1. Filesystem Capacity Status

| Mount Point | Filesystem | Total Size | Used Space | Available Space | Use % | Preflight Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `/` (Repository Mount) | `/dev/nvme0n1p2` | **1.8 TB** | 1.2 TB | **545 GB** | 69% | **SUFFICIENT** |
| `/run/media/.../14e107c3...` | `/dev/sda2` (Secondary) | 1.8 TB | 76 GB | **1.7 TB** | 5% | **EXPEDIENT BACKUP** |

---

## 2. Dataset V3 Storage Budget Breakdown

Estimates are calculated based on actual content-length headers and voxel volume dimensions:

| Component | Compressed Size (GB) | Extracted NIfTI (GB) | Point Clouds & Normals (GB) | QC Figures & Artifacts (GB) | Total Allocated (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TotalSegmentator** (1,228 cases) | 21.8 GB | ~45.0 GB | 1.8 GB | 0.2 GB | **68.8 GB** |
| **DAP Atlas** (533 cases, AutoPET CT + 142 masks) | 36.0 GB | ~85.0 GB | 0.9 GB | 0.2 GB | **122.1 GB** |
| **Dataset V2 Re-Processing** (440 cases) | 0.0 GB (Local) | 0.0 GB (Local) | 0.7 GB | 0.1 GB | **0.8 GB** |
| **Temporary Extraction & Scratch** | — | ~30.0 GB | — | — | **30.0 GB** |
| **Dataset V3 Final Artifacts (`pointclouds_v3.pt` + cases NPZ)** | — | — | ~3.5 GB | — | **3.5 GB** |
| **TOTAL ESTIMATED DEMAND** | **57.8 GB** | **160.0 GB** | **6.9 GB** | **0.5 GB** | **225.2 GB** |

---

## 3. Storage Safety Margin

$$\text{Available Free Space} = 545.0\text{ GB}$$
$$\text{Total Estimated Demand} = 225.2\text{ GB}$$
$$\text{Net Safety Buffer Remaining} = 545.0 - 225.2 = 319.8\text{ GB (58.7\% Headroom)}$$

### Safety Invariants & Policies:
1. **Never delete raw archives automatically**: Keep compressed `.zip` files in `data_external/<source>/raw/` for provenance verification and checksum matching.
2. **Streaming Extraction**: Unpack archives directly into structured target folders (`data_external/<source>/extracted/`) to minimize duplicate scratch buffering.
3. **Chunked Case Storage**: Save extracted point clouds as individual lightweight `.npz` files (~3 MB per case) to allow memory-efficient incremental training and robust resumption.
4. **Disk Guard**: All download and preprocessing scripts will monitor available space and halt execution if free disk falls below 50.0 GB.

---

## 4. Preflight Verdict

### `STORAGE_PREFLIGHT_PASSED`

The system has **545 GB** of available NVMe storage, well exceeding the required **225.2 GB** budget with over **319 GB** of reserve safety headroom. Acquisition may proceed safely.
