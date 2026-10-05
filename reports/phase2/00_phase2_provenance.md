# Phase 2 Provenance & Dataset V2 Verification

## 1. Environment & Hardware
- **Operating System**: Linux 7.0.0-29-generic x86_64
- **Python Version**: 3.11.15
- **PyTorch Version**: 2.13.0+cu130
- **CUDA Available**: True
- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER (16.72 GB VRAM)
- **Random Seeds Designated**: `42`, `43`, `44`

## 2. Dataset V2 Cryptographic Verification
- **Dataset Path**: `sharon/dataset_v2/pointclouds_v2.pt`
- **Dataset SHA-256**: `a816c7f8fca91571c84325cd5d7fa272bd0f5057ce4bd234234f67c0ffd2e00a`
- **Splits Path**: `sharon/dataset_v2/splits_v2.json`
- **Splits SHA-256**: `0fdf03288b0885f0b5c0282716e49c43335e84e22ea7c75e23d0fca03495e8d3`
- **Total Unique Patients**: 440
- **Training Split**: 352 cases (80.0%)
- **Validation Split**: 44 cases (10.0%)
- **Held-Out Test Split**: 44 cases (10.0%, **FROZEN - UNTOUCHED**)

## 3. Geometric & Representation Invariants
- **Global Metric Scale**: $S_{\text{global}} = 500.0\text{ mm}$
- **Translation Normalization**: Patient-specific surface-derived center $c_{\text{surface}}$ removed; identical center applied to targets.
- **Coordinate Bounds**: Model space coordinates strictly bounded in $[-1.0, 1.0]^3$.
- **Reconstruction Fidelity**: Maximum round-trip error $= 3.49622787e-05\text{ mm} < 10^{-3}\text{ mm}$.
- **Primary Benchmark Integrity**: 4 synthetic female pelvic organs strictly masked out (`primary_valid_mask = 0`).
