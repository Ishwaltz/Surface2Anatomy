# PHASE 9B FROZEN MODEL ARCHITECTURE SPECIFICATION

## 1. Frozen Target-Query Architecture (Phase 3 Baseline)
- **Surface Encoder:** `MultiScaleSurfacePointNet2Encoder` (PointNet++ Set Abstraction 1024 -> 256 -> 64 -> 1)
- **Target Queries:** 117 learned anatomical queries initialized with canonical training atlas coordinates
- **Cross-Attention Decoder:** 4-Layer Transformer Cross-Attention (`d_model = 256`, `nhead = 8`, `dropout = 0.05`)
- **Memory Tokens:** 320 multi-scale surface tokens (256 mid tokens + 64 coarse tokens) + 3D positional encoding
- **Coordinate Head:** Atlas-residual MLP predicting physical offset $\Delta \mathbf{p}$
- **Parameter Count:** ~1.85 Million Parameters

## 2. Frozen Experimental Conditions
- **Input Representation:** 4096 XYZ surface points ONLY (No normals, no 8192 points, no metadata, no source ID)
- **Coordinate Normalization:** Single training-wide scalar $S_{global} = 500.0\text{ mm}$
- **Loss Objective:** Masked physical radial MRE loss on 104 benchmark primary targets
- **Checkpoint Selection:** Best Validation Macro Target MRE on 104 benchmark primary targets
- **Optimizer & Schedule:** AdamW (`lr_enc = 2e-4`, `lr_dec = 5e-4`, `weight_decay = 1e-4`), Cosine Annealing 65 epochs, Batch size 16
---
