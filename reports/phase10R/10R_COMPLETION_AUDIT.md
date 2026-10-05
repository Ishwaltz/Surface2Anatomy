# Phase 10R: Experimental Completion Audit

> [!IMPORTANT]
> **REPRODUCIBILITY AUDIT CONFIRMATION**
> All primary comparisons use an identical 65-epoch optimization budget, identical AdamW + CosineAnnealingLR schedules, and identical Dataset V3 frozen splits.
> All legacy 20-epoch runs from Phase 10 are classified as `LEGACY_UNMATCHED_OPTIMIZATION` and excluded from publication.

| experiment | required_epochs | actual_epochs | seed | status | checkpoint | usable_for_publication_yes_no | reason |
|---|---|---|---|---|---|---|---|
| **C0_Population_Atlas** | 0 | 0 | 42 | `COMPLETE` | `C0_Population_Atlas.pt` | **YES** | Deterministic zero-parameter population mean baseline. |
| **C1_Linear_Ridge** | 0 | 0 | 42 | `COMPLETE` | `C1_Linear_Ridge.pt` | **YES** | Surface point cloud Ridge regression baseline (alpha=100). |
| **C5_Internal_SSM_PCA** | 0 | 0 | 42 | `COMPLETE` | `C5_Internal_SSM_PCA.pt` | **YES** | Surface PCA (64 components) + Ridge regressor baseline. |
| **C4_Proposed_seed42** | 65 | 65 | 42 | `COMPLETE` | `C4_Proposed_seed42.pt` | **YES** | Canonical 65-epoch full budget run; verified checkpoint hash. |
| **C4_Proposed_seed43** | 65 | 65 | 43 | `COMPLETE` | `C4_Proposed_seed43.pt` | **YES** | Canonical 65-epoch full budget run; verified checkpoint hash. |
| **C4_Proposed_seed44** | 65 | 65 | 44 | `COMPLETE` | `C4_Proposed_seed44.pt` | **YES** | Canonical 65-epoch full budget run; verified checkpoint hash. |
| **C2_PointNet2_seed42** | 65 | 65 | 42 | `COMPLETE` | `C2_PointNet2_seed42.pt` | **YES** | Matched 65-epoch PointNet++ direct regressor; verified checkpoint. |
| **C2_PointNet2_seed43** | 65 | 65 | 43 | `COMPLETE` | `C2_PointNet2_seed43.pt` | **YES** | Matched 65-epoch PointNet++ direct regressor; verified checkpoint. |
| **C2_PointNet2_seed44** | 65 | 65 | 44 | `COMPLETE` | `C2_PointNet2_seed44.pt` | **YES** | Matched 65-epoch PointNet++ direct regressor; verified checkpoint. |
| **C3_DGCNN_seed42** | 65 | 65 | 42 | `COMPLETE` | `C3_DGCNN_seed42.pt` | **YES** | Matched 65-epoch PointNet++ + DGCNN Target Decoder; verified checkpoint. |
| **C3_DGCNN_seed43** | 65 | 65 | 43 | `COMPLETE` | `C3_DGCNN_seed43.pt` | **YES** | Matched 65-epoch PointNet++ + DGCNN Target Decoder; verified checkpoint. |
| **C3_DGCNN_seed44** | 65 | 65 | 44 | `COMPLETE` | `C3_DGCNN_seed44.pt` | **YES** | Matched 65-epoch PointNet++ + DGCNN Target Decoder; verified checkpoint. |
| **Domain_V2only_seed42** | 65 | 65 | 42 | `COMPLETE` | `Domain_V2only_seed42.pt` | **YES** | Matched 65-epoch V2-only training run. |
| **Domain_V2only_seed43** | 65 | 65 | 43 | `COMPLETE` | `Domain_V2only_seed43.pt` | **YES** | Matched 65-epoch V2-only training run. |
| **Domain_V2only_seed44** | 65 | 65 | 44 | `COMPLETE` | `Domain_V2only_seed44.pt` | **YES** | Matched 65-epoch V2-only training run. |
| **Domain_TSonly_seed42** | 65 | 65 | 42 | `COMPLETE` | `Domain_TSonly_seed42.pt` | **YES** | Matched 65-epoch TotalSegmentator-only training run. |
| **Domain_TSonly_seed43** | 65 | 65 | 43 | `COMPLETE` | `Domain_TSonly_seed43.pt` | **YES** | Matched 65-epoch TotalSegmentator-only training run. |
| **Domain_TSonly_seed44** | 65 | 65 | 44 | `COMPLETE` | `Domain_TSonly_seed44.pt` | **YES** | Matched 65-epoch TotalSegmentator-only training run. |
| **D2_SingleScale_SA3** | 65 | 65 | 42 | `COMPLETE` | `D2_SingleScale_SA3.pt` | **YES** | Matched 65-epoch coarse SA3 memory ablation. |
| **D3_NoAtlasPrior** | 65 | 65 | 42 | `COMPLETE` | `D3_NoAtlasPrior.pt` | **YES** | Matched 65-epoch no-atlas prior ablation. |
| **D4_GlobalTokenOnly** | 65 | 65 | 42 | `COMPLETE` | `D4_GlobalTokenOnly.pt` | **YES** | Matched 65-epoch global token only ablation. |
| **D5_SelfAttnOnly** | 65 | 65 | 42 | `COMPLETE` | `D5_SelfAttnOnly.pt` | **YES** | Matched 65-epoch self-attention only (no cross-attn) ablation. |
| **D6_SelfAndCrossAttn** | 65 | 65 | 42 | `COMPLETE` | `D6_SelfAndCrossAttn.pt` | **YES** | Matched 65-epoch self-attn + cross-attn variant. |
| **D7_Layer2** | 65 | 65 | 42 | `COMPLETE` | `D7_Layer2.pt` | **YES** | Matched 65-epoch 2-decoder-layer depth ablation. |
| **D7_Layer6** | 65 | 65 | 42 | `COMPLETE` | `D7_Layer6.pt` | **YES** | Matched 65-epoch 6-decoder-layer depth ablation. |
| **D9_QueryPermutation** | 0 | 0 | 42 | `COMPLETE` | `D9_QueryPermutation_Diagnostic.pt` | **YES** | Inference diagnostic (target query index permutation). |
| **D10_TokenShuffle** | 0 | 0 | 42 | `COMPLETE` | `D10_TokenShuffle_Diagnostic.pt` | **YES** | Inference diagnostic (patient token shuffle). |
| **E1_Points_1024** | 65 | 65 | 42 | `COMPLETE` | `E1_Points_1024.pt` | **YES** | Matched 65-epoch input subsampling ablation (1024 points). |
| **E1_Points_2048** | 65 | 65 | 42 | `COMPLETE` | `E1_Points_2048.pt` | **YES** | Matched 65-epoch input subsampling ablation (2048 points). |
| **E1_Points_8192** | 65 | 65 | 42 | `COMPLETE` | `E1_Points_8192.pt` | **YES** | Matched 65-epoch input subsampling ablation (8192 points). |
| **E2_XYZ_Plus_Normals** | 65 | 65 | 42 | `COMPLETE` | `E2_XYZ_Plus_Normals.pt` | **YES** | Matched 65-epoch surface normals ablation. |
| **Legacy_Phase10_20epoch_runs** | 65 | 20 | 42 | `LEGACY_UNMATCHED_OPTIMIZATION` | `Various (experiments/phase10/)` | **NO** | Retired legacy 20-epoch runs; excluded from publication. |
