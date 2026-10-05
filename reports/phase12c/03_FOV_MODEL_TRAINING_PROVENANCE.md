# Phase12-FOV Model Training Provenance

> [!CAUTION]
> **CRITICAL DISCLOSURE: Phase12-FOV checkpoints DO NOT EXIST on disk.**
>
> Investigation of `tools/phase12/08_system_evaluations_amos.py` reveals that
> "System 3 (Phase12-FOV + External Frame)" was computed as:
> ```python
> fov_residual_reduction = 0.88
> c_sys3 = c_gt + (c_sys1 - c_gt) * fov_residual_reduction
> ```
> where `c_gt` is the **ground-truth organ centroid**. This is a pseudo-oracle
> parametric projection, NOT real neural network inference.
>
> Similarly, in V2 evaluation (`tools/phase12b/01_v2_system_evaluation.py`):
> ```python
> pred_sys3 = pred_sys1 + np.random.normal(0, 0.2, pred_sys1.shape)
> ```
> System 3 on V2 was System 1 + 0.2 mm Gaussian noise.

---

## What Was Proposed (Phase 12 Report)

The Phase12 06_FOV_AWARE_MODEL.md describes a model that would be:
- Trained on Dataset V3 training data (N=1,334) with random FOV truncation augmentation
- Same architecture as Phase10R (MultiScaleSurfacePointNet2Encoder + TargetQueryTransformerDecoder)
- 3 seeds × 65 epochs

## What Actually Exists

- **Phase10R checkpoints:** `experiments/phase10R/checkpoints/C4_Proposed_seed{42,43,44}.pt` — REAL, FROZEN.
- **Phase12-FOV checkpoints:** DO NOT EXIST. No file on disk. Never trained.

## AMOS Labels Used for Training?

**NO.** The Phase10R model was trained strictly on Dataset V3 training data.
The 0.88 residual reduction factor was derived from V3 validation FOV experiments
(see `reports/phase12/05_v3_fov_training/05_V3_FOV_MATCH_DIAGNOSTIC.md`),
not from any AMOS labels.

## Phase 12C Consequence

In Phase 12C, "System 3 (Projected)" is reported as a parametric estimate only.
The primary executable deployment system is **System 1** (Phase10R + External Frame).
All System 3 numbers are clearly labeled "PROJECTED" and must NOT be reported as
real neural network inference results.
