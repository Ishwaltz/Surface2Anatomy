# Phase 10R: Corrected External Landmark Evaluation (Final)

> [!IMPORTANT]
> **ZERO NaNs AUDIT CONFIRMATION**
> In Phase 10, external landmark evaluation suffered from an unmasked averaging bug over unsegmented slices, producing NaNs.
> In Phase 10R, ground-truth masks are strictly applied, verifying 100% finite values (0 NaNs).

| Target Landmark | Target Index | Train Support | Validation Support | Corrected Val MRE (mm) | Median (mm) | SDR@20 (%) | Phase 10 Buggy Result |
|---|---|---|---|---|---|---|---|
| **sternum** | 115 | 1054 | 128 | **18.24** | 15.18 | 70.3% | `NaN` (FIXED) |
| **clavicula_left** | 72 | 673 | 74 | **25.00** | 23.53 | 35.1% | `NaN` (FIXED) |
| **clavicula_right** | 73 | 669 | 73 | **26.14** | 22.43 | 41.1% | `NaN` (FIXED) |
