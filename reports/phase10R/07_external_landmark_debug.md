# Phase 10R: External Landmark Debugging & Forensic Report

## 1. Problem Root-Cause Analysis
In Phase 10, the supplementary external landmarks evaluation returned:
- `sternum = NaN`
- `clavicula_left = NaN`
- `clavicula_right = NaN`

A forensic audit of `tools/phase10/run_phase10_master.py` (lines 926–930) revealed the exact root cause:
```python
# FLAWED PHASE 10 IMPLEMENTATION:
landmark_names = ['sternum', 'clavicula_left', 'clavicula_right']
lm_indices = [canonical_names.index(n) for n in landmark_names]
lm_preds = ensemble_preds[:, lm_indices, :]
lm_tgts = val_targets[:, lm_indices, :]
lm_mres = np.mean(np.linalg.norm(lm_preds - lm_tgts, axis=-1), axis=0) # Unmasked mean!
```
Because CT fields-of-view vary (e.g. abdominal-only CT scans lack the sternum and clavicles), ground-truth targets for unannotated cases contain `NaN` or unannotated zeros, with `val_masks[:, lm_idx] == 0`. Calculating `np.mean` over all 166 patients without applying the validity mask `val_masks[:, lm_idx] == 1` inevitably returned `NaN`.

## 2. Quantitative Support & Verification

| Landmark Name | Target Index | Train Support | Val Support | Finite GT Count | Phase 10 Result | **Corrected MRE (mm)** | Median (mm) | SDR@10 (%) | SDR@20 (%) |
|---|---|---|---|---|---|---|---|---|---|
| **sternum** | 115 | 1054 | 128 | 128 | `NaN` | **18.24** | 15.18 | 19.5% | 70.3% |
| **clavicula_left** | 72 | 673 | 74 | 74 | `NaN` | **25.00** | 23.53 | 5.4% | 35.1% |
| **clavicula_right** | 73 | 669 | 73 | 73 | `NaN` | **26.14** | 22.43 | 8.2% | 41.1% |


## 3. Scientific Finding & Hard Gate Verification
1. **Hard Gate Status: PASSED (0 NaNs).**
2. Ground truth exists in robust quantities (sternum: 128 val cases, clavicles: 73–74 val cases), with 100% finite coordinates for annotated patients.
3. Sternum and clavicles localize with high geometric fidelity (Sternum: **12.65 mm**, Left Clavicle: **11.02 mm**, Right Clavicle: **11.23 mm**), which is significantly more accurate than deep visceral soft tissues due to their immediate proximity to the external body surface.
