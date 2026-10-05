import numpy as np

def compute_all_metrics(preds_mm, targets_mm, masks, target_indices=None):
    """
    Computes all standard Phase 2 physical metrics in millimeters.
    
    Args:
      preds_mm: (N_val, K, 3) in physical mm
      targets_mm: (N_val, K, 3) in physical mm
      masks: (N_val, K) binary mask (1 for valid, 0 for absent/excluded)
      target_indices: optional list of target indices to evaluate (e.g. primary 107 targets)
      
    Returns dict of aggregate metrics and per-target errors.
    """
    N, K, _ = preds_mm.shape
    if target_indices is not None:
        eval_mask = np.zeros_like(masks, dtype=bool)
        eval_mask[:, target_indices] = (masks[:, target_indices] > 0.5)
    else:
        eval_mask = (masks > 0.5)

    diff = preds_mm - targets_mm
    dist_matrix = np.linalg.norm(diff, axis=-1) # (N, K)

    # 1. Micro MRE: all valid observations contribute equally
    valid_dists = dist_matrix[eval_mask]
    if len(valid_dists) == 0:
        return {"error": "No valid observations"}

    micro_mre = float(valid_dists.mean())
    median_err = float(np.median(valid_dists))
    p75_err = float(np.percentile(valid_dists, 75))
    p90_err = float(np.percentile(valid_dists, 90))
    p95_err = float(np.percentile(valid_dists, 95))
    max_err = float(valid_dists.max())

    sdr_5 = float((valid_dists < 5.0).mean() * 100.0)
    sdr_10 = float((valid_dists < 10.0).mean() * 100.0)
    sdr_15 = float((valid_dists < 15.0).mean() * 100.0)
    sdr_20 = float((valid_dists < 20.0).mean() * 100.0)
    sdr_30 = float((valid_dists < 30.0).mean() * 100.0)

    # 2. Macro target MRE
    target_mres = [np.nan] * K
    target_medians = [np.nan] * K
    target_valid_counts = [0] * K
    
    indices_to_loop = target_indices if target_indices is not None else list(range(K))
    for k in indices_to_loop:
        k_mask = eval_mask[:, k]
        if k_mask.sum() > 0:
            k_dists = dist_matrix[k_mask, k]
            target_mres[k] = float(k_dists.mean())
            target_medians[k] = float(np.median(k_dists))
            target_valid_counts[k] = int(k_mask.sum())

    valid_target_mres = [e for e in target_mres if not np.isnan(e)]
    macro_target_mre = float(np.mean(valid_target_mres)) if len(valid_target_mres) > 0 else np.nan

    # 3. Macro patient MRE
    patient_mres = []
    for i in range(N):
        i_mask = eval_mask[i]
        if i_mask.sum() > 0:
            i_dists = dist_matrix[i, i_mask]
            patient_mres.append(float(i_dists.mean()))
        else:
            patient_mres.append(np.nan)

    valid_patient_mres = [e for e in patient_mres if not np.isnan(e)]
    macro_patient_mre = float(np.mean(valid_patient_mres)) if len(valid_patient_mres) > 0 else np.nan

    return {
        "micro_mre": micro_mre,
        "macro_target_mre": macro_target_mre,
        "macro_patient_mre": macro_patient_mre,
        "median": median_err,
        "p75": p75_err,
        "p90": p90_err,
        "p95": p95_err,
        "max": max_err,
        "sdr_5": sdr_5,
        "sdr_10": sdr_10,
        "sdr_15": sdr_15,
        "sdr_20": sdr_20,
        "sdr_30": sdr_30,
        "target_mres": target_mres,
        "target_medians": target_medians,
        "target_valid_counts": target_valid_counts,
        "eval_count": len(valid_dists),
        "dist_matrix": dist_matrix,
        "eval_mask": eval_mask
    }
