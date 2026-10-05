import numpy as np
from typing import Dict, Tuple

def compute_uncertainty(seed_preds: Dict[str, np.ndarray], ens_pred: np.ndarray) -> Tuple[float, str]:
    """
    Computes ensemble model disagreement as the RMS Euclidean distance of seed predictions
    from the ensemble mean.
    
    Returns:
        rms_mm (float): Root-mean-square distance in millimeters
        level (str): Qualitative disagreement rating ('Low', 'Moderate', 'High')
    """
    diffs = []
    for s_key, p in seed_preds.items():
        diffs.append(np.sum((p - ens_pred) ** 2))
        
    mean_sq_dist = float(np.mean(diffs))
    rms_mm = float(np.sqrt(max(0.0, mean_sq_dist)))
    
    if rms_mm <= 15.0:
        level = "Low"
    elif rms_mm <= 30.0:
        level = "Moderate"
    else:
        level = "High"
        
    return round(rms_mm, 2), level
