"""
Ensemble Disagreement & Spatial Uncertainty Quantification.
Computes RMS Euclidean distance of individual seed outputs from the ensemble centroid.
"""

import numpy as np
from typing import Dict, List, Tuple

def compute_ensemble_disagreement_mm(seed_predictions_mm: Dict[str, List[float]]) -> float:
    """
    Calculates the Root Mean Square (RMS) Euclidean deviation among seed predictions:
    disagreement = sqrt( (1 / S) * sum_s || p_s - p_mean ||^2 )
    
    Parameters
    ----------
    seed_predictions_mm : Dict[str, List[float]]
        Dictionary of seed ID -> [X, Y, Z] in millimeters.
        
    Returns
    -------
    float
        Ensemble disagreement in millimeters.
    """
    if not seed_predictions_mm:
        return 0.0

    coords = np.array(list(seed_predictions_mm.values()), dtype=np.float64)
    if len(coords) <= 1:
        return 0.0

    mean_coord = np.mean(coords, axis=0)
    diffs = coords - mean_coord
    squared_dists = np.sum(diffs ** 2, axis=1)
    rms = np.sqrt(np.mean(squared_dists))
    return float(round(rms, 2))
