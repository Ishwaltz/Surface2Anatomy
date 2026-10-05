"""Tests ensemble disagreement calculation."""

import numpy as np
from surface2anatomy.uncertainty import compute_ensemble_disagreement_mm

def test_disagreement_zero():
    seeds = {
        "42": [10.0, 20.0, 30.0],
        "43": [10.0, 20.0, 30.0],
        "44": [10.0, 20.0, 30.0]
    }
    assert compute_ensemble_disagreement_mm(seeds) == 0.0

def test_disagreement_positive():
    seeds = {
        "42": [0.0, 0.0, 0.0],
        "43": [3.0, 0.0, 0.0],
        "44": [-3.0, 0.0, 0.0]
    }
    # mean is 0, diffs are 0, 3, -3, squared: 0, 9, 9 -> mean 6 -> sqrt(6) = 2.45
    dis = compute_ensemble_disagreement_mm(seeds)
    assert abs(dis - 2.45) < 0.05
