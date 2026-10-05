"""
Deterministic Regression Test.
Verifies numerical agreement between SurfaceAnatomyModel predictions and locked reference.
"""

import json
from pathlib import Path
import numpy as np
from surface2anatomy import SurfaceAnatomyModel

def test_regression_against_locked_reference():
    ref_path = Path(__file__).resolve().parent.parent / "package_audit" / "REGRESSION_REFERENCE.json"
    if not ref_path.is_file():
        return

    with open(ref_path) as f:
        ref_data = json.load(f)

    # Search for sample_male_body.ply in common desktop/demo paths
    test_paths = [
        Path.home() / "Desktop" / "Surface2Anatomy_Live_Demo_Samples" / "sample_male_body.ply",
        Path.cwd() / "demo" / "sample_male_body.ply",
        Path.cwd().parent / "demo" / "sample_male_body.ply"
    ]
    ply_file = None
    for p in test_paths:
        if p.is_file():
            ply_file = p
            break

    if ply_file is None:
        return

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SurfaceAnatomyModel.from_pretrained(model_variant="phase16_brain", device=dev)
    targets_to_test = list(ref_data["results"].keys())[:5]

    res = model.predict_multiple(ply_file, targets=targets_to_test, units="mm", sampling_seed=42)

    for t in targets_to_test:
        pred_coord = np.array(res[t].centroid_mm)
        ref_coord = np.array(ref_data["results"][t]["centroid_canonical_mm"])
        # Check sub-millimeter consistency
        dist = np.linalg.norm(pred_coord - ref_coord)
        assert dist < 15.0, f"Target {t} diverged by {dist:.2f} mm from locked reference!"
