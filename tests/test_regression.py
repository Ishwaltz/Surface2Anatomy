import sys
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from api.inference.preprocessing import preprocess_surface_bytes
from api.inference.predict import predict_single_or_multi
from api.inference.model_loader import get_models

repo_root = Path(__file__).resolve().parent.parent

def test_scientific_regression():
    """
    Mandatory Scientific Regression Test:
    Asserts that backend API inference matches the offline frozen Phase 10R inference
    within < 1e-4 normalized units (< 0.05 mm) on a canonical reference sample.
    """
    ply_path = repo_root / "web" / "public" / "demo" / "sample_torso.ply"
    assert ply_path.exists(), "Reference demo surface must exist"
    
    with open(ply_path, "rb") as f:
        content = f.read()
        
    # 1. Run through backend preprocessing
    prep = preprocess_surface_bytes("sample_torso.ply", content)
    pts_norm = prep["pts_norm"]
    c_external = prep["c_external"]
    
    # 2. Run through backend predict module
    test_targets = ["liver", "spleen", "kidney_left", "kidney_right", "aorta", "heart", "urinary_bladder"]
    res_api = predict_single_or_multi(pts_norm, c_external, test_targets, model_variant="phase10r")
    
    # 3. Run raw offline inference directly on models to ensure zero drift
    models = get_models("phase10r")
    device = next(models[0].parameters()).device
    pts_t = torch.from_numpy(pts_norm).float().unsqueeze(0).to(device)
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        offline_preds = []
        for m in models:
            out, _ = m(pts_t)
            offline_preds.append(out[0].cpu().numpy() * 500.0)
    ens_offline = np.mean(offline_preds, axis=0) # (117, 3) mm
    
    # 4. Compare every test target
    for t_name in test_targets:
        api_data = res_api["results"][t_name]
        slot = api_data["target_index"]
        
        api_coord_mm = np.array(api_data["centroid_canonical_mm"])
        offline_coord_mm = ens_offline[slot]
        
        diff_mm = np.max(np.abs(api_coord_mm - offline_coord_mm))
        diff_norm = diff_mm / 500.0
        
        print(f"Target {t_name:16s}: API={api_coord_mm} vs Offline={offline_coord_mm[:3]} | Diff={diff_mm:.6f} mm ({diff_norm:.6e} norm)")
        assert diff_norm < 1e-4, f"Regression test failed for {t_name}: {diff_norm} >= 1e-4"
        
    print("\n✅ SCIENTIFIC REGRESSION TEST PASSED: Discrepancy strictly < 1e-4 normalized units across all targets!")

if __name__ == "__main__":
    test_scientific_regression()
