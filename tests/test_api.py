import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.utils.target_catalog import resolve_target
from api.inference.pointcloud import detect_and_normalize_units, sample_4096_points

client = TestClient(app)
repo_root = Path(__file__).resolve().parent.parent

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] >= 1
    assert data["targets_available"] >= 104

def test_get_targets():
    response = client.get("/targets")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "Abdominal" in data["categories"]
    assert "liver" in data["all_targets"]
    assert "spleen" in data["all_targets"]

def test_target_resolution():
    assert resolve_target("spleen")[0] == "spleen"
    assert resolve_target("left kidney")[0] == "kidney_left"
    assert resolve_target("renal right")[0] == "kidney_right"
    assert resolve_target("urinary bladder")[0] == "urinary_bladder"
    assert resolve_target("brain")[0] == "brain"
    assert resolve_target("non_existent_xyz_123") is None

def test_unit_detection():
    # Scale in meters (e.g. 0.4 m)
    pts_m = np.random.uniform(0.0, 0.45, size=(100, 3)).astype(np.float32)
    pts_conv, unit_str = detect_and_normalize_units(pts_m)
    assert "meters" in unit_str
    assert np.max(pts_conv) > 100.0 # Converted to mm

    # Scale in millimeters
    pts_mm = np.random.uniform(0.0, 450.0, size=(100, 3)).astype(np.float32)
    pts_conv2, unit_str2 = detect_and_normalize_units(pts_mm)
    assert "millimeters" in unit_str2
    assert np.isclose(pts_conv2, pts_mm).all()

def test_sample_4096():
    pts = np.random.randn(8000, 3).astype(np.float32)
    sampled = sample_4096_points(pts)
    assert sampled.shape == (4096, 3)

def test_predict_demo_surface():
    ply_path = repo_root / "web" / "public" / "demo" / "sample_torso.ply"
    assert ply_path.exists(), "Demo surface file should exist"
    
    with open(ply_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"surface_file": ("sample_torso.ply", f, "application/octet-stream")},
            data={"target": "spleen", "model_variant": "phase10r"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["target"] == "spleen"
    assert len(data["centroid_canonical_mm"]) == 3
    assert len(data["centroid_input_world_mm"]) == 3
    assert data["uncertainty_mm"] >= 0.0
    assert "disclaimer" in data

def test_predict_multiple_demo_surface():
    ply_path = repo_root / "web" / "public" / "demo" / "sample_torso.ply"
    with open(ply_path, "rb") as f:
        response = client.post(
            "/predict-multiple",
            files={"surface_file": ("sample_torso.ply", f, "application/octet-stream")},
            data={"targets": "liver,spleen,kidney_left,kidney_right", "model_variant": "phase10r"}
        )
    assert response.status_code == 200
    data = response.json()
    assert "liver" in data["results"]
    assert "spleen" in data["results"]
    assert "kidney_left" in data["results"]
    assert "kidney_right" in data["results"]

def test_nan_rejection():
    # Construct invalid surface with NaN
    pts_nan = np.zeros((100, 3), dtype=np.float32)
    pts_nan[0, 0] = np.nan
    
    import io
    bio = io.BytesIO()
    np.save(bio, pts_nan)
    bio.seek(0)
    
    response = client.post(
        "/predict",
        files={"surface_file": ("test_nan.npy", bio.read(), "application/octet-stream")},
        data={"target": "liver"}
    )
    assert response.status_code == 400
    assert "invalid non-finite coordinates" in response.json()["detail"]
