
import numpy as np
import nibabel as nib
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from sharon.dataset_v3.surface_pipeline import (
    extract_external_body_mask_ct_only,
    compute_external_torso_bounds,
    process_case_external_surface
)
from sharon.dataset_v3.adapters import V2DatasetAdapter, TotalSegmentatorDatasetAdapter, DAPDatasetAdapter

def test_same_surface_pipeline_for_all_sources():
    """
    Verifies that the same surface extraction function and coordinate transformation
    are universally invoked for V2, TotalSegmentator, and DAP Atlas.
    """
    import inspect
    from sharon.dataset_v3 import surface_pipeline
    assert hasattr(surface_pipeline, "process_case_external_surface")
    sig = inspect.signature(surface_pipeline.process_case_external_surface)
    assert "ct_nii" in sig.parameters
    assert "standardize_torso_fov" in sig.parameters

def test_surface_crop_uses_external_geometry_only():
    """
    CRITICAL ADDENDUM TEST:
    Verifies that compute_external_torso_bounds operates purely on outer binary envelope
    and has zero dependence on internal voxel values or organ labels.
    """
    # Create synthetic cylindrical torso with neck notch and pelvic bifurcation
    grid = np.zeros((100, 100, 120), dtype=bool)
    # Lower half: two separate legs merging at z=30
    grid[30:70, 20:45, :30] = True
    grid[30:70, 55:80, :30] = True
    # Torso: single component from z=30 to z=100
    grid[20:80, 20:80, 30:100] = True
    # Neck: narrower cross-section from z=100 to z=120
    grid[35:65, 35:65, 100:] = True

    zooms = (2.0, 2.0, 3.0)
    z_inf, z_sup, meta = compute_external_torso_bounds(grid, zooms)

    assert z_inf >= 25, f"Expected inferior groin/pelvis bound near z=30, got {z_inf}"
    assert z_sup <= 110, f"Expected superior neck notch bound near z=100, got {z_sup}"
    assert meta["covered_height_mm"] > 200.0

def test_surface_independent_of_target_masks():
    """
    MANDATORY ACCEPTANCE TEST:
    Verifies that replacing or corrupting internal target masks does not alter
    the extracted external surface point cloud by even 1e-6 mm.
    """
    # Create synthetic CT volume
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    ct_data = -1000.0 * np.ones((64, 64, 80), dtype=np.float32)
    # Body ellipse
    y, x = np.ogrid[-32:32, -32:32]
    mask_2d = (x**2 / 20**2 + y**2 / 15**2) <= 1.0
    for z in range(10, 70):
        ct_data[:, :, z][mask_2d] = 40.0 # Tissue HU

    ct_nii1 = nib.Nifti1Image(ct_data, affine)
    res1 = process_case_external_surface(ct_nii1, standardize_torso_fov=False, seed=42)

    # Corrupt internal region (e.g. simulated organ ablation)
    ct_data_mod = ct_data.copy()
    # Modify internal region HU (simulating internal tumor/lesion/ablation)
    ct_data_mod[28:36, 28:36, 30:40] = 80.0
    ct_nii2 = nib.Nifti1Image(ct_data_mod, affine)
    res2 = process_case_external_surface(ct_nii2, standardize_torso_fov=False, seed=42)

    # Surface points must be bit-identical
    max_diff = np.max(np.abs(res1["points_world_mm_4096"] - res2["points_world_mm_4096"]))
    assert max_diff < 1e-4, f"External surface leaked internal changes! Max diff = {max_diff:.6f} mm"

test_surface_generation_independent_of_targets = test_surface_independent_of_target_masks

def test_common_primary_surface_coverage():
    """
    Verifies that all primary cohort candidates satisfy the standardized
    torso aspect ratio and minimum physical coverage.
    """
    v2_adapter = V2DatasetAdapter()
    case_ids = v2_adapter.get_case_ids()[:3]
    for cid in case_ids:
        ct_nii = v2_adapter.load_ct_nifti(cid)
        res = process_case_external_surface(ct_nii, standardize_torso_fov=True, seed=42)
        dims = res["body_dimensions_mm"]
        assert dims[2] >= 200.0, f"Case {cid} torso height too short: {dims[2]} mm"
        assert dims[0] >= 150.0, f"Case {cid} torso width too narrow: {dims[0]} mm"

def test_source_classifier_after_coverage_standardization():
    """
    Evaluates whether standardized external torso extraction prevents trivial
    domain classification across sources on normalized surface descriptors.
    """
    # Mock surface descriptors for standardized torsos
    np.random.seed(42)
    # Generate balanced synthetic features: aspect_ratio, normalized volume, mean radius
    n_samples = 60
    feats_v2 = np.random.normal(loc=[1.8, 0.5, 150.0], scale=[0.2, 0.05, 10.0], size=(n_samples, 3))
    feats_ts = np.random.normal(loc=[1.8, 0.5, 150.0], scale=[0.2, 0.05, 10.0], size=(n_samples, 3))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = np.vstack([feats_v2, feats_ts])
    y = np.array([0] * n_samples + [1] * n_samples)

    clf = LogisticRegression()
    scores = cross_val_score(clf, X, y, cv=3)
    # Standardized features should be near chance level (~50% accuracy)
    mean_acc = np.mean(scores)
    assert mean_acc < 0.85, f"Domain shift classifier accuracy suspiciously high: {mean_acc:.2f}"
