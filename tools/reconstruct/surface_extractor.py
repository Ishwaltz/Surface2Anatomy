from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import nibabel as nib
from nibabel.affines import apply_affine
import torch

repo_root = Path(__file__).resolve().parent.parent.parent

def fps(points, n_samples=4096, seed=42):
    """
    Farthest Point Sampling to select n_samples from points (N, 3).
    Deterministic with fixed seed.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts = torch.from_numpy(points).float().to(device)
    N = pts.shape[0]
    
    if N <= n_samples:
        return np.arange(N)

    centroids = torch.zeros(n_samples, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    
    # Deterministic start index
    rng = np.random.RandomState(seed)
    farthest = int(rng.randint(0, N))
    centroids[0] = farthest
    
    for i in range(1, n_samples):
        centroid = pts[farthest, :].view(1, 3)
        dist = torch.sum((pts - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1].item()
        centroids[i] = farthest
        
    return centroids.cpu().numpy()

def extract_patient_body_surface(ct_path_or_img, hu_threshold=-300, num_points=4096, seed=42):
    """
    Extracts real patient-specific external skin surface from CT scan.
    Guaranteed INDEPENDENT of any internal organ targets or segmentations.
    
    Returns dict with:
      - points_world_mm: (num_points, 3)
      - normals_world: (num_points, 3)
      - surface_center_mm: (3,)
      - body_dimensions_mm: (width, depth, height)
      - surface_area_mm2: float
      - body_mask_volume_mm3: float
      - qc_status: 'PASS' / 'WARNING'
    """
    if isinstance(ct_path_or_img, (str, bytes)) or hasattr(ct_path_or_img, "__fspath__"):
        img = nib.load(str(ct_path_or_img))
    else:
        img = ct_path_or_img
        
    ct_data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    zooms = img.header.get_zooms()

    # Step 1: Threshold tissue
    binary = ct_data > hu_threshold

    # Step 2: 2D axial slice hole-filling (removes lungs, bowel gas, internal cavities from outer mask)
    body_mask = np.zeros_like(binary, dtype=bool)
    for k in range(ct_data.shape[2]):
        body_mask[:, :, k] = ndi.binary_fill_holes(binary[:, :, k])

    # Step 3: 3D Largest Connected Component (excludes detached air artifacts & scanner bed)
    labeled, num_cc = ndi.label(body_mask)
    if num_cc == 0:
        raise ValueError("No body component detected in CT scan.")
    
    component_sizes = ndi.sum(body_mask, labeled, range(1, num_cc + 1))
    largest_label = np.argmax(component_sizes) + 1
    body_clean = (labeled == largest_label)

    # Compute body volume
    vox_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])
    body_volume_mm3 = float(body_clean.sum() * vox_vol_mm3)

    # Step 4: Marching Cubes in voxel space
    verts_vox, faces, normals_vox, _ = measure.marching_cubes(body_clean.astype(float), level=0.5)

    # Step 5: Convert vertices to physical millimeters via official NIfTI affine
    verts_world = apply_affine(affine, verts_vox)

    # Transform normals: normal_world = (A^{-1})^T * normal_vox
    linear_affine = affine[:3, :3]
    normal_transform = np.linalg.inv(linear_affine).T
    normals_world = normals_vox @ normal_transform.T
    norm_mags = np.linalg.norm(normals_world, axis=-1, keepdims=True)
    normals_world = normals_world / np.maximum(norm_mags, 1e-8)

    # Compute surface area
    v0 = verts_world[faces[:, 0]]
    v1 = verts_world[faces[:, 1]]
    v2 = verts_world[faces[:, 2]]
    cross_prod = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross_prod, axis=-1)
    surface_area_mm2 = float(tri_areas.sum())

    # Step 6: Uniform surface sampling via face area distribution
    rng = np.random.RandomState(seed)
    face_probs = tri_areas / surface_area_mm2
    oversample_factor = 3
    num_candidates = min(num_points * oversample_factor, len(faces))
    
    chosen_faces = rng.choice(len(faces), size=num_candidates, p=face_probs)
    
    # Barycentric coordinates
    r1 = rng.rand(num_candidates, 1)
    r2 = rng.rand(num_candidates, 1)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = r2 * sqrt_r1
    w = 1.0 - u - v

    sampled_pts = (u * verts_world[faces[chosen_faces, 0]] +
                   v * verts_world[faces[chosen_faces, 1]] +
                   w * verts_world[faces[chosen_faces, 2]])

    sampled_normals = (u * normals_world[faces[chosen_faces, 0]] +
                       v * normals_world[faces[chosen_faces, 1]] +
                       w * normals_world[faces[chosen_faces, 2]])
    s_norm_mags = np.linalg.norm(sampled_normals, axis=-1, keepdims=True)
    sampled_normals = sampled_normals / np.maximum(s_norm_mags, 1e-8)

    # Step 7: Farthest Point Sampling to select exact num_points
    fps_idx = fps(sampled_pts, n_samples=num_points, seed=seed)
    final_points = sampled_pts[fps_idx].astype(np.float32)
    final_normals = sampled_normals[fps_idx].astype(np.float32)

    # Compute bounding box & surface center
    min_bounds = final_points.min(axis=0)
    max_bounds = final_points.max(axis=0)
    surface_center_mm = ((min_bounds + max_bounds) / 2.0).astype(np.float32)
    body_dimensions_mm = (max_bounds - min_bounds).astype(np.float32)

    # QC check
    qc_status = "PASS"
    if body_dimensions_mm[2] > 1000.0 or body_dimensions_mm[0] < 150.0:
        qc_status = "WARNING"

    return {
        "points_world_mm": final_points,
        "normals_world": final_normals,
        "surface_center_mm": surface_center_mm,
        "body_dimensions_mm": body_dimensions_mm,
        "surface_area_mm2": surface_area_mm2,
        "body_mask_volume_mm3": body_volume_mm3,
        "qc_status": qc_status
    }

if __name__ == "__main__":
    res = extract_patient_body_surface(repo_root / "sharon" / "dataset" / "case_000" / "ct.nii.gz")
    print("Test extract_patient_body_surface on case_000:")
    print("  Points shape:", res["points_world_mm"].shape)
    print("  Normals shape:", res["normals_world"].shape)
    print("  Surface center (mm):", res["surface_center_mm"])
    print("  Body dimensions (W, D, H) mm:", res["body_dimensions_mm"])
    print("  Surface area:", res["surface_area_mm2"], "mm^2")
    print("  Body volume:", res["body_mask_volume_mm3"] / 1e6, "L")
    print("  QC status:", res["qc_status"])
