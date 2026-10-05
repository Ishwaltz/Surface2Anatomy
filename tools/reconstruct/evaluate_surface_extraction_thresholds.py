import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from skimage import measure

repo_root = Path(__file__).resolve().parent.parent.parent

def test_surface_extraction_on_scan(case_id, thresholds=[-700, -500, -300, -150]):
    ct_p = repo_root / "sharon" / "dataset" / case_id / "ct.nii.gz"
    img = nib.load(str(ct_p))
    ct = img.get_fdata(dtype=np.float32)
    affine = img.affine
    zooms = img.header.get_zooms()

    print(f"\n--- Testing Case: {case_id} (Shape: {ct.shape}, Spacing: {zooms}) ---")

    for th in thresholds:
        # Step 1: Threshold
        binary = ct > th

        # Step 2: Morphological hole filling on 2D axial slices (to include lungs/bowel air inside body)
        # Slices are along axis 2 (z)
        body_mask = np.zeros_like(binary, dtype=bool)
        for k in range(ct.shape[2]):
            body_mask[:, :, k] = ndi.binary_fill_holes(binary[:, :, k])

        # Step 3: Largest connected component in 3D
        labeled, num_cc = ndi.label(body_mask)
        if num_cc == 0:
            print(f"  Threshold {th} HU: NO COMPONENT FOUND")
            continue
        sizes = ndi.sum(body_mask, labeled, range(1, num_cc + 1))
        largest_label = np.argmax(sizes) + 1
        body_clean = (labeled == largest_label)

        # Morphological opening/erosion to disconnect thin table contacts if needed
        # Check volume
        vox_vol_cc = (zooms[0] * zooms[1] * zooms[2]) / 1000.0 # cm^3
        total_vol_liters = (body_clean.sum() * vox_vol_cc) / 1000.0

        # Marching cubes
        verts, faces, normals, values = measure.marching_cubes(body_clean.astype(float), level=0.5, spacing=zooms)
        
        # Bounding box in physical mm (using spacing directly from marching_cubes)
        extent = verts.max(axis=0) - verts.min(axis=0)
        print(f"  Threshold {th} HU: Vol = {total_vol_liters:.2f} L, Mesh Vertices = {len(verts)}, Mesh Faces = {len(faces)}, Extents (X,Y,Z) = ({extent[0]:.1f}, {extent[1]:.1f}, {extent[2]:.1f}) mm")

if __name__ == "__main__":
    for c in ["case_000", "case_005", "case_020"]:
        test_surface_extraction_on_scan(c)
