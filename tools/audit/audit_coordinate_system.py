import sys
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

def audit_coordinate_system():
    print("=" * 80)
    print("AUDIT: TRUE COORDINATE SYSTEM & VOXEL-TO-WORLD CONVERSION")
    print("=" * 80)

    dataset_dir = repo_root / "sharon" / "dataset"
    cases = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and (d / "segmentation.nii.gz").exists()])
    print(f"Total cases evaluated: {len(cases)}")

    # Sample cases for affine inspection
    sample_cases = cases[:10]
    axcode_counts = {}
    zoom_stats = []

    for c in sample_cases:
        seg = nib.load(str(c / "segmentation.nii.gz"))
        axcodes = "".join(nib.orientations.aff2axcodes(seg.affine))
        axcode_counts[axcodes] = axcode_counts.get(axcodes, 0) + 1
        zooms = seg.header.get_zooms()
        zoom_stats.append(zooms)

    print(f"\nOrientation codes in sample: {axcode_counts}")
    print("Sample voxel spacings (dx, dy, dz):")
    for c, z in zip(sample_cases, zoom_stats):
        print(f"  {c.name}: ({z[0]:.3f}, {z[1]:.3f}, {z[2]:.3f}) mm")

    # Detailed inspection of case_000
    case0 = cases[0]
    seg_img = nib.load(str(case0 / "segmentation.nii.gz"))
    affine = seg_img.affine
    shape = seg_img.shape
    zooms = seg_img.header.get_zooms()
    axcodes = nib.orientations.aff2axcodes(affine)

    print(f"\nCase 0 Detailed Header Inspection ({case0.name}):")
    print(f"  Shape:           {shape}")
    print(f"  Zooms:           {zooms}")
    print(f"  Orientation:     {axcodes}")
    print(f"  Affine Matrix:\n{affine}")

    # Inspect voxel axes vs physical axes
    seg_data = seg_img.get_fdata().astype(np.int16)
    spleen_mask = (seg_data == 1)
    com = ndi.center_of_mass(spleen_mask)
    print(f"\nSpleen (label 1) Center of Mass in Voxel Indices (axis 0, axis 1, axis 2):")
    print(f"  com = {com}")

    # Correct nibabel physical mapping: affine @ [i, j, k, 1]
    phys_correct = (affine @ np.array([com[0], com[1], com[2], 1.0]))[:3]

    # Current pointcloud_sampler.py mapping:
    cz, cy, cx = com
    phys_buggy = (affine @ np.array([cx, cy, cz, 1.0], dtype=np.float32))[:3]

    print("\nPhysical Coordinate Comparison:")
    print(f"  Official Nibabel Physical Point: [X={phys_correct[0]:.2f}, Y={phys_correct[1]:.2f}, Z={phys_correct[2]:.2f}] mm")
    print(f"  pointcloud_sampler.py Result:   [X={phys_buggy[0]:.2f}, Y={phys_buggy[1]:.2f}, Z={phys_buggy[2]:.2f}] mm")
    print(f"  Discrepancy:                     [dX={phys_buggy[0]-phys_correct[0]:.2f}, dY={phys_buggy[1]-phys_correct[1]:.2f}, dZ={phys_buggy[2]-phys_correct[2]:.2f}] mm")
    print(f"  Radial Error from Swap:          {np.linalg.norm(phys_buggy - phys_correct):.2f} mm")

    print("\nRoot Cause Analysis:")
    print("  nibabel arrays have shape (X_dim, Y_dim, Z_dim).")
    print("  scipy.ndimage.center_of_mass returns com in order of array axes: (i, j, k).")
    print("  pointcloud_sampler.py line 81 unpacked: cz, cy, cx = com")
    print("  This mapped array axis 0 (in-plane X) to cz, and array axis 2 (slice Z) to cx.")
    print("  Consequently, affine @ [cx, cy, cz, 1.0] swapped X and Z before multiplying with the affine matrix!")
    print("  This resulted in in-plane pixels being scaled by slice thickness (5.0 mm) -> Z > 2,500 mm,")
    print("  and slices being scaled by pixel spacing (-0.57 mm) -> X < 100 mm.")

if __name__ == "__main__":
    audit_coordinate_system()
