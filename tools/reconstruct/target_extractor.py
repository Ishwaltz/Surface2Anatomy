import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
import scipy.ndimage as ndi

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES, TOTAL_CLASSES_121, FEMALE_SPECIFIC_INDICES

def extract_patient_targets(seg_path_or_img, num_organs=121):
    """
    Extracts internal anatomical targets directly using the NIfTI affine.
    Uses library-supported apply_affine(affine, com_ijk) without manual axis swaps.
    
    Returns:
      - targets_world_mm: (121, 3) float32
      - target_mask: (121,) float32 (1.0 for valid, 0.0 for absent)
      - target_primary_mask: (121,) float32 (1.0 for genuine clinical target, 0.0 for synthetic/absent)
      - target_provenance: list of 121 str ('REAL_SEGMENTATION' or 'SYNTHETIC' or 'ABSENT')
      - validation_records: list of dicts for 02_target_coordinate_validation.csv
    """
    if isinstance(seg_path_or_img, (str, bytes)) or hasattr(seg_path_or_img, "__fspath__"):
        img = nib.load(str(seg_path_or_img))
    else:
        img = seg_path_or_img

    seg_data = img.get_fdata().astype(np.int16)
    affine = img.affine

    targets_world_mm = np.zeros((num_organs, 3), dtype=np.float32)
    target_mask = np.zeros(num_organs, dtype=np.float32)
    target_primary_mask = np.zeros(num_organs, dtype=np.float32)
    target_provenance = ["ABSENT"] * num_organs
    validation_records = []

    # Calculate center of mass for all labels 1..121 present in segmentation
    # Labels 1..121 map to 0..120
    indices = np.arange(1, num_organs + 1)
    coms = ndi.center_of_mass(np.ones_like(seg_data, dtype=np.uint8), labels=seg_data, index=indices)

    for idx, (com, name) in enumerate(zip(coms, ORGAN_NAMES)):
        organ_id = idx + 1
        
        # Check if structure is present (finite com)
        if com is not None and not np.isnan(com[0]):
            com_ijk = np.asarray(com, dtype=np.float64) # (i, j, k) in array order
            
            # 1. Primary transformation via official nibabel apply_affine
            coord_mm = apply_affine(affine, com_ijk)
            
            # 2. Independent second-method verification: p = A @ [i, j, k, 1]^T
            homog = np.array([com_ijk[0], com_ijk[1], com_ijk[2], 1.0], dtype=np.float64)
            coord_direct = (affine @ homog)[:3]
            disc = float(np.linalg.norm(coord_mm - coord_direct))
            
            targets_world_mm[idx] = coord_mm.astype(np.float32)
            target_mask[idx] = 1.0

            # Rule 6 / Stage 7: Separate synthetic targets (classes 118-121: uterus, ovaries, vagina)
            if organ_id in [118, 119, 120, 121]:
                prov = "SYNTHETIC"
                target_primary_mask[idx] = 0.0 # Excluded from clinical primary benchmark
            else:
                prov = "REAL_SEGMENTATION"
                target_primary_mask[idx] = 1.0 # Genuine clinical target
                
            target_provenance[idx] = prov

            validation_records.append({
                "target_index": idx,
                "target_name": name,
                "i": float(com_ijk[0]),
                "j": float(com_ijk[1]),
                "k": float(com_ijk[2]),
                "world_x_mm": float(coord_mm[0]),
                "world_y_mm": float(coord_mm[1]),
                "world_z_mm": float(coord_mm[2]),
                "verification_error_mm": disc
            })
            
    return {
        "targets_world_mm": targets_world_mm,
        "target_mask": target_mask,
        "target_primary_mask": target_primary_mask,
        "target_provenance": target_provenance,
        "validation_records": validation_records
    }

if __name__ == "__main__":
    res = extract_patient_targets(repo_root / "sharon" / "dataset" / "case_000" / "segmentation.nii.gz")
    valid_count = int(res["target_mask"].sum())
    prim_count = int(res["target_primary_mask"].sum())
    print("Test extract_patient_targets on case_000:")
    print(f"  Total valid targets:   {valid_count} / 121")
    print(f"  Primary valid targets: {prim_count} / 121")
    print(f"  Max verification error: {max(r['verification_error_mm'] for r in res['validation_records']):.8e} mm")
    print(f"  Spleen (idx 0) world coords: {res['targets_world_mm'][0]}")
