"""
Surface2Anatomy Target Catalog & Anatomical Categories.
Contains authoritative mappings for the 117 model architecture output slots,
the primary 104 benchmark targets, and the full 121 anatomical catalog.
"""

from typing import List, Dict, Tuple, Optional

# Canonical 117 Model Architecture Output Slots (Dataset V3 frozen ontology)
CANONICAL_TARGET_NAMES: Tuple[str, ...] = (
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas",
    "adrenal_gland_right", "adrenal_gland_left", "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "esophagus",
    "trachea", "thyroid_gland", "small_bowel", "duodenum", "colon", "urinary_bladder",
    "prostate", "kidney_cyst_left", "kidney_cyst_right", "sacrum", "vertebrae_S1",
    "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
    "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
    "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
    "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5",
    "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1", "heart", "aorta",
    "pulmonary_vein", "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left",
    "common_carotid_artery_right", "common_carotid_artery_left", "brachiocephalic_vein_left",
    "brachiocephalic_vein_right", "atrial_appendage_left", "superior_vena_cava", "inferior_vena_cava",
    "portal_vein_and_splenic_vein", "iliac_artery_left", "iliac_artery_right", "iliac_vena_left",
    "iliac_vena_right", "humerus_left", "humerus_right", "scapula_left", "scapula_right",
    "clavicula_left", "clavicula_right", "femur_left", "femur_right", "hip_left", "hip_right",
    "spinal_cord", "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left",
    "gluteus_medius_right", "gluteus_minimus_left", "gluteus_minimus_right", "autochthon_left",
    "autochthon_right", "iliopsoas_left", "iliopsoas_right", "brain", "skull", "rib_left_1",
    "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6", "rib_left_7",
    "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12", "rib_right_1",
    "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6", "rib_right_7",
    "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12", "sternum",
    "costal_cartilages",
    # Additional Pelvic Anatomy (Slots 117-120)
    "uterus", "ovary_left", "ovary_right", "vagina"
)

# Slots 0 to 103 are the Primary 104 Benchmark Targets
PRIMARY_BENCHMARK_TARGETS: Tuple[str, ...] = CANONICAL_TARGET_NAMES[:104]

TARGET_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_TARGET_NAMES)}

TARGET_CATEGORIES: Dict[str, List[str]] = {
    "Head & Brain": [
        "brain", "skull"
    ],
    "Thoracic & Respiratory": [
        "heart", "trachea", "esophagus", "thyroid_gland",
        "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"
    ],
    "Abdominal & Visceral": [
        "liver", "spleen", "pancreas", "gallbladder", "stomach", "duodenum",
        "small_bowel", "colon", "kidney_left", "kidney_right",
        "adrenal_gland_left", "adrenal_gland_right", "kidney_cyst_left", "kidney_cyst_right"
    ],
    "Vascular": [
        "aorta", "superior_vena_cava", "inferior_vena_cava", "pulmonary_vein",
        "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left",
        "common_carotid_artery_right", "common_carotid_artery_left",
        "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left",
        "portal_vein_and_splenic_vein", "iliac_artery_left", "iliac_artery_right",
        "iliac_vena_left", "iliac_vena_right"
    ],
    "Spine": [
        "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5",
        "vertebrae_C6", "vertebrae_C7",
        "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5",
        "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10",
        "vertebrae_T11", "vertebrae_T12",
        "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
        "vertebrae_S1", "spinal_cord"
    ],
    "Ribs & Thoracic Cage": [
        "sternum", "costal_cartilages",
        "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6",
        "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
        "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6",
        "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12"
    ],
    "Pelvis & Reproductive": [
        "urinary_bladder", "prostate", "uterus", "ovary_left", "ovary_right", "vagina",
        "sacrum", "hip_left", "hip_right"
    ],
    "Musculoskeletal": [
        "humerus_left", "humerus_right", "scapula_left", "scapula_right",
        "clavicula_left", "clavicula_right", "femur_left", "femur_right",
        "gluteus_maximus_left", "gluteus_maximus_right",
        "gluteus_medius_left", "gluteus_medius_right",
        "gluteus_minimus_left", "gluteus_minimus_right",
        "autochthon_left", "autochthon_right",
        "iliopsoas_left", "iliopsoas_right"
    ]
}

def list_targets(category: Optional[str] = None, benchmark_only: bool = False) -> List[str]:
    """
    Returns the supported internal anatomical targets.
    
    Parameters
    ----------
    category : Optional[str]
        Filter by anatomical category (e.g., 'Head & Brain', 'Abdominal & Visceral', 'Vascular').
    benchmark_only : bool
        If True, returns only the 104 primary benchmark targets.
        
    Returns
    -------
    List[str]
        List of canonical target names.
    """
    if category:
        # Search case-insensitively
        cat_match = None
        for k in TARGET_CATEGORIES:
            if k.lower() == category.lower():
                cat_match = k
                break
        if not cat_match:
            avail = ", ".join(f"'{k}'" for k in TARGET_CATEGORIES.keys())
            raise ValueError(f"Unknown category '{category}'. Available categories: {avail}")
        targets = TARGET_CATEGORIES[cat_match]
        if benchmark_only:
            return [t for t in targets if t in PRIMARY_BENCHMARK_TARGETS]
        return list(targets)

    if benchmark_only:
        return list(PRIMARY_BENCHMARK_TARGETS)
    return list(CANONICAL_TARGET_NAMES)
