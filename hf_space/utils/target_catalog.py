from typing import Dict, List, Tuple, Optional

# Canonical list of 117 targets from Dataset V3
# Slots 0-103 are primary 104 targets
CANONICAL_TARGET_NAMES = [
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
    # Female Reproductive Pelvic Anatomy (Slots 117-120)
    "uterus", "ovary_left", "ovary_right", "vagina"
]

TARGET_NAME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_TARGET_NAMES)}

MALE_TARGETS = {"prostate"}
FEMALE_TARGETS = {"uterus", "ovary_left", "ovary_right", "vagina"}

TARGET_CATEGORIES: Dict[str, List[str]] = {
    "Abdominal": [
        "liver", "spleen", "pancreas", "gallbladder", "stomach", "duodenum",
        "small_bowel", "colon", "kidney_left", "kidney_right", "adrenal_gland_left",
        "adrenal_gland_right", "kidney_cyst_left", "kidney_cyst_right"
    ],
    "Thoracic / Respiratory": [
        "heart", "trachea", "esophagus", "thyroid_gland",
        "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
        "lung_middle_lobe_right", "lung_lower_lobe_right"
    ],
    "Vascular": [
        "aorta", "superior_vena_cava", "inferior_vena_cava", "pulmonary_vein",
        "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left",
        "common_carotid_artery_right", "common_carotid_artery_left",
        "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left",
        "portal_vein_and_splenic_vein", "iliac_artery_left", "iliac_artery_right",
        "iliac_vena_left", "iliac_vena_right"
    ],
    "Pelvis & Reproductive": [
        "urinary_bladder", "prostate", "uterus", "ovary_left", "ovary_right", "vagina",
        "sacrum", "hip_left", "hip_right",
        "femur_left", "femur_right", "gluteus_maximus_left", "gluteus_maximus_right",
        "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left",
        "gluteus_minimus_right", "iliopsoas_left", "iliopsoas_right"
    ],
    "Spine": [
        "spinal_cord", "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4",
        "vertebrae_C5", "vertebrae_C6", "vertebrae_C7", "vertebrae_T1", "vertebrae_T2",
        "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6", "vertebrae_T7",
        "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
        "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
        "vertebrae_S1", "autochthon_left", "autochthon_right"
    ],
    "Ribs & Thoracic Skeleton": [
        "sternum", "costal_cartilages", "clavicula_left", "clavicula_right",
        "scapula_left", "scapula_right", "humerus_left", "humerus_right",
        "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5",
        "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10",
        "rib_left_11", "rib_left_12", "rib_right_1", "rib_right_2", "rib_right_3",
        "rib_right_4", "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8",
        "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12"
    ],
    "Head & Cranial": [
        "brain", "skull"
    ]
}

SYNONYMS: Dict[str, str] = {
    # Kidneys
    "left kidney": "kidney_left", "kidney left": "kidney_left", "l kidney": "kidney_left", "renal left": "kidney_left",
    "right kidney": "kidney_right", "kidney right": "kidney_right", "r kidney": "kidney_right", "renal right": "kidney_right",
    "kidneys": "kidney_left",
    # Major organs
    "liver": "liver", "hepatic": "liver",
    "spleen": "spleen", "splenic": "spleen",
    "pancreas": "pancreas", "pancreatic": "pancreas",
    "stomach": "stomach", "gastric": "stomach",
    "gallbladder": "gallbladder", "gall bladder": "gallbladder", "cholecyst": "gallbladder",
    "urinary bladder": "urinary_bladder", "bladder": "urinary_bladder",
    "prostate": "prostate", "prostate gland": "prostate",
    # Female reproductive organs
    "uterus": "uterus", "womb": "uterus", "uterine": "uterus",
    "left ovary": "ovary_left", "ovary left": "ovary_left", "l ovary": "ovary_left",
    "right ovary": "ovary_right", "ovary right": "ovary_right", "r ovary": "ovary_right",
    "ovary": "ovary_left", "ovaries": "ovary_left",
    "vagina": "vagina", "vaginal canal": "vagina", "vaginal": "vagina",
    "brain": "brain", "cerebrum": "brain", "head": "brain",
    "skull": "skull", "cranium": "skull",
    "heart": "heart", "cardiac": "heart", "myocardium": "heart",
    "aorta": "aorta", "abdominal aorta": "aorta", "thoracic aorta": "aorta",
    "ivc": "inferior_vena_cava", "inferior vena cava": "inferior_vena_cava",
    "svc": "superior_vena_cava", "superior vena cava": "superior_vena_cava",
    "trachea": "trachea", "windpipe": "trachea",
    "esophagus": "esophagus", "gullet": "esophagus",
    "thyroid": "thyroid_gland", "thyroid gland": "thyroid_gland",
    "small bowel": "small_bowel", "small intestine": "small_bowel",
    "duodenum": "duodenum",
    "colon": "colon", "large intestine": "colon", "bowel": "colon",
    "spinal cord": "spinal_cord",
    # Lungs
    "left lung": "lung_upper_lobe_left", "right lung": "lung_upper_lobe_right",
    # Spine
    "l1": "vertebrae_L1", "l2": "vertebrae_L2", "l3": "vertebrae_L3", "l4": "vertebrae_L4", "l5": "vertebrae_L5",
    "t12": "vertebrae_T12", "t1": "vertebrae_T1", "c7": "vertebrae_C7",
    "sacrum": "sacrum",
}

def resolve_target(query: str) -> Optional[Tuple[str, int]]:
    """
    Resolves user search query or voice input to canonical target name and slot index.
    """
    q = query.strip().lower().replace("-", "_").replace(" ", "_")
    
    # 1. Exact match
    if q in TARGET_NAME_TO_INDEX:
        return q, TARGET_NAME_TO_INDEX[q]
        
    # 2. Check synonyms
    q_clean = query.strip().lower()
    if q_clean in SYNONYMS:
        canon = SYNONYMS[q_clean]
        return canon, TARGET_NAME_TO_INDEX[canon]
        
    # 3. Substring match
    for canon, idx in TARGET_NAME_TO_INDEX.items():
        if q in canon or canon in q:
            return canon, idx
            
    return None
