# 121 Organ Class Mapping for TotalSegmentator v2 + Pelvic Anatomy + SAMe Skeletal/Soft-Tissue Partition
from typing import Union, List, Dict, Tuple, Optional
import torch
import numpy as np

TOTAL_CLASSES_121 = {
    "spleen": 1,
    "kidney_right": 2,
    "kidney_left": 3,
    "gallbladder": 4,
    "liver": 5,
    "stomach": 6,
    "pancreas": 7,
    "adrenal_gland_right": 8,
    "adrenal_gland_left": 9,
    "lung_upper_lobe_left": 10,
    "lung_lower_lobe_left": 11,
    "lung_upper_lobe_right": 12,
    "lung_middle_lobe_right": 13,
    "lung_lower_lobe_right": 14,
    "esophagus": 15,
    "trachea": 16,
    "thyroid_gland": 17,
    "small_bowel": 18,
    "duodenum": 19,
    "colon": 20,
    "urinary_bladder": 21,
    "prostate": 22,
    "sacrum": 23,
    "vertebrae_S1": 24,
    "vertebrae_L5": 25,
    "vertebrae_L4": 26,
    "vertebrae_L3": 27,
    "vertebrae_L2": 28,
    "vertebrae_L1": 29,
    "vertebrae_T12": 30,
    "vertebrae_T11": 31,
    "vertebrae_T10": 32,
    "vertebrae_T9": 33,
    "vertebrae_T8": 34,
    "vertebrae_T7": 35,
    "vertebrae_T6": 36,
    "vertebrae_T5": 37,
    "vertebrae_T4": 38,
    "vertebrae_T3": 39,
    "vertebrae_T2": 40,
    "vertebrae_T1": 41,
    "vertebrae_C7": 42,
    "vertebrae_C6": 43,
    "vertebrae_C5": 44,
    "vertebrae_C4": 45,
    "vertebrae_C3": 46,
    "vertebrae_C2": 47,
    "vertebrae_C1": 48,
    "heart": 49,
    "aorta": 50,
    "pulmonary_vein": 51,
    "brachiocephalic_trunk": 52,
    "subclavian_artery_right": 53,
    "subclavian_artery_left": 54,
    "common_carotid_artery_right": 55,
    "common_carotid_artery_left": 56,
    "superior_vena_cava": 57,
    "inferior_vena_cava": 58,
    "portal_vein_and_splenic_vein": 59,
    "iliac_artery_right": 60,
    "iliac_artery_left": 61,
    "iliac_vena_right": 62,
    "iliac_vena_left": 63,
    "humerus_right": 64,
    "humerus_left": 65,
    "scapula_right": 66,
    "scapula_left": 67,
    "clavicula_right": 68,
    "clavicula_left": 69,
    "femur_right": 70,
    "femur_left": 71,
    "hip_right": 72,
    "hip_left": 73,
    "spinal_cord": 74,
    "gluteus_maximus_right": 75,
    "gluteus_maximus_left": 76,
    "gluteus_medius_right": 77,
    "gluteus_medius_left": 78,
    "gluteus_minimus_right": 79,
    "gluteus_minimus_left": 80,
    "autochthon_right": 81,
    "autochthon_left": 82,
    "iliopsoas_right": 83,
    "iliopsoas_left": 84,
    "brain": 85,
    "skull": 86,
    "rib_left_1": 87,
    "rib_left_2": 88,
    "rib_left_3": 89,
    "rib_left_4": 90,
    "rib_left_5": 91,
    "rib_left_6": 92,
    "rib_left_7": 93,
    "rib_left_8": 94,
    "rib_left_9": 95,
    "rib_left_10": 96,
    "rib_left_11": 97,
    "rib_left_12": 98,
    "rib_right_1": 99,
    "rib_right_2": 100,
    "rib_right_3": 101,
    "rib_right_4": 102,
    "rib_right_5": 103,
    "rib_right_6": 104,
    "rib_right_7": 105,
    "rib_right_8": 106,
    "rib_right_9": 107,
    "rib_right_10": 108,
    "rib_right_11": 109,
    "rib_right_12": 110,
    "sternum": 111,
    "costal_cartilages": 112,
    "atrial_appendage_left": 113,
    "subclavian_vein_right": 114,
    "subclavian_vein_left": 115,
    "brachiocephalic_vein_right": 116,
    "brachiocephalic_vein_left": 117,
    # Female Pelvic Reproductive Anatomy (Classes 118-121)
    "uterus": 118,
    "ovary_left": 119,
    "ovary_right": 120,
    "vagina": 121,
}

ORGAN_INDEX = TOTAL_CLASSES_121
NUM_ORGANS = len(ORGAN_INDEX)
ORGAN_IDS = list(ORGAN_INDEX.values())
ORGAN_NAMES = list(ORGAN_INDEX.keys())
ORGAN_NAME_MAP = {v: k for k, v in ORGAN_INDEX.items()}

# Sex-specific organ definitions
MALE_SPECIFIC_ORGANS = ["prostate"]
FEMALE_SPECIFIC_ORGANS = ["uterus", "ovary_left", "ovary_right", "vagina"]

MALE_SPECIFIC_INDICES = [ORGAN_NAMES.index(o) for o in MALE_SPECIFIC_ORGANS]
FEMALE_SPECIFIC_INDICES = [ORGAN_NAMES.index(o) for o in FEMALE_SPECIFIC_ORGANS]

# ─────────────────────────────────────────────────────────────────────────────
# SAMe Explicit Partition: SKELETAL vs. SOFT TISSUE
# ─────────────────────────────────────────────────────────────────────────────
SKELETAL_CLASSES = [
    name for name in ORGAN_NAMES
    if any(k in name for k in [
        "vertebrae_", "rib_", "sacrum", "sternum", "costal_cartilages",
        "humerus_", "scapula_", "clavicula_", "femur_", "hip_", "skull"
    ])
]

SOFT_TISSUE_CLASSES = [name for name in ORGAN_NAMES if name not in SKELETAL_CLASSES]

NUM_SKELETAL = len(SKELETAL_CLASSES)       # 63
NUM_SOFT_TISSUE = len(SOFT_TISSUE_CLASSES) # 58

SKELETAL_INDICES = [ORGAN_NAMES.index(name) for name in SKELETAL_CLASSES]
SOFT_TISSUE_INDICES = [ORGAN_NAMES.index(name) for name in SOFT_TISSUE_CLASSES]


def get_skeletal_indices(device: torch.device = None) -> torch.Tensor:
    return torch.tensor(SKELETAL_INDICES, dtype=torch.long, device=device)


def get_soft_tissue_indices(device: torch.device = None) -> torch.Tensor:
    return torch.tensor(SOFT_TISSUE_INDICES, dtype=torch.long, device=device)


def get_skeletal_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask[:, SKELETAL_INDICES]


def get_soft_tissue_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask[:, SOFT_TISSUE_INDICES]


def combine_skeletal_and_soft_predictions(
    y_skel: torch.Tensor,
    mu_soft: torch.Tensor,
) -> torch.Tensor:
    B = y_skel.shape[0]
    device = y_skel.device
    target_dtype = y_skel.dtype
    mu_soft = mu_soft.to(dtype=target_dtype)
    full_coords = torch.zeros((B, NUM_ORGANS, 3), dtype=target_dtype, device=device)
    full_coords[:, SKELETAL_INDICES, :] = y_skel
    full_coords[:, SOFT_TISSUE_INDICES, :] = mu_soft
    return full_coords


# 4 Functional Anatomical Regions:
REGION_NAMES = ["Thoracic", "Abdominal", "Pelvic", "Skeletal"]
NUM_REGIONS = 4

ANATOMICAL_REGIONS: Dict[str, int] = {}
for name in ORGAN_NAMES:
    if any(k in name for k in ["lung", "heart", "aorta", "esophagus", "trachea", "pulmonary", "brachiocephalic", "subclavian", "carotid", "superior_vena_cava", "sternum", "costal", "atrial", "thyroid"]):
        ANATOMICAL_REGIONS[name] = 0
    elif any(k in name for k in ["liver", "spleen", "kidney", "gallbladder", "stomach", "pancreas", "adrenal", "bowel", "duodenum", "colon", "inferior_vena_cava", "portal_vein"]):
        ANATOMICAL_REGIONS[name] = 1
    elif any(k in name for k in ["bladder", "prostate", "uterus", "ovary", "vagina", "iliac", "gluteus", "iliopsoas"]):
        ANATOMICAL_REGIONS[name] = 2
    else:
        ANATOMICAL_REGIONS[name] = 3


def get_organ_region_indices(device: torch.device = None) -> torch.Tensor:
    indices = [ANATOMICAL_REGIONS[name] for name in ORGAN_NAMES]
    return torch.tensor(indices, dtype=torch.long, device=device)


def compute_regional_ground_truth(
    centroids: torch.Tensor,
    mask: torch.Tensor,
    region_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    is_batched = centroids.dim() == 3
    if not is_batched:
        centroids = centroids.unsqueeze(0)
        mask = mask.unsqueeze(0)

    B, K, _ = centroids.shape
    device = centroids.device
    if region_indices is None:
        region_indices = get_organ_region_indices(device=device)

    R_gt = torch.zeros((B, NUM_REGIONS, 3), dtype=centroids.dtype, device=device)

    for r in range(NUM_REGIONS):
        r_mask = (region_indices == r).unsqueeze(0).expand(B, -1)
        valid_r = (mask * r_mask.float()).unsqueeze(-1)
        sum_pos = (centroids * valid_r).sum(dim=1)
        count = valid_r.sum(dim=1).clamp(min=1.0)
        R_gt[:, r] = sum_pos / count

    return R_gt if is_batched else R_gt.squeeze(0)


def get_sex_organ_mask(
    sex: Union[int, float, str, torch.Tensor, np.ndarray, List],
    num_organs: int = 121,
    device: torch.device = None,
) -> torch.Tensor:
    if isinstance(sex, torch.Tensor):
        if sex.dim() == 0:
            s_val = int(sex.item())
            mask = torch.ones(num_organs, dtype=torch.float32, device=sex.device)
            if s_val == 1:
                for idx in FEMALE_SPECIFIC_INDICES:
                    if idx < num_organs:
                        mask[idx] = 0.0
            else:
                for idx in MALE_SPECIFIC_INDICES:
                    if idx < num_organs:
                        mask[idx] = 0.0
            return mask
        elif sex.dim() == 1:
            if sex.numel() == 2 and (sex.dtype in (torch.float32, torch.float64)) and (sex[0] + sex[1] == 1 or (sex[0] in (0, 1) and sex[1] in (0, 1))):
                s_val = int(torch.argmax(sex).item())
                return get_sex_organ_mask(s_val, num_organs=num_organs, device=sex.device)
            B = sex.shape[0]
            mask = torch.ones((B, num_organs), dtype=torch.float32, device=sex.device)
            for b in range(B):
                s_val = int(sex[b].item())
                if s_val == 1:
                    for idx in FEMALE_SPECIFIC_INDICES:
                        if idx < num_organs:
                            mask[b, idx] = 0.0
                else:
                    for idx in MALE_SPECIFIC_INDICES:
                        if idx < num_organs:
                            mask[b, idx] = 0.0
            return mask
        elif sex.dim() == 2:
            B = sex.shape[0]
            mask = torch.ones((B, num_organs), dtype=torch.float32, device=sex.device)
            if sex.shape[1] == 2:
                is_male = torch.argmax(sex, dim=-1)
            else:
                is_male = (sex.squeeze(-1) > 0.5).long()

            for b in range(B):
                if is_male[b] == 1:
                    for idx in FEMALE_SPECIFIC_INDICES:
                        if idx < num_organs:
                            mask[b, idx] = 0.0
                else:
                    for idx in MALE_SPECIFIC_INDICES:
                        if idx < num_organs:
                            mask[b, idx] = 0.0
            return mask

    if isinstance(sex, (list, tuple, np.ndarray)):
        arr = np.array(sex)
        if arr.ndim == 0 or (arr.ndim == 1 and len(arr) == 1):
            val = int(arr.item())
            return get_sex_organ_mask(val, num_organs=num_organs, device=device)
        elif arr.ndim == 1 and len(arr) == 2 and (arr[0] in (0, 1) and arr[1] in (0, 1)):
            val = int(np.argmax(arr))
            return get_sex_organ_mask(val, num_organs=num_organs, device=device)
        else:
            t = torch.tensor(arr)
            return get_sex_organ_mask(t, num_organs=num_organs, device=device)

    if isinstance(sex, str):
        s_upper = sex.strip().upper()
        s_val = 1 if s_upper in ("M", "MALE", "1") else 0
    else:
        s_val = 1 if int(sex) == 1 else 0

    mask = torch.ones(num_organs, dtype=torch.float32, device=device)
    if s_val == 1:
        for idx in FEMALE_SPECIFIC_INDICES:
            if idx < num_organs:
                mask[idx] = 0.0
    else:
        for idx in MALE_SPECIFIC_INDICES:
            if idx < num_organs:
                mask[idx] = 0.0
    return mask


def get_organ_names():
    return ORGAN_NAMES
