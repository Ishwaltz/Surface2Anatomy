"""
Deterministic Anatomical Target Synonym Resolution and Fuzzy Matching.
Strictly maps colloquial, clinical, and abbreviated names to canonical targets.
"""

import difflib
from typing import Dict, List, Optional, Tuple

from surface2anatomy.targets import CANONICAL_TARGET_NAMES, TARGET_TO_INDEX
from surface2anatomy.exceptions import UnknownTargetError

SYNONYM_MAP: Dict[str, str] = {
    # Left Kidney
    "left kidney": "kidney_left",
    "kidney left": "kidney_left",
    "left renal": "kidney_left",
    "renal left": "kidney_left",
    "l kidney": "kidney_left",
    "lkidney": "kidney_left",
    
    # Right Kidney
    "right kidney": "kidney_right",
    "kidney right": "kidney_right",
    "right renal": "kidney_right",
    "renal right": "kidney_right",
    "r kidney": "kidney_right",
    "rkidney": "kidney_right",

    # Spleen
    "lien": "spleen",
    "splen": "spleen",

    # Liver
    "hepar": "liver",
    "hepatic": "liver",

    # Gallbladder
    "gall bladder": "gallbladder",
    "vesica biliaris": "gallbladder",
    "cholecyst": "gallbladder",

    # Urinary Bladder
    "bladder": "urinary_bladder",
    "urinary bladder": "urinary_bladder",
    "vesica urinaria": "urinary_bladder",

    # Heart
    "cor": "heart",
    "cardiac": "heart",

    # Brain
    "cerebrum": "brain",
    "encephalon": "brain",

    # Great Vessels
    "ivc": "inferior_vena_cava",
    "inferior vena cava": "inferior_vena_cava",
    "svc": "superior_vena_cava",
    "superior vena cava": "superior_vena_cava",
    "vena cava": "inferior_vena_cava",
    "aortic arch": "aorta",
    "portal vein": "portal_vein_and_splenic_vein",
    "splenic vein": "portal_vein_and_splenic_vein",

    # Digestive
    "stomach": "stomach",
    "ventriculus": "stomach",
    "gaster": "stomach",
    "gastric": "stomach",
    "bowel": "small_bowel",
    "small intestine": "small_bowel",
    "large intestine": "colon",

    # Female Pelvic
    "uterus": "uterus",
    "womb": "uterus",
    "left ovary": "ovary_left",
    "right ovary": "ovary_right",
    "vagina": "vagina"
}

def resolve_target(query: str) -> Tuple[str, int]:
    """
    Deterministically resolves a target string (canonical or synonym) to (canonical_name, slot_index).
    If unresolved, raises UnknownTargetError with close suggestions.
    """
    clean = query.strip().lower().replace("-", "_")

    # 1. Exact match in canonical list
    if clean in TARGET_TO_INDEX:
        return clean, TARGET_TO_INDEX[clean]

    # 2. Match with spaces converted to underscores
    clean_under = clean.replace(" ", "_")
    if clean_under in TARGET_TO_INDEX:
        return clean_under, TARGET_TO_INDEX[clean_under]

    # 3. Match in synonym dictionary
    clean_spaces = clean.replace("_", " ")
    if clean_spaces in SYNONYM_MAP:
        canon = SYNONYM_MAP[clean_spaces]
        return canon, TARGET_TO_INDEX[canon]
    if clean in SYNONYM_MAP:
        canon = SYNONYM_MAP[clean]
        return canon, TARGET_TO_INDEX[canon]

    # 4. Unknown target -> compute closest suggestions
    all_known = list(CANONICAL_TARGET_NAMES) + list(SYNONYM_MAP.keys())
    close_matches = difflib.get_close_matches(clean, all_known, n=4, cutoff=0.5)
    resolved_suggestions = []
    for m in close_matches:
        if m in TARGET_TO_INDEX:
            resolved_suggestions.append(m)
        elif m in SYNONYM_MAP:
            resolved_suggestions.append(SYNONYM_MAP[m])
    
    # Deduplicate while preserving order
    seen = set()
    unique_suggestions = [x for x in resolved_suggestions if not (x in seen or seen.add(x))]

    raise UnknownTargetError(query, suggestions=unique_suggestions)
