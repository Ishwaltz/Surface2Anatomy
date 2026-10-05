"""
Pretrained Model Manifest & Cryptographic SHA256 Checksums.
All checkpoints are verified against these hashes prior to deserialization.
"""

from typing import Dict, Any

# Primary remote release repositories
HF_SPACE_BASE = "https://huggingface.co/spaces/SharonMelhi/surface2anatomy-backend/resolve/main/checkpoints"
GITHUB_RELEASE_BASE = "https://github.com/Sharon-codes/S2A-Net-IIT-Mandi/releases/download/v0.1.0"
HF_RELEASE_BASE = "https://huggingface.co/Sharon-codes/Surface2Anatomy-Checkpoints/resolve/main"

MODEL_MANIFEST: Dict[str, Dict[str, Any]] = {
    "phase10r": {
        "description": "Official Phase 10R Locked Test Ensemble (C4 Target-Query Transformer)",
        "seed42": {
            "filename": "C4_Proposed_seed42.pt",
            "sha256": "760adfee1c47d80fc3f59492971b6c83459fc10ce2166f13fc1d7e4d708cb936",
            "size_bytes": 20012962,
            "urls": [
                f"{HF_SPACE_BASE}/C4_Proposed_seed42.pt",
                f"{GITHUB_RELEASE_BASE}/C4_Proposed_seed42.pt",
                f"{HF_RELEASE_BASE}/C4_Proposed_seed42.pt"
            ]
        },
        "seed43": {
            "filename": "C4_Proposed_seed43.pt",
            "sha256": "3e3c739c992b80f193ec41041e62ff59191436e2e5f1e78bd0a328c8310f7d91",
            "size_bytes": 20013026,
            "urls": [
                f"{HF_SPACE_BASE}/C4_Proposed_seed43.pt",
                f"{GITHUB_RELEASE_BASE}/C4_Proposed_seed43.pt",
                f"{HF_RELEASE_BASE}/C4_Proposed_seed43.pt"
            ]
        },
        "seed44": {
            "filename": "C4_Proposed_seed44.pt",
            "sha256": "28fbc7b30d8539f84c265b4f95f4f73b63978fb5b252f9a3ad20ea4fa1792cdd",
            "size_bytes": 20013346,
            "urls": [
                f"{HF_SPACE_BASE}/C4_Proposed_seed44.pt",
                f"{GITHUB_RELEASE_BASE}/C4_Proposed_seed44.pt",
                f"{HF_RELEASE_BASE}/C4_Proposed_seed44.pt"
            ]
        }
    },
    "phase16_brain": {
        "description": "Phase 16 Brain-Aware Retrained Ensemble",
        "seed42": {
            "filename": "BrainAware_seed42.pt",
            "sha256": "3e1adad13a0db447cd1de28cd148c0a1ccb033358f3d7f143b211bb52a87980a",
            "size_bytes": 19646595,
            "urls": [
                f"{HF_SPACE_BASE}/BrainAware_seed42.pt",
                f"{GITHUB_RELEASE_BASE}/BrainAware_seed42.pt",
                f"{HF_RELEASE_BASE}/BrainAware_seed42.pt"
            ]
        },
        "seed43": {
            "filename": "BrainAware_seed43.pt",
            "sha256": "ee44025a72a936a55ada128e06c5648f9d363d7f707f14522d05b9361efdfed3",
            "size_bytes": 19646595,
            "urls": [
                f"{HF_SPACE_BASE}/BrainAware_seed43.pt",
                f"{GITHUB_RELEASE_BASE}/BrainAware_seed43.pt",
                f"{HF_RELEASE_BASE}/BrainAware_seed43.pt"
            ]
        },
        "seed44": {
            "filename": "BrainAware_seed44.pt",
            "sha256": "968782effdacfc445d0c86d8c85d6217705c6fe67b420904bd1cd4e1b1297e6b",
            "size_bytes": 19646595,
            "urls": [
                f"{HF_SPACE_BASE}/BrainAware_seed44.pt",
                f"{GITHUB_RELEASE_BASE}/BrainAware_seed44.pt",
                f"{HF_RELEASE_BASE}/BrainAware_seed44.pt"
            ]
        }
    },
    "canonical_alignment": {
        "description": "External Geometry Canonical Ridge Alignment Regressor",
        "ridge_model": {
            "filename": "canonical_alignment_v3_ridge.joblib",
            "sha256": "b9901d72c3aa73e70b1fa77aefdbc1ebf8e682d5f5fa011ee840772827488e26",
            "size_bytes": 1651,
            "urls": [
                f"{HF_SPACE_BASE}/canonical_alignment_v3_ridge.joblib",
                f"{GITHUB_RELEASE_BASE}/canonical_alignment_v3_ridge.joblib",
                f"{HF_RELEASE_BASE}/canonical_alignment_v3_ridge.joblib"
            ]
        }
    }
}
