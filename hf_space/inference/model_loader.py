import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch

space_root = Path(__file__).resolve().parent.parent
repo_root = space_root.parent
sys.path.insert(0, str(space_root))
sys.path.insert(0, str(space_root / "sharon"))
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from sharon.model_target_query import TargetQueryTransformerDecoder

def get_current_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

_loaded_models: Dict[str, List[TargetQueryTransformerDecoder]] = {}

def get_models(model_variant: str = "phase10r", target_device: Optional[torch.device] = None) -> List[TargetQueryTransformerDecoder]:
    """
    Returns the loaded ensemble of 3 models (Seeds 42, 43, 44) for the requested variant.
    model_variant: 'phase10r' (official frozen baseline) or 'phase16_brain' (retrained brain-aware)
    """
    global _loaded_models
    device = target_device or get_current_device()

    if model_variant in _loaded_models:
        return _loaded_models[model_variant]
        
    models = []
    seeds = [42, 43, 44]
    
    # Priority: 1. local space checkpoints, 2. repo experiments checkpoints
    local_ckpt_dir = space_root / "checkpoints"
    
    if model_variant == "phase16_brain":
        prefix = "BrainAware_seed"
        ckpt_dir = local_ckpt_dir if (local_ckpt_dir / f"{prefix}42.pt").exists() else repo_root / "experiments" / "phase16_brain" / "checkpoints"
    else:
        prefix = "C4_Proposed_seed"
        ckpt_dir = local_ckpt_dir if (local_ckpt_dir / f"{prefix}42.pt").exists() else repo_root / "experiments" / "phase10R" / "checkpoints"
        
    print(f"Loading {model_variant} 3-model ensemble from {ckpt_dir} onto {device}...")
    for s in seeds:
        ckpt_path = ckpt_dir / f"{prefix}{s}.pt"
        if not ckpt_path.exists():
            # Fallback to BrainAware or C4 in local directory
            if (local_ckpt_dir / f"BrainAware_seed{s}.pt").exists():
                ckpt_path = local_ckpt_dir / f"BrainAware_seed{s}.pt"
            elif (local_ckpt_dir / f"C4_Proposed_seed{s}.pt").exists():
                ckpt_path = local_ckpt_dir / f"C4_Proposed_seed{s}.pt"
            else:
                fallback_path = repo_root / "experiments" / "phase10R" / "checkpoints" / f"C4_Proposed_seed{s}.pt"
                print(f"Notice: {ckpt_path.name} not found, falling back to {fallback_path.name}")
                ckpt_path = fallback_path
            
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = TargetQueryTransformerDecoder(atlas_coords=torch.zeros(117, 3).to(device), num_organs=117).to(device)
        m.load_state_dict(saved["model_state_dict"])
        m.eval()
        models.append(m)
        
    _loaded_models[model_variant] = models
    print(f"Successfully initialized {len(models)} models on {device}.")
    return models
