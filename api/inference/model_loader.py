import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from sharon.model_target_query import TargetQueryTransformerDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_loaded_models: Dict[str, List[TargetQueryTransformerDecoder]] = {}

def get_models(model_variant: str = "phase10r") -> List[TargetQueryTransformerDecoder]:
    """
    Returns the loaded ensemble of 3 models (Seeds 42, 43, 44) for the requested variant.
    model_variant: 'phase10r' (official frozen baseline) or 'phase16_brain' (retrained brain-aware)
    """
    global _loaded_models
    if model_variant in _loaded_models:
        return _loaded_models[model_variant]
        
    models = []
    seeds = [42, 43, 44]
    
    if model_variant == "phase16_brain":
        ckpt_dir = repo_root / "experiments" / "phase16_brain" / "checkpoints"
        prefix = "BrainAware_seed"
    else:
        ckpt_dir = repo_root / "experiments" / "phase10R" / "checkpoints"
        prefix = "C4_Proposed_seed"
        
    print(f"Loading {model_variant} 3-model ensemble from {ckpt_dir} onto {device}...")
    for s in seeds:
        ckpt_path = ckpt_dir / f"{prefix}{s}.pt"
        if not ckpt_path.exists():
            # Fallback to Phase 10R if Phase 16 still training
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
