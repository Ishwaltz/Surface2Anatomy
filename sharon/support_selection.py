import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

def select_support_tokens(
    mem_xyz: torch.Tensor,             # (B, N_tokens, 3)
    mem_feat: torch.Tensor,            # (B, N_tokens, D)
    attn_weights: torch.Tensor,        # (B, K, N_tokens)
    vote_weights: Optional[torch.Tensor] = None, # (B, K, N_tokens) or None
    p0: Optional[torch.Tensor] = None, # (B, K, 3) coarse predictions in model space
    strategy: str = "combined",        # 'attention', 'voting', 'combined', 'axial_band'
    M: int = 64,
    k_anchors: int = 1,
    min_anchor_dist: float = 0.10,     # in model space (~50 mm)
    normals: Optional[torch.Tensor] = None # (B, N_tokens, 3)
) -> Dict[str, torch.Tensor]:
    """
    Selects target-specific external surface support tokens according to candidate strategy.
    Returns:
      - voter_xyz: (B, K, M, 3)
      - voter_feat: (B, K, M, D)
      - scores: (B, K, M, 1)
      - voter_normals: (B, K, M, 3) if normals provided, else None
    """
    B, N_tokens, D = mem_feat.shape
    K = attn_weights.shape[1]
    M_actual = min(M, N_tokens)

    # 1. Compute candidate token scores per target
    # Normalize attention weights: (B, K, N_tokens)
    a_norm = attn_weights / (attn_weights.max(dim=-1, keepdim=True)[0].clamp(min=1e-8))

    if vote_weights is not None:
        v_norm = vote_weights / (vote_weights.max(dim=-1, keepdim=True)[0].clamp(min=1e-8))
    else:
        v_norm = a_norm

    if strategy == "attention":
        token_scores = a_norm
    elif strategy == "voting":
        token_scores = v_norm
    elif strategy == "combined":
        token_scores = 0.5 * (a_norm + v_norm)
    elif strategy == "axial_band":
        base_scores = 0.5 * (a_norm + v_norm)
        if p0 is not None:
            # Distance along z-axis (axial): |z_j - z_k^0|
            # mem_xyz: (B, 1, N, 3), p0: (B, K, 1, 3)
            diff_z = torch.abs(mem_xyz.unsqueeze(1)[:, :, :, 2] - p0.unsqueeze(2)[:, :, :, 2]) # (B, K, N)
            # Soft axial band mask: exp(-(diff_z / 0.15)^2)
            axial_weight = torch.exp(-((diff_z / 0.15) ** 2))
            token_scores = base_scores * (0.5 + 0.5 * axial_weight)
        else:
            token_scores = base_scores
    else:
        token_scores = a_norm

    # 2. Multi-Anchor or Single-Anchor Selection
    if k_anchors <= 1:
        # Standard top-M selection
        top_scores, top_idx = torch.topk(token_scores, k=M_actual, dim=-1) # (B, K, M)
    else:
        # Multi-anchor selection: select K_anchor diverse anchor centers, then allocate M/K_anchor tokens per anchor
        m_per_anchor = max(1, M_actual // k_anchors)
        top_idx_list = []
        top_scores_list = []

        # Iterate over batch and target for multi-anchor suppression
        # (Optimized tensor formulation)
        cand_scores = token_scores.clone() # (B, K, N)
        selected_indices = []

        for a_idx in range(k_anchors):
            # Best available anchor
            best_s, best_i = torch.topk(cand_scores, k=m_per_anchor, dim=-1) # (B, K, m_per_anchor)
            selected_indices.append(best_i)
            
            # Suppress tokens around the anchor center for next anchor
            anchor_pos = torch.gather(
                mem_xyz.unsqueeze(1).expand(-1, K, -1, -1),
                2,
                best_i[:, :, :1].unsqueeze(-1).expand(-1, -1, -1, 3)
            ) # (B, K, 1, 3)
            
            # Distance from all tokens to this anchor: (B, K, N)
            dist_to_anchor = torch.norm(
                mem_xyz.unsqueeze(1) - anchor_pos,
                dim=-1
            )
            cand_scores[dist_to_anchor < min_anchor_dist] = -1e9

        top_idx = torch.cat(selected_indices, dim=-1)[:, :, :M_actual] # (B, K, M_actual)
        top_scores = torch.gather(token_scores, 2, top_idx)

    # 3. Gather Coordinates, Features, and Scores
    top_idx_xyz = top_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    top_idx_feat = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)

    mem_xyz_exp = mem_xyz.unsqueeze(1).expand(-1, K, -1, -1)
    mem_feat_exp = mem_feat.unsqueeze(1).expand(-1, K, -1, -1)

    voter_xyz = torch.gather(mem_xyz_exp, 2, top_idx_xyz) # (B, K, M, 3)
    voter_feat = torch.gather(mem_feat_exp, 2, top_idx_feat) # (B, K, M, D)
    scores = top_scores.unsqueeze(-1) # (B, K, M, 1)

    voter_normals = None
    if normals is not None:
        mem_norm_exp = normals.unsqueeze(1).expand(-1, K, -1, -1)
        voter_normals = torch.gather(mem_norm_exp, 2, top_idx_xyz)

    return {
        "voter_xyz": voter_xyz,
        "voter_feat": voter_feat,
        "scores": scores,
        "voter_normals": voter_normals,
        "indices": top_idx
    }
