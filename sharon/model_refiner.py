import math
from typing import Optional, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sharon.model_voting import TargetConditionedVotingModel
from sharon.support_selection import select_support_tokens

class TargetSpecificSurfaceRefiner(nn.Module):
    """
    Phase 5 Local Surface Refiner:
    Processes target-specific support tokens in target-centered local frame
    and predicts bounded residual Delta p_k.
    """
    def __init__(
        self,
        d_model: int = 256,
        d_ref: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_organs: int = 121,
        use_normals: bool = False,
        use_gating: bool = False,
        dropout: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ref = d_ref
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_organs = num_organs
        self.use_normals = use_normals
        self.use_gating = use_gating

        in_dim = d_model * 2 + 3 + 1 + 3 + 1
        if use_normals:
            in_dim += 3

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_ref),
            nn.LayerNorm(d_ref),
            nn.ReLU(),
            nn.Linear(d_ref, d_ref)
        )

        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(d_ref, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_ref) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_ref, d_ref * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ref * 2, d_ref)
            )
            for _ in range(num_layers)
        ])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_ref) for _ in range(num_layers)])

        self.query_proj_pool = nn.Linear(d_model, d_ref)
        self.token_proj_pool = nn.Linear(d_ref, d_ref)

        self.residual_mlp = nn.Sequential(
            nn.Linear(d_ref, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        if use_gating:
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_ref + 2, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Sigmoid()
            )

    def forward(
        self,
        p0: torch.Tensor,
        h_queries: torch.Tensor,
        atlas_coords: torch.Tensor,
        voter_xyz: torch.Tensor,
        voter_feat: torch.Tensor,
        scores: torch.Tensor,
        R_bounds: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        d_qv: Optional[torch.Tensor] = None,
        dispersion: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, K, M, _ = voter_xyz.shape

        p0_exp = p0.unsqueeze(2)
        r_jk = voter_xyz - p0_exp
        r_dist = torch.norm(r_jk, dim=-1, keepdim=True)

        a_exp = atlas_coords.unsqueeze(0).unsqueeze(2).expand(B, K, M, 3)
        rel_atlas = voter_xyz - a_exp

        q_exp = h_queries.unsqueeze(2).expand(B, K, M, self.d_model)

        feat_list = [voter_feat, q_exp, r_jk, r_dist, rel_atlas, scores]
        if self.use_normals and normals is not None:
            feat_list.append(normals)

        raw_in = torch.cat(feat_list, dim=-1)
        BK = B * K
        tokens = self.input_proj(raw_in.view(BK, M, -1))

        for i in range(self.num_layers):
            t_self, _ = self.self_attns[i](tokens, tokens, tokens)
            tokens = self.norms1[i](tokens + t_self)
            tokens = self.norms2[i](tokens + self.ffns[i](tokens))

        q_pool = self.query_proj_pool(h_queries.view(BK, 1, self.d_model))
        k_pool = self.token_proj_pool(tokens)

        pool_logits = torch.bmm(q_pool, k_pool.transpose(1, 2)) / math.sqrt(self.d_ref)
        pool_weights = F.softmax(pool_logits, dim=-1)

        r_k = torch.bmm(pool_weights, tokens).squeeze(1).view(B, K, self.d_ref)

        z_k = self.residual_mlp(r_k)
        delta_p = R_bounds.unsqueeze(0) * torch.tanh(z_k)

        if self.use_gating:
            gate_feats = [r_k]
            if d_qv is not None:
                gate_feats.append(d_qv)
            else:
                gate_feats.append(torch.zeros(B, K, 1, device=p0.device))
            if dispersion is not None:
                gate_feats.append(dispersion)
            else:
                gate_feats.append(torch.zeros(B, K, 1, device=p0.device))
            gate_in = torch.cat(gate_feats, dim=-1)
            gate = self.gate_mlp(gate_in)
            p1 = p0 + gate * delta_p
        else:
            gate = torch.ones_like(delta_p)
            p1 = p0 + delta_p

        return {
            "p1": p1,
            "delta_p": delta_p,
            "gate": gate,
            "r_k": r_k,
            "pool_weights": pool_weights.view(B, K, M)
        }

class SurfaceRefinementPipeline(nn.Module):
    """
    End-to-end Phase 5 pipeline connecting the reproduced Phase 4 Base
    and the Target-Specific Surface Refiner.
    """
    def __init__(
        self,
        base_model: TargetConditionedVotingModel,
        refiner: TargetSpecificSurfaceRefiner,
        R_bounds_mm: torch.Tensor,      # (121,) in physical mm
        S_global: float = 500.0,
        support_strategy: str = "combined",
        M: int = 64,
        k_anchors: int = 1
    ):
        super().__init__()
        self.base_model = base_model
        self.refiner = refiner
        self.S_global = S_global
        self.support_strategy = support_strategy
        self.M = M
        self.k_anchors = k_anchors

        # Convert R_bounds from mm to model coordinate space: R_m = R_mm / S_global
        R_bounds_m = (R_bounds_mm / S_global).float().view(-1, 1)
        self.register_buffer("R_bounds_m", R_bounds_m)
        self.register_buffer("atlas_coords", base_model.atlas_coords.clone())

    def forward(
        self,
        pts: torch.Tensor,                              # (B, 4096, 3)
        metadata: Optional[torch.Tensor] = None,        # (B, 6)
        raw_normals: Optional[torch.Tensor] = None,     # (B, 4096, 3)
        freeze_base: bool = True,
        control_shuffled_support: bool = False,         # D2: Shuffled support
        control_random_support: bool = False,           # D3: Random support
        control_wrong_target_support: bool = False      # D4: Wrong target support
    ) -> Dict[str, torch.Tensor]:
        B = pts.shape[0]
        K = self.base_model.num_organs

        # 1. Base Model Forward Pass
        if freeze_base:
            with torch.no_grad():
                base_out = self.base_model(pts, metadata=metadata)
        else:
            base_out = self.base_model(pts, metadata=metadata)

        p0 = base_out["p_final"]            # (B, K, 3)
        h_queries = base_out["queries"]     # (B, K, d_model)
        mem_xyz = base_out["mem_xyz"]       # (B, 320, 3)
        mem_feat = base_out["mem_tokens"]   # (B, 320, d_model)
        attn_w = base_out["last_attn_weights"] # (B, K, 320)
        disp = base_out["dispersion"].unsqueeze(-1) # (B, K, 1)

        d_qv = torch.norm(base_out["p_query"] - base_out["p_vote"], dim=-1, keepdim=True) # (B, K, 1)

        # 2. Match Normals to Memory Tokens (if provided)
        mem_normals = None
        if raw_normals is not None and self.refiner.use_normals:
            # Find nearest neighbor in pts for each mem_xyz point
            diff = mem_xyz.unsqueeze(2) - pts.unsqueeze(1) # (B, 320, 4096, 3)
            nn_idx = torch.argmin(torch.norm(diff, dim=-1), dim=-1) # (B, 320)
            nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, 3)
            mem_normals = torch.gather(raw_normals, 1, nn_idx_exp) # (B, 320, 3)

        # 3. Select Support Tokens
        support_out = select_support_tokens(
            mem_xyz=mem_xyz,
            mem_feat=mem_feat,
            attn_weights=attn_w,
            vote_weights=None,
            p0=p0,
            strategy=self.support_strategy,
            M=self.M,
            k_anchors=self.k_anchors,
            normals=mem_normals
        )
        v_xyz = support_out["voter_xyz"]     # (B, K, M, 3)
        v_feat = support_out["voter_feat"]   # (B, K, M, D)
        scores = support_out["scores"]       # (B, K, M, 1)
        v_norm = support_out["voter_normals"]

        # 4. Apply Forensic Controls if Requested
        if control_shuffled_support:
            # D2: Swap support tokens across patients in batch
            perm_p = torch.randperm(B, device=pts.device)
            v_xyz = v_xyz[perm_p]
            v_feat = v_feat[perm_p]
            scores = scores[perm_p]
            if v_norm is not None:
                v_norm = v_norm[perm_p]

        if control_random_support:
            # D3: Random surface tokens
            N_tokens = mem_xyz.shape[1]
            rand_idx = torch.randint(0, N_tokens, (B, K, self.M), device=pts.device)
            rand_idx_xyz = rand_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            rand_idx_feat = rand_idx.unsqueeze(-1).expand(-1, -1, -1, self.base_model.d_model)

            v_xyz = torch.gather(mem_xyz.unsqueeze(1).expand(-1, K, -1, -1), 2, rand_idx_xyz)
            v_feat = torch.gather(mem_feat.unsqueeze(1).expand(-1, K, -1, -1), 2, rand_idx_feat)
            scores = torch.full_like(scores, 1.0 / self.M)
            if v_norm is not None:
                v_norm = torch.gather(mem_normals.unsqueeze(1).expand(-1, K, -1, -1), 2, rand_idx_xyz)

        if control_wrong_target_support:
            # D4: Permute target support sets across targets
            perm_t = torch.randperm(K, device=pts.device)
            v_xyz = v_xyz[:, perm_t]
            v_feat = v_feat[:, perm_t]
            scores = scores[:, perm_t]
            if v_norm is not None:
                v_norm = v_norm[:, perm_t]

        # 5. Local Surface Refiner Forward Pass
        ref_out = self.refiner(
            p0=p0,
            h_queries=h_queries,
            atlas_coords=self.atlas_coords,
            voter_xyz=v_xyz,
            voter_feat=v_feat,
            scores=scores,
            R_bounds=self.R_bounds_m,
            normals=v_norm,
            d_qv=d_qv,
            dispersion=disp
        )

        return {
            "p1": ref_out["p1"],
            "p0": p0,
            "delta_p": ref_out["delta_p"],
            "gate": ref_out["gate"],
            "r_k": ref_out["r_k"],
            "voter_xyz": v_xyz,
            "scores": scores
        }
