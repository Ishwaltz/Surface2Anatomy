import math
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from sharon.model_target_query import MultiScaleSurfacePointNet2Encoder

class TargetConditionedVotingModel(nn.Module):
    """
    Target-Conditioned Point-Wise Surface Voting with Gated Fusion (Phase 4).
    1. Extracts multi-scale surface tokens from PointNet++.
    2. Runs target-query cross-attention with geometric bias and target self-attention.
    3. Produces direct query prediction p_query.
    4. Selects candidate surface tokens via attention weights.
    5. Predicts offsets and confidence logits per (target, surface token) pair.
    6. Aggregates proposals via confidence weights to produce p_vote.
    7. Fuses p_query and p_vote via learned target-specific fusion gate g_k.
    """
    def __init__(
        self,
        atlas_coords: torch.Tensor,       # (121, 3) in model coordinates
        num_organs: int = 121,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        top_m: int = 64,                  # Number of surface voter candidates per target
        voting_mode: str = "gated",       # 'uniform', 'confidence', 'gated'
        use_metadata: bool = True,
        use_geo_bias: bool = True,
        use_self_attn: bool = True,
        dropout: float = 0.05
    ):
        super().__init__()
        self.num_organs = num_organs
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.top_m = top_m
        self.voting_mode = voting_mode
        self.use_metadata = use_metadata
        self.use_geo_bias = use_geo_bias
        self.use_self_attn = use_self_attn

        self.register_buffer("atlas_coords", atlas_coords.clone().float())

        # 1. Surface Encoder & Projections
        self.encoder = MultiScaleSurfacePointNet2Encoder(in_channel=3, out_dim=1024)
        self.proj_l2 = nn.Linear(256, d_model)
        self.proj_l3 = nn.Linear(512, d_model)
        self.scale_embed = nn.Embedding(2, d_model)
        self.pe_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # 2. Target Queries & Metadata
        self.target_embed = nn.Embedding(num_organs, d_model)
        self.atlas_pe_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        if use_metadata:
            self.meta_mlp = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, d_model)
            )

        if use_geo_bias:
            self.geo_bias_mlp = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, nhead)
            )

        # 3. Transformer Layers
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        if use_self_attn:
            self.self_attns = nn.ModuleList([
                nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ])
            self.norms_self = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            )
            for _ in range(num_layers)
        ])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # 4. Direct Query Head
        self.query_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        # 5. Shared Target-Conditioned Voting Head
        # Inputs: f_j (d_model), h_k (d_model), (x_j - a_k) (3), (x_j - p_query) (3) -> 2*d_model + 6
        pair_in_dim = d_model * 2 + 6
        self.voting_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.offset_head = nn.Linear(128, 3)
        self.confidence_head = nn.Linear(128, 1)

        # 6. Gated Fusion MLP
        # Inputs: h_k (d_model), (p_vote - p_query) (3), diagnostics [dispersion, n_eff, entropy] (3) -> d_model + 6
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(d_model + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 3), # per-axis gate in [0, 1]
            nn.Sigmoid()
        )

    def forward(
        self,
        pts: torch.Tensor,                          # (B, 4096, 3)
        metadata: Optional[torch.Tensor] = None,    # (B, 6)
        query_perm: Optional[torch.Tensor] = None,  # (121,) optional diagnostic
        token_shuffle: bool = False,                # Diagnostic D3
        random_voters: bool = False,                # Diagnostic D4
        shuffle_weights: bool = False,              # Diagnostic D5
        zero_offsets: bool = False                  # Sanity control
    ) -> Dict[str, torch.Tensor]:
        B = pts.shape[0]
        K = self.num_organs
        M = self.top_m

        # 1. Extract and project surface tokens
        enc_out = self.encoder(pts)
        l2_xyz, l2_f = enc_out["l2_xyz"], enc_out["l2_feat"]
        l3_xyz, l3_f = enc_out["l3_xyz"], enc_out["l3_feat"]

        t_l2 = self.proj_l2(l2_f) + self.scale_embed.weight[0]
        t_l3 = self.proj_l3(l3_f) + self.scale_embed.weight[1]

        mem_xyz = torch.cat([l2_xyz, l3_xyz], dim=1) # (B, 320, 3)
        mem_f = torch.cat([t_l2, t_l3], dim=1)       # (B, 320, d_model)
        mem_tokens = mem_f + self.pe_mlp(mem_xyz)     # (B, 320, d_model)

        if token_shuffle:
            perm = torch.randperm(B, device=pts.device)
            mem_tokens = mem_tokens[perm]
            mem_xyz = mem_xyz[perm]

        # 2. Target Query Initialization
        q_idx = torch.arange(K, device=pts.device)
        if query_perm is not None:
            q_idx = q_idx[query_perm]

        target_emb = self.target_embed(q_idx)
        atlas_emb = self.atlas_pe_mlp(self.atlas_coords)
        queries = (target_emb + atlas_emb).unsqueeze(0).expand(B, -1, -1) # (B, K, d_model)

        if self.use_metadata and metadata is not None:
            queries = queries + self.meta_mlp(metadata).unsqueeze(1)

        # 3. Geometric Bias
        attn_bias = None
        if self.use_geo_bias:
            diff = mem_xyz.unsqueeze(1) - self.atlas_coords.unsqueeze(0).unsqueeze(2) # (B, K, 320, 3)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            r_feat = torch.cat([diff, dist], dim=-1)
            geo_b = self.geo_bias_mlp(r_feat)
            attn_bias = geo_b.permute(0, 3, 1, 2).reshape(B * self.nhead, K, -1)

        # 4. Cross-Attention
        last_attn_weights = None
        for i in range(self.num_layers):
            if self.use_self_attn:
                q_self, _ = self.self_attns[i](queries, queries, queries)
                queries = self.norms_self[i](queries + q_self)

            q_cross, attn_w = self.cross_attns[i](
                queries, mem_tokens, mem_tokens,
                attn_mask=attn_bias,
                need_weights=True
            )
            queries = self.norms1[i](queries + q_cross)
            queries = self.norms2[i](queries + self.ffns[i](queries))
            last_attn_weights = attn_w # (B, K, 320)

        # 5. Direct Query Prediction
        delta_p = self.query_head(queries) # (B, K, 3)
        p_query = self.atlas_coords.unsqueeze(0) + delta_p # (B, K, 3)

        # 6. Candidate Surface Token Selection
        # last_attn_weights: (B, K, 320)
        N_tokens = mem_tokens.shape[1] # 320
        M_actual = min(M, N_tokens)
        
        if random_voters:
            cand_idx = torch.randint(0, N_tokens, (B, K, M_actual), device=pts.device)
        else:
            _, cand_idx = torch.topk(last_attn_weights, k=M_actual, dim=-1) # (B, K, M)

        # Gather candidate surface coordinates and features
        # cand_idx: (B, K, M) -> expand to gather from mem_xyz (B, 320, 3) and mem_tokens (B, 320, D)
        cand_idx_xyz = cand_idx.unsqueeze(-1).expand(-1, -1, -1, 3) # (B, K, M, 3)
        cand_idx_feat = cand_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model) # (B, K, M, D)
        
        mem_xyz_exp = mem_xyz.unsqueeze(1).expand(-1, K, -1, -1) # (B, K, 320, 3)
        mem_feat_exp = mem_tokens.unsqueeze(1).expand(-1, K, -1, -1) # (B, K, 320, D)
        
        voter_xyz = torch.gather(mem_xyz_exp, 2, cand_idx_xyz) # (B, K, M, 3)
        voter_feat = torch.gather(mem_feat_exp, 2, cand_idx_feat) # (B, K, M, D)

        # Gather detached attention weights for proposal loss weighting
        cand_attn = torch.gather(last_attn_weights, 2, cand_idx) # (B, K, M)
        cand_attn_norm = cand_attn / cand_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # 7. Construct Pair Features for Voting Head
        # queries: (B, K, D) -> expand to (B, K, M, D)
        q_exp = queries.unsqueeze(2).expand(-1, -1, M_actual, -1)
        a_exp = self.atlas_coords.unsqueeze(0).unsqueeze(2).expand(B, -1, M_actual, -1)
        pq_exp = p_query.unsqueeze(2).expand(-1, -1, M_actual, -1)

        rel_atlas = voter_xyz - a_exp   # (B, K, M, 3)
        rel_query = voter_xyz - pq_exp  # (B, K, M, 3)

        pair_feat = torch.cat([voter_feat, q_exp, rel_atlas, rel_query], dim=-1) # (B, K, M, 2D+6)
        h_vote = self.voting_mlp(pair_feat) # (B, K, M, 128)

        # 8. Offset Proposals and Confidence Weights
        if zero_offsets:
            offsets = torch.zeros_like(voter_xyz)
        else:
            offsets = self.offset_head(h_vote) # (B, K, M, 3)

        v_proposals = voter_xyz + offsets # (B, K, M, 3)

        if self.voting_mode == "uniform":
            weights = torch.full((B, K, M_actual), 1.0 / M_actual, device=pts.device)
        else:
            conf_logits = self.confidence_head(h_vote).squeeze(-1) # (B, K, M)
            weights = F.softmax(conf_logits, dim=-1) # (B, K, M)

        if shuffle_weights:
            p_w = torch.randperm(M_actual, device=pts.device)
            weights = weights[:, :, p_w]

        # 9. Voting Aggregation
        p_vote = torch.sum(weights.unsqueeze(-1) * v_proposals, dim=2) # (B, K, 3)

        # 10. Voting Diagnostics
        # Dispersion D_k
        diff_prop = v_proposals - p_vote.unsqueeze(2) # (B, K, M, 3)
        dispersion = torch.sqrt(torch.sum(weights * torch.sum(diff_prop**2, dim=-1), dim=-1) + 1e-8) # (B, K)
        # Effective voters N_eff = 1 / sum(w^2)
        n_eff = 1.0 / torch.sum(weights**2, dim=-1).clamp(min=1e-8) # (B, K)
        # Entropy H_k = -sum(w * log(w))
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1) # (B, K)

        # 11. Gated Fusion
        diff_qv = p_vote - p_query # (B, K, 3)
        diag_feat = torch.stack([dispersion, n_eff / M_actual, entropy / math.log(M_actual)], dim=-1) # (B, K, 3)
        fusion_in = torch.cat([queries, diff_qv, diag_feat], dim=-1) # (B, K, D+6)
        
        if self.voting_mode == "gated":
            gate = self.fusion_gate_mlp(fusion_in) # (B, K, 3) in [0, 1]
            p_final = p_query + gate * diff_qv # (B, K, 3)
        elif self.voting_mode in ["uniform", "confidence"]:
            gate = torch.ones_like(p_query)
            p_final = p_vote
        else:
            gate = torch.zeros_like(p_query)
            p_final = p_query

        return {
            "p_final": p_final,
            "p_query": p_query,
            "p_vote": p_vote,
            "v_proposals": v_proposals,
            "weights": weights,
            "cand_attn_norm": cand_attn_norm.detach(),
            "gate": gate,
            "dispersion": dispersion,
            "n_eff": n_eff,
            "entropy": entropy,
            "mem_xyz": mem_xyz,
            "mem_tokens": mem_tokens,
            "last_attn_weights": last_attn_weights,
            "queries": queries
        }
