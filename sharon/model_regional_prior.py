import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionalContextPooling(nn.Module):
    """
    Learned attention pooling over a region's target query features.
    Maps H_r in R^{B x K_r x D} to c_r in R^{B x D}.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, h_region: torch.Tensor) -> torch.Tensor:
        # h_region: (B, K_r, d_model)
        attn_scores = self.attn_mlp(h_region) # (B, K_r, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, K_r, 1)
        pooled = torch.sum(attn_weights * h_region, dim=1) # (B, d_model)
        return pooled


class SkeletalAnchorEncoder(nn.Module):
    """
    Encodes predicted skeletal coordinates S_pred in R^{B x K_skel x 3}
    into a compact global skeletal frame representation e_skel in R^{B x d_skel}.
    """
    def __init__(self, num_skel_targets: int = 39, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        in_dim = num_skel_targets * 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, s_pred: torch.Tensor) -> torch.Tensor:
        # s_pred: (B, K_skel, 3) in mm -> normalize to ~500mm scale
        B = s_pred.shape[0]
        s_flat = (s_pred / 500.0).reshape(B, -1)
        return self.mlp(s_flat) # (B, out_dim)


class RegionalResidualPriorModel(nn.Module):
    """
    Phase 7 Regional Residual Prior Architecture.
    Combines:
    1. Regional context pooling over Phase 4 target queries (c_r in R^D)
    2. Optional predicted skeletal context (e_skel in R^128) with skeletal dropout
    3. Regional latent predictors f_r(c_r) -> z_hat_r in R^{d_r}
    4. Structured regional residual correction: Delta P_r = U_r z_hat_r
    5. Conservative per-target gating: P_final = P_base + g * Delta P
    """
    def __init__(
        self,
        regions: Dict[str, List[int]],         # region_name -> list of local target indices
        regional_bases: Dict[str, torch.Tensor], # region_name -> U_r in R^{3K_r x d_r}
        regional_dims: Dict[str, int],          # region_name -> d_r
        d_model: int = 256,
        use_skeletal_conditioning: bool = False,
        use_gating: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.regions = regions
        self.regional_dims = regional_dims
        self.use_skeletal_conditioning = use_skeletal_conditioning
        self.use_gating = use_gating

        # Register regional bases as frozen buffers
        for r_name, basis in regional_bases.items():
            self.register_buffer(f"basis_{r_name}", basis.clone().float())

        # 1. Regional context poolers
        self.poolers = nn.ModuleDict({
            r_name: RegionalContextPooling(d_model=d_model)
            for r_name in regions
        })

        # 2. Skeletal Encoder (if conditioned)
        if use_skeletal_conditioning:
            num_skel = len(regions.get("skeletal", []))
            self.skel_encoder = SkeletalAnchorEncoder(num_skel_targets=num_skel, out_dim=128)
            self.skel_dropout = nn.Dropout(p=0.2)

        # 3. Regional latent prediction heads
        self.latent_heads = nn.ModuleDict()
        for r_name in regions:
            d_r = regional_dims[r_name]
            in_dim = d_model
            # For soft-tissue regions, append skeletal context if enabled
            if use_skeletal_conditioning and r_name != "skeletal":
                in_dim += 128 # skeletal context dimension

            self.latent_heads[r_name] = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, d_r)
            )

        # 4. Conservative Target-Specific Gate
        if use_gating:
            # Inputs: h_k (d_model), delta_p_k (3), diff_qv_k (3) -> d_model + 6
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_model + 6, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            # Initialize final gate bias to -1.5 -> initial gate sigmoid(-1.5) ~ 0.18
            nn.init.constant_(self.gate_mlp[-2].bias, -1.5)

    def forward(
        self,
        queries: torch.Tensor,                      # (B, K, d_model)
        p_base: torch.Tensor,                       # (B, K, 3) in mm
        p_query: Optional[torch.Tensor] = None,     # (B, K, 3) in mm
        p_vote: Optional[torch.Tensor] = None,      # (B, K, 3) in mm
        s_pred: Optional[torch.Tensor] = None       # (B, K_skel, 3) in mm
    ) -> Dict[str, torch.Tensor]:
        B, K, _ = queries.shape
        device = queries.device

        # 1. Encode predicted skeletal frame if enabled
        e_skel = None
        if self.use_skeletal_conditioning and s_pred is not None:
            e_skel_raw = self.skel_encoder(s_pred)
            if self.training:
                e_skel = self.skel_dropout(e_skel_raw)
            else:
                e_skel = e_skel_raw

        z_hat_by_region = {}
        delta_p_full = torch.zeros(B, K, 3, device=device)

        # 2. Predict regional latent codes & structured corrections
        for r_name, local_indices in self.regions.items():
            h_r = queries[:, local_indices, :] # (B, K_r, d_model)
            c_r = self.poolers[r_name](h_r)   # (B, d_model)

            if self.use_skeletal_conditioning and r_name != "skeletal" and e_skel is not None:
                trunk_in = torch.cat([c_r, e_skel], dim=-1)
            else:
                trunk_in = c_r

            z_hat_r = self.latent_heads[r_name](trunk_in) # (B, d_r)
            z_hat_by_region[r_name] = z_hat_r

            # Basis reconstruction: U_r is (3K_r, d_r)
            U_r = getattr(self, f"basis_{r_name}") # (3K_r, d_r)
            y_r = torch.matmul(z_hat_r, U_r.t())   # (B, 3K_r)
            delta_p_r = y_r.view(B, len(local_indices), 3) # (B, K_r, 3) in mm
            delta_p_full[:, local_indices, :] = delta_p_r

        # 3. Gated Fusion
        if self.use_gating:
            if p_query is not None and p_vote is not None:
                diff_qv = p_vote - p_query
            else:
                diff_qv = torch.zeros_like(p_base)

            gate_in = torch.cat([queries, delta_p_full, diff_qv], dim=-1) # (B, K, d_model + 6)
            gate = self.gate_mlp(gate_in) # (B, K, 1) in [0, 1]
            p_final = p_base + gate * delta_p_full
        else:
            gate = torch.ones(B, K, 1, device=device)
            p_final = p_base + delta_p_full

        return {
            "p_final": p_final,
            "delta_p": delta_p_full,
            "gate": gate,
            "z_hat": z_hat_by_region
        }
