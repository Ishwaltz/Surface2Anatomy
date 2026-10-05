import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES
from sharon.model_gnn import PointNet2Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 80)
    print("PART B: FORENSIC EIGHT-PATIENT OVERFIT INVESTIGATION")
    print("=" * 80)

    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"

    data = torch.load(str(pt_path), weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)

    # 8 fixed train patients
    d0_idx = np.array(splits["train_indices"][:8])
    case_ids = [data["case_ids"][i] for i in d0_idx]
    print(f"8 Fixed Patients: {case_ids}")

    pts_m = torch.from_numpy(data["points_model"].numpy()[d0_idx]).float().to(device) # (8, 4096, 3)
    tgt_m = torch.from_numpy(data["targets_model"].numpy()[d0_idx]).float().to(device) # (8, 121, 3)
    p_mask = torch.from_numpy(data["target_primary_mask"].numpy()[d0_idx]).float().to(device) # (8, 121)
    S_global = float(data["s_global"])

    # -------------------------------------------------------------
    # B6: D0_ID_ORACLE (One-hot patient ID -> MLP -> 121*3)
    # -------------------------------------------------------------
    print("\n--- Running B6: D0_ID_ORACLE (Pipeline Diagnostic) ---")
    one_hot = torch.eye(8, device=device) # (8, 8)
    
    oracle_mlp = nn.Sequential(
        nn.Linear(8, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 121 * 3)
    ).to(device)
    
    opt_oracle = torch.optim.Adam(oracle_mlp.parameters(), lr=5e-3)
    for epoch in range(300):
        oracle_mlp.train()
        opt_oracle.zero_grad()
        out = oracle_mlp(one_hot).view(8, 121, 3)
        diff_sq = torch.sum((out - tgt_m)**2, dim=-1)
        loss = (torch.sqrt(diff_sq + 1e-6) * p_mask).sum() / p_mask.sum()
        loss.backward()
        opt_oracle.step()

    oracle_mlp.eval()
    with torch.no_grad():
        out_eval = oracle_mlp(one_hot).view(8, 121, 3)
        err_mm = torch.norm((out_eval - tgt_m) * S_global, dim=-1)
        m_bool = p_mask > 0.5
        oracle_mre = err_mm[m_bool].mean().item()

    print(f"  D0_ID_ORACLE Training MRE: {oracle_mre:.6f} mm")
    oracle_pass = oracle_mre < 0.5
    print(f"  D0_ID_ORACLE Status: {'PASS (< 0.5 mm)' if oracle_pass else 'FAIL'}")
    assert oracle_pass, "Pipeline itself is broken!"

    # -------------------------------------------------------------
    # B2 & B3: PointNet++ Head Normalization & Dropout Diagnostic
    # -------------------------------------------------------------
    print("\n--- Running B2-B7: PointNet++ Architecture & Normalization Variants ---")

    # Variant 1: D0A - PointNet++ with BatchNorm in head + dropout 0.1 (Phase 2 setting)
    class ModelD0A(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = PointNet2Encoder(in_channel=3, out_dim=1024)
            self.head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 121 * 3)
            )
        def forward(self, x):
            return self.head(self.encoder(x)).view(-1, 121, 3)

    # Variant 2: D0B - PointNet++ with LayerNorm in head and NO dropout
    class ModelD0B(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = PointNet2Encoder(in_channel=3, out_dim=1024)
            self.head = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 121 * 3)
            )
        def forward(self, x):
            return self.head(self.encoder(x)).view(-1, 121, 3)

    def train_and_eval_8(model, name, epochs=250, lr=1e-3):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            out = model(pts_m)
            diff_sq = torch.sum((out - tgt_m)**2, dim=-1)
            loss = (torch.sqrt(diff_sq + 1e-6) * p_mask).sum() / p_mask.sum()
            loss.backward()
            opt.step()

        # Check error in eval mode AND train mode
        model.eval()
        with torch.no_grad():
            out_eval = model(pts_m)
            err_eval = torch.norm((out_eval - tgt_m) * S_global, dim=-1)[p_mask > 0.5].mean().item()
        
        # In train mode (without eval batchnorm running stats)
        model.train()
        with torch.no_grad():
            out_tr = model(pts_m)
            err_tr = torch.norm((out_tr - tgt_m) * S_global, dim=-1)[p_mask > 0.5].mean().item()

        print(f"  {name}: Eval MRE = {err_eval:.2f} mm | Train Mode MRE = {err_tr:.2f} mm")
        return err_eval, err_tr

    torch.manual_seed(42)
    m_d0a = ModelD0A().to(device)
    eval_a, tr_a = train_and_eval_8(m_d0a, "D0A (BatchNorm head + Dropout 0.1)")

    torch.manual_seed(42)
    m_d0b = ModelD0B().to(device)
    eval_b, tr_b = train_and_eval_8(m_d0b, "D0B (LayerNorm head, No Dropout, 1024-dim)")

    # Save diagnostic results
    debug_res = {
        "D0_ID_ORACLE_mre": oracle_mre,
        "D0_ID_ORACLE_pass": oracle_pass,
        "D0A_eval_mre": eval_a,
        "D0A_train_mre": tr_a,
        "D0B_eval_mre": eval_b,
        "D0B_train_mre": tr_b,
        "case_ids": case_ids
    }
    with open(repo_root / "experiments" / "phase3" / "d0_overfit_debug.json", "w") as f:
        json.dump(debug_res, f, indent=2)

    print(f"\n[Done] Part B complete and saved.")

if __name__ == "__main__":
    main()
