import sys
import json
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from sharon.model_voting import TargetConditionedVotingModel
from sharon.model_refiner import TargetSpecificSurfaceRefiner, SurfaceRefinementPipeline
from tools.phase3.run_phase3_experiments import Phase3Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 80)
    print("PHASE 5 - STAGE 28: EIGHT-PATIENT MEMORIZATION GATE (GATE A)")
    print("=" * 80)

    # 1. Load Data
    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
    ckpt_path = repo_root / "experiments" / "phase5" / "r0_phase4_base_seed42.pt"
    base_results = json.load(open(repo_root / "experiments" / "phase5" / "step1_baseline_results.json"))

    data = torch.load(str(pt_path), weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)

    d0_idx = splits["train_indices"][:8]
    pts_m = torch.from_numpy(data["points_model"].numpy()[d0_idx]).float().to(device)
    tgt_m = torch.from_numpy(data["targets_model"].numpy()[d0_idx]).float().to(device)
    p_mask = torch.from_numpy(data["target_primary_mask"].numpy()[d0_idx]).float().to(device)
    S_global = float(data["s_global"])

    body_dims = data["body_dimensions_mm"].numpy()[d0_idx] / 500.0
    sex = data["sex"].numpy()[d0_idx].astype("float32")
    meta = torch.from_numpy(np.column_stack([body_dims, np.zeros((8, 2)), sex])).float().to(device)

    ckpt = torch.load(str(ckpt_path), weights_only=False)
    base_model = TargetConditionedVotingModel(atlas_coords=ckpt["atlas_coords"]).to(device)
    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()

    # Freeze BatchNorm in base model
    for m in base_model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.track_running_stats = False

    # Residual bounds from step 1
    R_bounds_mm = torch.tensor(base_results["R_k_p95_mm"]).float().to(device)

    refiner = TargetSpecificSurfaceRefiner(d_model=256, d_ref=128, nhead=4, num_layers=2, use_gating=False).to(device)
    pipeline = SurfaceRefinementPipeline(base_model, refiner, R_bounds_mm=R_bounds_mm, S_global=S_global, M=64).to(device)

    opt = torch.optim.AdamW(refiner.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)

    # Initial error before refinement
    with torch.no_grad():
        init_out = pipeline(pts_m, metadata=meta, freeze_base=True)
        init_err = torch.norm((init_out["p0"] - tgt_m) * S_global, dim=-1)[p_mask > 0.5].mean().item()
    print(f"Initial 8-patient coarse MRE (p0): {init_err:.2f} mm")

    print("Training refiner on 8 patients for 400 epochs...")
    for epoch in range(400):
        refiner.train()
        opt.zero_grad()
        out = pipeline(pts_m, metadata=meta, freeze_base=True)
        p1 = out["p1"]
        loss = (torch.norm(p1 - tgt_m, dim=-1) * p_mask).sum() / p_mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        opt.step()
        sched.step()

    refiner.eval()
    with torch.no_grad():
        final_out = pipeline(pts_m, metadata=meta, freeze_base=True)
        final_err = torch.norm((final_out["p1"] - tgt_m) * S_global, dim=-1)[p_mask > 0.5].mean().item()

    print(f"\nFinal 8-patient refined MRE (p1): {final_err:.2f} mm")
    if final_err < 3.0:
        print(">>> GATE A PASSED: 8-patient MRE < 3.0 mm! <<<")
    else:
        print(">>> GATE A FAILED: 8-patient MRE >= 3.0 mm! <<<")

    gate_a_res = {
        "init_p0_mre_mm": init_err,
        "final_p1_mre_mm": final_err,
        "gate_a_passed": bool(final_err < 3.0)
    }
    with open(repo_root / "experiments" / "phase5" / "gate_a_results.json", "w") as f:
        json.dump(gate_a_res, f, indent=2)

if __name__ == "__main__":
    main()
