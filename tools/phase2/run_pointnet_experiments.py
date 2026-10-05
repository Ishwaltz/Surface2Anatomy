import sys
import json
import csv
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES
from sharon.model_gnn import PointNet2Encoder
from tools.phase2.metrics import compute_all_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Clean PointNet++ Models
class PointNet2PlainRegressor(nn.Module):
    def __init__(self, num_organs=121, dropout=0.1):
        super().__init__()
        self.encoder = PointNet2Encoder(in_channel=3, out_dim=1024)
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_organs * 3)
        )
    def forward(self, pts):
        f = self.encoder(pts) # (B, 1024)
        out = self.head(f)    # (B, 121*3)
        return out.view(-1, 121, 3)

class PointNet2ConditionedRegressor(nn.Module):
    def __init__(self, num_organs=121, cond_dim=2, dropout=0.1):
        super().__init__()
        self.encoder = PointNet2Encoder(in_channel=3, out_dim=1024)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Linear(1024 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_organs * 3)
        )
    def forward(self, pts, cond):
        f = self.encoder(pts)
        c = self.cond_mlp(cond)
        fc = torch.cat([f, c], dim=-1)
        out = self.head(fc)
        return out.view(-1, 121, 3)

# 2. Dataset
class V2Dataset(Dataset):
    def __init__(self, pts_m, tgt_m, masks, prim_masks, cond=None, augment=False):
        self.pts = torch.from_numpy(pts_m).float()
        self.tgt = torch.from_numpy(tgt_m).float()
        self.mask = torch.from_numpy(masks).float()
        self.prim = torch.from_numpy(prim_masks).float()
        self.cond = torch.from_numpy(cond).float() if cond is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        pts = self.pts[idx]
        if self.augment:
            # Minimal sensor jitter only (sigma = 0.005)
            noise = torch.randn_like(pts) * 0.005
            pts = pts + noise
        item = {
            "pts": pts,
            "tgt": self.tgt[idx],
            "mask": self.mask[idx],
            "prim": self.prim[idx]
        }
        if self.cond is not None:
            item["cond"] = self.cond[idx]
        return item

def train_pointnet(train_loader, val_loader, model, epochs=80, lr=5e-4, S_global=500.0, primary_indices=None, val_tgt_c=None, has_cond=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_macro_val = 999.0
    best_preds_mm = None
    train_history = []
    
    eps_sq = (1.0 / S_global) ** 2

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_valid = 0.0
        
        for b in train_loader:
            pts = b["pts"].to(device)
            tgt = b["tgt"].to(device)
            m = b["prim"].to(device) # Train strictly on genuine targets!
            
            opt.zero_grad()
            if has_cond:
                pred = model(pts, b["cond"].to(device))
            else:
                pred = model(pts)
                
            diff_sq = torch.sum((pred - tgt) ** 2, dim=-1)
            loss_mat = torch.sqrt(diff_sq + eps_sq)
            loss = (loss_mat * m).sum() / m.sum().clamp(min=1.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item() * len(pts)
            total_valid += len(pts)
            
        sched.step()

        # Validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for b in val_loader:
                pts = b["pts"].to(device)
                if has_cond:
                    pred = model(pts, b["cond"].to(device))
                else:
                    pred = model(pts)
                val_preds_list.append(pred.cpu().numpy())
                
        val_preds_m = np.concatenate(val_preds_list, axis=0)
        val_preds_mm = val_preds_m * S_global # In physical centered mm
        
        # Evaluate macro MRE on primary targets
        val_masks = val_loader.dataset.prim.numpy()
        mets = compute_all_metrics(val_preds_mm, val_tgt_c, val_masks, primary_indices)
        macro_val = mets["macro_target_mre"]
        
        train_mre_mm = (total_loss / total_valid) * S_global
        train_history.append((train_mre_mm, macro_val))
        
        if macro_val < best_macro_val:
            best_macro_val = macro_val
            best_preds_mm = val_preds_mm.copy()
            
    return best_macro_val, best_preds_mm, train_history

def main():
    print("=" * 80)
    print("STAGES 10-35: POINTNET++ BENCHMARKS & IDENTIFIABILITY ANALYSIS")
    print("=" * 80)

    # 1. Load Data
    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
    tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"

    data = torch.load(str(pt_path), weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)

    tr_idx = splits["train_indices"]
    val_idx = splits["val_indices"]

    with open(tiers_path) as f:
        primary_107_indices = [int(r["target_index"]) for r in csv.DictReader(f) if r["tier"] in ["TIER_A", "TIER_B"]]

    pts_m = data["points_model"].numpy()
    tgt_m = data["targets_model"].numpy()
    tgt_mask = data["target_mask"].numpy()
    prim_mask = data["target_primary_mask"].numpy()
    val_tgt_c = data["targets_centered_mm"].numpy()[val_idx]
    S_global = float(data["s_global"])

    # Metadata features
    sex = data["sex"].numpy() # 0 or 1
    sex_one_hot = np.zeros((len(sex), 2), dtype=np.float32)
    for i, s in enumerate(sex):
        sex_one_hot[i, int(s)] = 1.0

    body_dims = data["body_dimensions_mm"].numpy() # (440, 3)
    dim_norm = body_dims / 500.0 # Normalized around order 1

    # STAGE 17: D0_OVERFIT_8
    print("\n--- Stage 17: Eight-Patient Overfit Test (D0_OVERFIT_8) ---")
    d0_idx = tr_idx[:8]
    d0_ds = V2Dataset(pts_m[d0_idx], tgt_m[d0_idx], tgt_mask[d0_idx], prim_mask[d0_idx], augment=False)
    d0_loader = DataLoader(d0_ds, batch_size=8, shuffle=False)

    torch.manual_seed(42)
    d0_model = PointNet2PlainRegressor().to(device)
    d0_opt = torch.optim.AdamW(d0_model.parameters(), lr=1e-3, weight_decay=0.0)

    for epoch in range(150):
        d0_model.train()
        for b in d0_loader:
            pts = b["pts"].to(device)
            tgt = b["tgt"].to(device)
            m = b["prim"].to(device)
            d0_opt.zero_grad()
            pred = d0_model(pts)
            loss = (torch.sqrt(torch.sum((pred - tgt)**2, dim=-1) + 1e-4) * m).sum() / m.sum()
            loss.backward()
            d0_opt.step()

    d0_model.eval()
    with torch.no_grad():
        pred_8 = d0_model(torch.from_numpy(pts_m[d0_idx]).float().to(device)).cpu().numpy() * S_global
        tgt_8 = tgt_m[d0_idx] * S_global
        err_8 = np.linalg.norm(pred_8 - tgt_8, axis=-1)
        m_8 = prim_mask[d0_idx] > 0.5
        d0_mre = float(err_8[m_8].mean())

    print(f"  D0 Overfit 8-Patient Training MRE: {d0_mre:.2f} mm")
    d0_passed = d0_mre < 5.0
    print(f"  D0 Overfit Test Status: {'PASS (Memorization verified)' if d0_passed else 'FAIL'}")

    # STAGE 18: B7_POINTNETPP_PLAIN across 3 seeds (42, 43, 44)
    print("\n--- Stage 18: Full PointNet++ Training (B7_POINTNETPP_PLAIN across 3 seeds) ---")
    val_ds = V2Dataset(pts_m[val_idx], tgt_m[val_idx], tgt_mask[val_idx], prim_mask[val_idx], augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    b7_results = []
    b7_seed_preds = []

    for seed in [42, 43, 44]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr_ds = V2Dataset(pts_m[tr_idx], tgt_m[tr_idx], tgt_mask[tr_idx], prim_mask[tr_idx], augment=True)
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, drop_last=True)
        
        m_b7 = PointNet2PlainRegressor().to(device)
        best_macro, best_preds, hist = train_pointnet(
            tr_loader, val_loader, m_b7, epochs=70, lr=4e-4, S_global=S_global,
            primary_indices=primary_107_indices, val_tgt_c=val_tgt_c, has_cond=False
        )
        mets = compute_all_metrics(best_preds, val_tgt_c, prim_mask[val_idx], primary_107_indices)
        b7_results.append(mets)
        b7_seed_preds.append(best_preds)
        print(f"  Seed {seed}: Macro MRE = {mets['macro_target_mre']:.2f} mm | Micro MRE = {mets['micro_mre']:.2f} mm | SDR@10 = {mets['sdr_10']:.2f}%")

    b7_macro_scores = [m["macro_target_mre"] for m in b7_results]
    b7_micro_scores = [m["micro_mre"] for m in b7_results]
    b7_sdr10_scores = [m["sdr_10"] for m in b7_results]

    mean_b7_macro = float(np.mean(b7_macro_scores))
    std_b7_macro = float(np.std(b7_macro_scores))
    mean_b7_micro = float(np.mean(b7_micro_scores))
    std_b7_micro = float(np.std(b7_micro_scores))
    mean_b7_sdr10 = float(np.mean(b7_sdr10_scores))

    print(f"\n  B7 Overall (3 seeds): Macro Target MRE = {mean_b7_macro:.2f} ± {std_b7_macro:.2f} mm | Micro MRE = {mean_b7_micro:.2f} ± {std_b7_micro:.2f} mm | SDR@10 = {mean_b7_sdr10:.2f}%")

    # STAGE 19: B8_POINTNETPP_SEX (seed 42)
    print("\n--- Stage 19: PointNet++ + Sex Ablation (B8) ---")
    torch.manual_seed(42)
    tr_ds_sex = V2Dataset(pts_m[tr_idx], tgt_m[tr_idx], tgt_mask[tr_idx], prim_mask[tr_idx], cond=sex_one_hot[tr_idx], augment=True)
    val_ds_sex = V2Dataset(pts_m[val_idx], tgt_m[val_idx], tgt_mask[val_idx], prim_mask[val_idx], cond=sex_one_hot[val_idx], augment=False)
    tr_loader_sex = DataLoader(tr_ds_sex, batch_size=16, shuffle=True, drop_last=True)
    val_loader_sex = DataLoader(val_ds_sex, batch_size=16, shuffle=False)

    m_b8 = PointNet2ConditionedRegressor(cond_dim=2).to(device)
    _, best_preds_b8, _ = train_pointnet(
        tr_loader_sex, val_loader_sex, m_b8, epochs=70, lr=4e-4, S_global=S_global,
        primary_indices=primary_107_indices, val_tgt_c=val_tgt_c, has_cond=True
    )
    mets_b8 = compute_all_metrics(best_preds_b8, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    print(f"  B8 (+Sex) Macro Target MRE: {mets_b8['macro_target_mre']:.2f} mm | Micro MRE: {mets_b8['micro_mre']:.2f} mm | SDR@10: {mets_b8['sdr_10']:.2f}%")

    # STAGE 20: B9_POINTNETPP_BODYMETA (seed 42)
    print("\n--- Stage 20: PointNet++ + Body Dimensions Ablation (B9) ---")
    torch.manual_seed(42)
    tr_ds_dim = V2Dataset(pts_m[tr_idx], tgt_m[tr_idx], tgt_mask[tr_idx], prim_mask[tr_idx], cond=dim_norm[tr_idx], augment=True)
    val_ds_dim = V2Dataset(pts_m[val_idx], tgt_m[val_idx], tgt_mask[val_idx], prim_mask[val_idx], cond=dim_norm[val_idx], augment=False)
    tr_loader_dim = DataLoader(tr_ds_dim, batch_size=16, shuffle=True, drop_last=True)
    val_loader_dim = DataLoader(val_ds_dim, batch_size=16, shuffle=False)

    m_b9 = PointNet2ConditionedRegressor(cond_dim=3).to(device)
    _, best_preds_b9, _ = train_pointnet(
        tr_loader_dim, val_loader_dim, m_b9, epochs=70, lr=4e-4, S_global=S_global,
        primary_indices=primary_107_indices, val_tgt_c=val_tgt_c, has_cond=True
    )
    mets_b9 = compute_all_metrics(best_preds_b9, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    print(f"  B9 (+Dimensions) Macro Target MRE: {mets_b9['macro_target_mre']:.2f} mm | Micro MRE: {mets_b9['micro_mre']:.2f} mm | SDR@10: {mets_b9['sdr_10']:.2f}%")

    # Best neural predictions for diagnostics: B7 seed 42
    best_nn_preds = b7_seed_preds[0]

    # STAGE 22: PREDICTION VARIANCE COLLAPSE ANALYSIS
    print("\n--- Stage 22: Prediction Variance Collapse Analysis ---")
    # r_k = tr(Cov(pred_k)) / tr(Cov(gt_k))
    r_k_list = []
    val_masks = prim_mask[val_idx] > 0.5
    for k in primary_107_indices:
        k_m = val_masks[:, k]
        if k_m.sum() >= 5:
            p_k = best_nn_preds[k_m, k]
            g_k = val_tgt_c[k_m, k]
            cov_p = np.cov(p_k, rowvar=False)
            cov_g = np.cov(g_k, rowvar=False)
            tr_p = np.trace(cov_p) if cov_p.ndim == 2 else 0.0
            tr_g = np.trace(cov_g) if cov_g.ndim == 2 else 1e-4
            r_k = tr_p / max(tr_g, 1e-4)
            r_k_list.append(r_k)

    median_rk = float(np.median(r_k_list))
    mean_rk = float(np.mean(r_k_list))
    print(f"  Variance Ratio r_k (median): {median_rk:.4f}")
    print(f"  Variance Ratio r_k (mean):   {mean_rk:.4f}")
    collapse_present = median_rk < 0.35
    print(f"  Population-Mean Collapse Detected: {'YES' if collapse_present else 'NO'}")

    # STAGE 24: PATIENT MISMATCH CONTROL
    print("\n--- Stage 24: Patient Mismatch Control ---")
    # Permute predictions across patients
    rng = np.random.RandomState(42)
    perm_idx = rng.permutation(len(val_idx))
    mismatched_preds = best_nn_preds[perm_idx]
    mets_mismatch = compute_all_metrics(mismatched_preds, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    print(f"  Matched Macro MRE:    {b7_results[0]['macro_target_mre']:.2f} mm")
    print(f"  Mismatched Macro MRE: {mets_mismatch['macro_target_mre']:.2f} mm")
    diff_mismatch = mets_mismatch['macro_target_mre'] - b7_results[0]['macro_target_mre']
    print(f"  Degradation on Mismatch: +{diff_mismatch:.2f} mm")
    matched_better = diff_mismatch > 1.0

    # STAGE 25: SURFACE SHUFFLE CONTROL (D1_SHUFFLED_SURFACE)
    print("\n--- Stage 25: Surface Shuffle Control (D1) ---")
    tr_perm = rng.permutation(len(tr_idx))
    tr_ds_shuf = V2Dataset(pts_m[tr_idx[tr_perm]], tgt_m[tr_idx], tgt_mask[tr_idx], prim_mask[tr_idx], augment=True)
    tr_loader_shuf = DataLoader(tr_ds_shuf, batch_size=16, shuffle=True, drop_last=True)
    m_d1 = PointNet2PlainRegressor().to(device)
    _, best_preds_d1, _ = train_pointnet(
        tr_loader_shuf, val_loader, m_d1, epochs=50, lr=4e-4, S_global=S_global,
        primary_indices=primary_107_indices, val_tgt_c=val_tgt_c, has_cond=False
    )
    mets_d1 = compute_all_metrics(best_preds_d1, val_tgt_c, prim_mask[val_idx], primary_107_indices)
    print(f"  D1 Shuffled Surface Macro MRE: {mets_d1['macro_target_mre']:.2f} mm (vs Matched {b7_results[0]['macro_target_mre']:.2f} mm)")

    # STAGE 33: LEARNING CURVE (25%, 50%, 75%, 100%)
    print("\n--- Stage 33: Learning Curve Subsets ---")
    lc_results = {}
    for pct in [0.25, 0.50, 0.75, 1.00]:
        n_sub = int(round(len(tr_idx) * pct))
        sub_indices = tr_idx[:n_sub]
        tr_ds_sub = V2Dataset(pts_m[sub_indices], tgt_m[sub_indices], tgt_mask[sub_indices], prim_mask[sub_indices], augment=True)
        tr_loader_sub = DataLoader(tr_ds_sub, batch_size=16, shuffle=True, drop_last=True)
        
        torch.manual_seed(42)
        m_sub = PointNet2PlainRegressor().to(device)
        macro_sub, _, _ = train_pointnet(
            tr_loader_sub, val_loader, m_sub, epochs=60, lr=4e-4, S_global=S_global,
            primary_indices=primary_107_indices, val_tgt_c=val_tgt_c, has_cond=False
        )
        lc_results[int(pct * 100)] = macro_sub
        print(f"  Subset {int(pct*100)}% ({n_sub} patients): Macro Target MRE = {macro_sub:.2f} mm")

    # STAGE 35: BOOTSTRAP CONFIDENCE INTERVALS (1000 resamples)
    print("\n--- Stage 35: Bootstrap Confidence Intervals (1000 resamples) ---")
    boot_macro = []
    boot_micro = []
    boot_sdr10 = []

    val_mask_bool = prim_mask[val_idx] > 0.5
    for b_iter in range(1000):
        b_pat = rng.choice(len(val_idx), size=len(val_idx), replace=True)
        b_preds = best_nn_preds[b_pat]
        b_gt = val_tgt_c[b_pat]
        b_m = val_mask_bool[b_pat]
        b_mets = compute_all_metrics(b_preds, b_gt, b_m, primary_107_indices)
        boot_macro.append(b_mets["macro_target_mre"])
        boot_micro.append(b_mets["micro_mre"])
        boot_sdr10.append(b_mets["sdr_10"])

    ci_macro = (float(np.percentile(boot_macro, 2.5)), float(np.percentile(boot_macro, 97.5)))
    ci_micro = (float(np.percentile(boot_micro, 2.5)), float(np.percentile(boot_micro, 97.5)))
    ci_sdr10 = (float(np.percentile(boot_sdr10, 2.5)), float(np.percentile(boot_sdr10, 97.5)))

    print(f"  Macro Target MRE 95% CI: [{ci_macro[0]:.2f}, {ci_macro[1]:.2f}] mm")
    print(f"  Micro MRE 95% CI:        [{ci_micro[0]:.2f}, {ci_micro[1]:.2f}] mm")
    print(f"  SDR@10 95% CI:           [{ci_sdr10[0]:.2f}%, {ci_sdr10[1]:.2f}%]")

    # Save all results to JSON
    phase2_all_results = {
        "D0_OVERFIT_8_MRE": d0_mre,
        "D0_PASSED": d0_passed,
        "B7_macro_mean": mean_b7_macro,
        "B7_macro_std": std_b7_macro,
        "B7_micro_mean": mean_b7_micro,
        "B7_micro_std": std_b7_micro,
        "B7_sdr10_mean": mean_b7_sdr10,
        "B8_macro": mets_b8["macro_target_mre"],
        "B8_micro": mets_b8["micro_mre"],
        "B9_macro": mets_b9["macro_target_mre"],
        "B9_micro": mets_b9["micro_mre"],
        "variance_ratio_median": median_rk,
        "variance_ratio_mean": mean_rk,
        "collapse_present": collapse_present,
        "matched_macro": b7_results[0]["macro_target_mre"],
        "mismatched_macro": mets_mismatch["macro_target_mre"],
        "matched_better": matched_better,
        "D1_shuffled_macro": mets_d1["macro_target_mre"],
        "learning_curve": lc_results,
        "ci_macro": ci_macro,
        "ci_micro": ci_micro,
        "ci_sdr10": ci_sdr10,
        "best_b7_preds": best_nn_preds.tolist(),
        "primary_indices": primary_107_indices
    }
    with open(repo_root / "experiments" / "phase2" / "pointnet_results.json", "w") as f:
        json.dump(phase2_all_results, f, indent=2)

    print("\n[Done] All PointNet++ experiments completed and saved.")

if __name__ == "__main__":
    main()
