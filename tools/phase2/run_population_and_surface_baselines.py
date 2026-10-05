import sys
import json
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES
from tools.phase2.metrics import compute_all_metrics

def main():
    print("=" * 80)
    print("STAGE 3-9: RUNNING POPULATION & SURFACE BASELINES (B0 - B6)")
    print("=" * 80)

    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
    tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"

    data = torch.load(str(pt_path), weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)

    tr_idx = splits["train_indices"]
    val_idx = splits["val_indices"]

    # Load evaluation cohort tiers
    with open(tiers_path) as f:
        tier_reader = csv.DictReader(f)
        primary_107_indices = [int(r["target_index"]) for r in tier_reader if r["tier"] in ["TIER_A", "TIER_B"]]

    print(f"Primary Evaluation Targets: {len(primary_107_indices)} / 121")

    # Arrays
    targets_c = data["targets_centered_mm"].numpy() # (440, 121, 3) in physical mm centered
    p_mask = data["target_primary_mask"].numpy()    # (440, 121)
    body_dims = data["body_dimensions_mm"].numpy()  # (440, 3)
    sex = data["sex"].numpy()                       # (440,)
    pts_c = data["points_centered_mm"].numpy()      # (440, 4096, 3)
    
    # Load manifest to get surface area and body volume
    man_csv = repo_root / "sharon" / "dataset_v2" / "manifest_v2.csv"
    areas = np.zeros(len(sex), dtype=np.float32)
    volumes = np.zeros(len(sex), dtype=np.float32)
    with open(man_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            areas[i] = float(row["surface_area_mm2"])
            volumes[i] = float(row["body_volume_liters"])

    # Training and Validation slices
    tr_tgt = targets_c[tr_idx] # (352, 121, 3)
    val_tgt = targets_c[val_idx] # (44, 121, 3)
    tr_mask = p_mask[tr_idx]
    val_mask = p_mask[val_idx]

    tr_sex = sex[tr_idx]
    val_sex = sex[val_idx]

    # Build external body feature matrix: [W, D, H, Area, Vol, Sex] (6 features)
    features = np.column_stack([
        body_dims[:, 0] / 100.0,
        body_dims[:, 1] / 100.0,
        body_dims[:, 2] / 100.0,
        areas / 1e5,
        volumes / 10.0,
        sex.astype(np.float32)
    ])
    tr_feat = features[tr_idx]
    val_feat = features[val_idx]

    # STAGE 3: B0_GLOBAL_MEAN_ATLAS
    print("\n--- Running B0: Global Mean Atlas ---")
    b0_atlas = np.zeros((121, 3), dtype=np.float32)
    for k in range(121):
        k_m = tr_mask[:, k] > 0.5
        if k_m.sum() > 0:
            b0_atlas[k] = tr_tgt[k_m, k].mean(axis=0)

    val_pred_b0 = np.tile(b0_atlas[None, :, :], (len(val_idx), 1, 1))
    metrics_b0 = compute_all_metrics(val_pred_b0, val_tgt, val_mask, primary_107_indices)

    print(f"  B0 Macro Target MRE: {metrics_b0['macro_target_mre']:.2f} mm")
    print(f"  B0 Micro MRE:        {metrics_b0['micro_mre']:.2f} mm")
    print(f"  B0 Median Error:     {metrics_b0['median']:.2f} mm")
    print(f"  B0 P90 Error:        {metrics_b0['p90']:.2f} mm")
    print(f"  B0 SDR@10:           {metrics_b0['sdr_10']:.2f}%")

    # STAGE 4: B1_SEX_MEAN_ATLAS
    print("\n--- Running B1: Sex-Conditioned Mean Atlas ---")
    b1_atlas_m = np.zeros((121, 3), dtype=np.float32)
    b1_atlas_f = np.zeros((121, 3), dtype=np.float32)
    for k in range(121):
        m_mask = (tr_mask[:, k] > 0.5) & (tr_sex == 1)
        f_mask = (tr_mask[:, k] > 0.5) & (tr_sex == 0)
        b1_atlas_m[k] = tr_tgt[m_mask, k].mean(axis=0) if m_mask.sum() > 0 else b0_atlas[k]
        b1_atlas_f[k] = tr_tgt[f_mask, k].mean(axis=0) if f_mask.sum() > 0 else b0_atlas[k]

    val_pred_b1 = np.zeros_like(val_tgt)
    for i in range(len(val_idx)):
        val_pred_b1[i] = b1_atlas_m if val_sex[i] == 1 else b1_atlas_f
    metrics_b1 = compute_all_metrics(val_pred_b1, val_tgt, val_mask, primary_107_indices)

    print(f"  B1 Macro Target MRE: {metrics_b1['macro_target_mre']:.2f} mm")
    print(f"  B1 Micro MRE:        {metrics_b1['micro_mre']:.2f} mm")
    print(f"  B1 Median Error:     {metrics_b1['median']:.2f} mm")
    print(f"  B1 SDR@10:           {metrics_b1['sdr_10']:.2f}%")

    # STAGE 5: B2_BODY_LINEAR (Ridge Regression)
    print("\n--- Running B2: Body-Size Conditioned Linear Baseline ---")
    val_pred_b2 = np.tile(b0_atlas[None, :, :], (len(val_idx), 1, 1))
    for k in primary_107_indices:
        k_m = tr_mask[:, k] > 0.5
        if k_m.sum() >= 10:
            X_tr = tr_feat[k_m]
            Y_tr = tr_tgt[k_m, k]
            ridge = RidgeCV(alphas=np.logspace(-2, 3, 10))
            ridge.fit(X_tr, Y_tr)
            val_pred_b2[:, k] = ridge.predict(val_feat)

    metrics_b2 = compute_all_metrics(val_pred_b2, val_tgt, val_mask, primary_107_indices)
    print(f"  B2 Macro Target MRE: {metrics_b2['macro_target_mre']:.2f} mm")
    print(f"  B2 Micro MRE:        {metrics_b2['micro_mre']:.2f} mm")
    print(f"  B2 Median Error:     {metrics_b2['median']:.2f} mm")
    print(f"  B2 SDR@10:           {metrics_b2['sdr_10']:.2f}%")

    # STAGE 6: B3_BODY_MLP
    print("\n--- Running B3: Body-Size Nonlinear Baseline (MLP) ---")
    class BodyMLP(nn.Module):
        def __init__(self, in_dim=6, out_dim=121*3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, out_dim)
            )
        def forward(self, x):
            return self.net(x).view(-1, 121, 3)

    mlp = BodyMLP(in_dim=6, out_dim=121*3)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    x_tr_t = torch.from_numpy(tr_feat).float()
    y_tr_t = torch.from_numpy(tr_tgt).float()
    m_tr_t = torch.from_numpy(tr_mask).float().unsqueeze(-1)

    for epoch in range(150):
        mlp.train()
        opt.zero_grad()
        out = mlp(x_tr_t)
        diff = (out - y_tr_t) * m_tr_t
        loss = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1.0).sum() / m_tr_t.sum().clamp(min=1.0)
        loss.backward()
        opt.step()

    mlp.eval()
    with torch.no_grad():
        val_pred_b3 = mlp(torch.from_numpy(val_feat).float()).numpy()
    metrics_b3 = compute_all_metrics(val_pred_b3, val_tgt, val_mask, primary_107_indices)
    print(f"  B3 Macro Target MRE: {metrics_b3['macro_target_mre']:.2f} mm")
    print(f"  B3 Micro MRE:        {metrics_b3['micro_mre']:.2f} mm")
    print(f"  B3 Median Error:     {metrics_b3['median']:.2f} mm")
    print(f"  B3 SDR@10:           {metrics_b3['sdr_10']:.2f}%")

    # STAGE 7: EXTERNAL-SURFACE NEAREST-NEIGHBOUR (B4, B5, B6)
    print("\n--- Running B4-B6: External Surface Nearest-Neighbour Baselines ---")
    # Descriptor: PCA on flattened surface point clouds (subsampled 512 points for speed and compactness)
    sub_idx = np.linspace(0, 4095, 512).astype(int)
    tr_surf_flat = pts_c[tr_idx][:, sub_idx, :].reshape(len(tr_idx), -1)
    val_surf_flat = pts_c[val_idx][:, sub_idx, :].reshape(len(val_idx), -1)

    pca = PCA(n_components=20, random_state=42)
    tr_desc = pca.fit_transform(tr_surf_flat)
    val_desc = pca.transform(val_surf_flat)

    nbrs = NearestNeighbors(n_neighbors=10, metric="euclidean").fit(tr_desc)
    distances, indices = nbrs.kneighbors(val_desc)

    def eval_knn(k_val):
        preds = np.zeros_like(val_tgt)
        for i in range(len(val_idx)):
            nn_idx = indices[i, :k_val]
            for target_k in range(121):
                m_nn = tr_mask[nn_idx, target_k] > 0.5
                if m_nn.sum() > 0:
                    preds[i, target_k] = tr_tgt[nn_idx[m_nn], target_k].mean(axis=0)
                else:
                    preds[i, target_k] = b0_atlas[target_k]
        return compute_all_metrics(preds, val_tgt, val_mask, primary_107_indices)

    metrics_b4 = eval_knn(1)
    metrics_b5 = eval_knn(3)
    metrics_b6 = eval_knn(5)

    print(f"  B4 (1NN) Macro Target MRE: {metrics_b4['macro_target_mre']:.2f} mm | Micro: {metrics_b4['micro_mre']:.2f} mm | SDR@10: {metrics_b4['sdr_10']:.2f}%")
    print(f"  B5 (3NN) Macro Target MRE: {metrics_b5['macro_target_mre']:.2f} mm | Micro: {metrics_b5['micro_mre']:.2f} mm | SDR@10: {metrics_b5['sdr_10']:.2f}%")
    print(f"  B6 (5NN) Macro Target MRE: {metrics_b6['macro_target_mre']:.2f} mm | Micro: {metrics_b6['micro_mre']:.2f} mm | SDR@10: {metrics_b6['sdr_10']:.2f}%")

    # STAGE 8 & 9: EMPIRICAL LOCAL VS GLOBAL ANATOMICAL VARIANCE
    print("\n--- Stages 8 & 9: Local vs Global Anatomical Variability ---")
    global_sigmas = []
    local_sigmas_5nn = []
    local_sigmas_10nn = []

    for k in primary_107_indices:
        # Global target variance
        k_m = tr_mask[:, k] > 0.5
        if k_m.sum() > 5:
            coords = tr_tgt[k_m, k]
            cov = np.cov(coords, rowvar=False)
            sigma_g = np.sqrt(np.trace(cov))
            global_sigmas.append(sigma_g)
        else:
            global_sigmas.append(np.nan)

        # Local target variance across 5NN
        local_spreads = []
        for i in range(len(val_idx)):
            nn_idx = indices[i, :5]
            m_nn = tr_mask[nn_idx, k] > 0.5
            if m_nn.sum() >= 2:
                nn_coords = tr_tgt[nn_idx[m_nn], k]
                mean_coord = nn_coords.mean(axis=0)
                spread = np.sqrt(np.mean(np.sum((nn_coords - mean_coord) ** 2, axis=-1)))
                local_spreads.append(spread)
        if len(local_spreads) > 0:
            local_sigmas_5nn.append(np.median(local_spreads))
        else:
            local_sigmas_5nn.append(np.nan)

    val_g = [s for s in global_sigmas if not np.isnan(s)]
    val_l = [s for s in local_sigmas_5nn if not np.isnan(s)]

    print(f"  Global Anatomical Spread (mean across targets): {np.mean(val_g):.2f} mm (median: {np.median(val_g):.2f} mm)")
    print(f"  Local 5NN Anatomical Spread (mean across targets): {np.mean(val_l):.2f} mm (median: {np.median(val_l):.2f} mm)")
    print(f"  Variability Reduction (Local / Global): {np.mean(val_l) / np.mean(val_g):.2f}x")

    # Save summary report: reports/phase2/B0_global_mean_atlas.md
    out_b0 = repo_root / "reports" / "phase2" / "B0_global_mean_atlas.md"
    with open(out_b0, "w") as f:
        f.write("# B0: Global Mean Atlas Baseline Report\n\n")
        f.write("## 1. Description\n")
        f.write("The Global Mean Atlas computes the empirical centroid for each anatomical target across all valid training cases ($N=352$) in the patient-centered metric coordinate frame, and predicts this constant coordinate for every validation patient.\n\n")
        f.write("## 2. Validation Metrics (Primary 107 Targets)\n")
        f.write(f"- **Macro-Target MRE**: **{metrics_b0['macro_target_mre']:.2f} mm**\n")
        f.write(f"- **Micro MRE**: **{metrics_b0['micro_mre']:.2f} mm**\n")
        f.write(f"- **Macro-Patient MRE**: **{metrics_b0['macro_patient_mre']:.2f} mm**\n")
        f.write(f"- **Median Radial Error**: **{metrics_b0['median']:.2f} mm**\n")
        f.write(f"- **P75 / P90 Error**: {metrics_b0['p75']:.2f} mm / {metrics_b0['p90']:.2f} mm\n")
        f.write(f"- **SDR@10 / SDR@20**: {metrics_b0['sdr_10']:.2f}% / {metrics_b0['sdr_20']:.2f}%\n")

    # Save all baseline metrics to JSON for table generation
    all_b_metrics = {
        "B0_GLOBAL_MEAN_ATLAS": metrics_b0,
        "B1_SEX_MEAN_ATLAS": metrics_b1,
        "B2_BODY_LINEAR": metrics_b2,
        "B3_BODY_MLP": metrics_b3,
        "B4_SURFACE_1NN": metrics_b4,
        "B5_SURFACE_3NN": metrics_b5,
        "B6_SURFACE_5NN": metrics_b6,
        "global_spread_mean": float(np.mean(val_g)),
        "local_5nn_spread_mean": float(np.mean(val_l))
    }
    with open(repo_root / "experiments" / "phase2" / "simple_baselines_metrics.json", "w") as f:
        json.dump(all_b_metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else None)

    print(f"[Done] Simple baselines complete and saved.")

if __name__ == "__main__":
    main()
