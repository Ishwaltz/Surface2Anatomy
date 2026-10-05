import sys
import json
import csv
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from labels import ORGAN_NAMES

pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"
results_path = repo_root / "experiments" / "phase2" / "pointnet_results.json"
tiers_path = repo_root / "reports" / "phase2" / "01_evaluation_cohort_tiers.csv"

data = torch.load(str(pt_path), weights_only=False)
with open(splits_path) as f:
    splits = json.load(f)
with open(results_path) as f:
    res = json.load(f)

val_idx = splits["val_indices"]
primary_indices = res["primary_indices"]
target_mres = res["target_mres"]

# Compute depth from surface for every target in validation set
pts_w = data["points_world_mm"].numpy()[val_idx] # (44, 4096, 3)
tgt_w = data["targets_world_mm"].numpy()[val_idx] # (44, 121, 3)
p_mask = data["target_primary_mask"].numpy()[val_idx] > 0.5 # (44, 121)

depths_per_target = []
mres_per_target = []

for k in primary_indices:
    k_depths = []
    for i in range(len(val_idx)):
        if p_mask[i, k]:
            d = np.min(np.linalg.norm(pts_w[i] - tgt_w[i, k], axis=-1))
            k_depths.append(d)
    if len(k_depths) > 0 and not np.isnan(target_mres[k]):
        depths_per_target.append(np.mean(k_depths))
        mres_per_target.append(target_mres[k])

depths = np.array(depths_per_target)
mres = np.array(mres_per_target)

pearson_r, pearson_p = stats.pearsonr(depths, mres)
spearman_r, spearman_p = stats.spearmanr(depths, mres)

print(f"Depth vs Error Correlation:")
print(f"  Pearson:  r = {pearson_r:.4f} (p = {pearson_p:.4e})")
print(f"  Spearman: r = {spearman_r:.4f} (p = {spearman_p:.4e})")

# Extremes among primary targets
sorted_targets = []
for k in primary_indices:
    if not np.isnan(target_mres[k]):
        sorted_targets.append((ORGAN_NAMES[k], target_mres[k], k))

sorted_targets.sort(key=lambda x: x[1])

print("\nTop 5 Easiest Primary Targets:")
for name, err, k in sorted_targets[:5]:
    print(f"  {name:30s}: {err:.2f} mm")

print("\nTop 5 Hardest Primary Targets:")
for name, err, k in sorted_targets[-5:]:
    print(f"  {name:30s}: {err:.2f} mm")

# Save correlation summary
with open(repo_root / "experiments" / "phase2" / "depth_correlations.json", "w") as f:
    json.dump({
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "easiest": sorted_targets[0],
        "hardest": sorted_targets[-1]
    }, f, indent=2)
