import sys
import csv
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

def run_roundtrip_test():
    print("=" * 80)
    print("AUDIT: COORDINATE ROUND-TRIP TRANSFORMATION TEST")
    print("=" * 80)

    pt_path = repo_root / "sharon" / "dataset" / "pointclouds_450.pt"
    data = torch.load(str(pt_path), weights_only=False)

    case_ids = data["case_ids"]
    raw_points = data["raw_points"]       # (450, 4096, 3)
    norm_points = data["points"]          # (450, 4096, 3)
    raw_centroids = data["raw_centroids"] # (450, 121, 3)
    norm_centroids = data["centroids"]    # (450, 121, 3)
    centers = data["centers"]              # (450, 3)
    scales = data["scales"]                # (450, 3)
    masks = data["masks"]                  # (450, 121)

    N = len(case_ids)
    patient_stats = []

    all_target_errs = []
    all_surface_errs = []

    for i in range(N):
        cid = case_ids[i]
        c = centers[i]
        s = scales[i]

        # Reconstruct Targets
        m = masks[i] > 0.5
        rec_targets = norm_centroids[i] * s + c
        t_diff = torch.norm((raw_centroids[i] - rec_targets)[m], dim=-1)
        t_errs = t_diff.cpu().numpy()

        # Reconstruct Surface (subsample 100 random points)
        idx_sub = np.random.choice(4096, 100, replace=False)
        rec_pts = norm_points[i, idx_sub] * s + c
        s_diff = torch.norm(raw_points[i, idx_sub] - rec_pts, dim=-1)
        s_errs = s_diff.cpu().numpy()

        all_target_errs.extend(t_errs)
        all_surface_errs.extend(s_errs)

        combined_errs = np.concatenate([t_errs, s_errs])
        patient_stats.append({
            "patient_id": cid,
            "mean_error_mm": float(np.mean(combined_errs)),
            "max_error_mm": float(np.max(combined_errs)),
            "p99_error_mm": float(np.percentile(combined_errs, 99)),
        })

    # Overall summary
    all_combined = np.concatenate([all_target_errs, all_surface_errs])
    mean_err = float(np.mean(all_combined))
    med_err = float(np.median(all_combined))
    max_err = float(np.max(all_combined))
    p99_err = float(np.percentile(all_combined, 99))

    print(f"Total Patients Tested: {N}")
    print(f"Overall Mean Round-Trip Error:   {mean_err:.8e} mm")
    print(f"Overall Median Round-Trip Error: {med_err:.8e} mm")
    print(f"Overall Max Round-Trip Error:    {max_err:.8e} mm")
    print(f"Overall P99 Round-Trip Error:    {p99_err:.8e} mm")

    if max_err < 1e-3:
        status = "PASS (Exact floating-point invertibility verified)"
    elif max_err < 0.1:
        status = "ACCEPTABLE (<0.1 mm)"
    else:
        status = "FAIL (>0.1 mm numerical error)"

    print(f"Verdict: {status}")

    # Write CSV
    out_csv = repo_root / "reports" / "phase1" / "roundtrip_test.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "mean_error_mm", "max_error_mm", "p99_error_mm"])
        writer.writeheader()
        writer.writerows(patient_stats)

    print(f"[Done] CSV written to: {out_csv}")
    return mean_err, med_err, max_err, p99_err, status

if __name__ == "__main__":
    run_roundtrip_test()
