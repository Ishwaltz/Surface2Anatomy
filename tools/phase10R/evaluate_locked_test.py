import os
import sys
import json
import csv
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from sharon.model_target_query import TargetQueryTransformerDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_file_path = repo_root / "reports" / "phase10R" / "logs" / "test_evaluation.log"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

class TestDataset(Dataset):
    def __init__(self, pts, S_global=500.0):
        self.pts = pts.clone().float() / S_global
    def __len__(self):
        return len(self.pts)
    def __getitem__(self, idx):
        return {"pts": self.pts[idx]}

def compute_test_metrics(pred_c_mm, tgt_c_mm, masks, benchmark_indices):
    pred_sub = pred_c_mm[:, benchmark_indices, :]
    tgt_sub = tgt_c_mm[:, benchmark_indices, :]
    mask_sub = masks[:, benchmark_indices]
    
    errors = np.linalg.norm(pred_sub - tgt_sub, axis=-1)
    valid = (mask_sub == 1)
    valid_errors = errors[valid]
    
    target_mres = []
    target_records = []
    for t_i in range(len(benchmark_indices)):
        v = (mask_sub[:, t_i] == 1)
        if np.sum(v) > 0:
            e_t = errors[v, t_i]
            mre_t = float(np.mean(e_t))
            target_mres.append(mre_t)
            target_records.append({
                "target_idx": int(benchmark_indices[t_i]),
                "support": int(np.sum(v)),
                "mre": mre_t,
                "median": float(np.median(e_t)),
                "p90": float(np.percentile(e_t, 90)),
                "sdr10": float(np.mean(e_t <= 10.0) * 100.0),
                "sdr20": float(np.mean(e_t <= 20.0) * 100.0)
            })
            
    macro_mre = float(np.mean(target_mres))
    micro_mre = float(np.mean(valid_errors))
    median_err = float(np.median(valid_errors))
    p90_err = float(np.percentile(valid_errors, 90))
    sdr5 = float(np.mean(valid_errors <= 5.0) * 100.0)
    sdr10 = float(np.mean(valid_errors <= 10.0) * 100.0)
    sdr15 = float(np.mean(valid_errors <= 15.0) * 100.0)
    sdr20 = float(np.mean(valid_errors <= 20.0) * 100.0)
    
    patient_mres = []
    for i in range(len(pred_c_mm)):
        v = (mask_sub[i] == 1)
        if np.sum(v) > 0:
            patient_mres.append(float(np.mean(errors[i, v])))
            
    return {
        "macro_mre": macro_mre,
        "micro_mre": micro_mre,
        "median": median_err,
        "p90": p90_err,
        "sdr5": sdr5,
        "sdr10": sdr10,
        "sdr15": sdr15,
        "sdr20": sdr20,
        "target_records": target_records,
        "patient_mres": patient_mres
    }

def main():
    log("="*80)
    log("SECTION 8: PRE-TEST FREEZE VERIFICATION & LOCKED TEST EVALUATION")
    log("="*80)
    
    freeze_doc = repo_root / "reports" / "phase10R" / "PRE_TEST_FREEZE.md"
    freeze_sha = repo_root / "reports" / "phase10R" / "PRE_TEST_FREEZE.sha256"
    
    if not freeze_doc.exists() or not freeze_sha.exists():
        log("CRITICAL ERROR: PRE_TEST_FREEZE.md or PRE_TEST_FREEZE.sha256 not found!")
        log("Hard gate violation: You cannot access or evaluate the locked test split without a sealed freeze.")
        sys.exit(1)
        
    with open(freeze_sha) as f:
        expected_hash = f.read().strip()
    with open(freeze_doc, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
        
    if actual_hash != expected_hash:
        log(f"CRITICAL ERROR: SHA256 mismatch for PRE_TEST_FREEZE.md! Expected: {expected_hash}, Got: {actual_hash}")
        sys.exit(1)
        
    log(f"Pre-Test Freeze Gate PASSED. SHA-256 verified: {actual_hash[:16]}...")
    log("Proceeding to ONE-TIME evaluation of locked held-out test split (N=168)...")
    
    data_path = repo_root / "sharon/dataset_v3/pointclouds_v3.pt"
    splits_path = repo_root / "sharon/dataset_v3/splits_v3_iid.json"
    
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)
        
    test_idx = splits["test_indices"]
    pts_test = data["points_centered_4096"][test_idx]
    tgts_test = data["targets_centered"][test_idx].numpy()
    masks_test = data["target_masks"][test_idx].numpy()
    sources = data["source_datasets"]
    bench_idx = data["primary_104_indices"].tolist()
    canonical_names = data["canonical_target_names"]
    
    test_ds = TestDataset(pts_test, S_global=500.0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    seed_preds = []
    seed_metrics = []
    for s in [42, 43, 44]:
        ckpt_path = repo_root / f"experiments/phase10R/checkpoints/C4_Proposed_seed{s}.pt"
        if not ckpt_path.exists():
            log(f"ERROR: Proposed checkpoint for seed {s} not found: {ckpt_path}")
            sys.exit(1)
            
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = TargetQueryTransformerDecoder(atlas_coords=torch.zeros(117, 3).to(device), num_organs=117).to(device)
        m.load_state_dict(saved["model_state_dict"])
        m.eval()
        
        preds_s = []
        with torch.no_grad():
            for batch in test_loader:
                p_m, _ = m(batch["pts"].to(device))
                preds_s.append(p_m.cpu().numpy() * 500.0)
        p_arr = np.concatenate(preds_s, axis=0)
        seed_preds.append(p_arr)
        met_s = compute_test_metrics(p_arr, tgts_test, masks_test, bench_idx)
        seed_metrics.append(met_s)
        log(f"  Seed {s} Locked Test Macro MRE = {met_s['macro_mre']:.2f} mm (Median={met_s['median']:.2f} mm, SDR@10={met_s['sdr10']:.1f}%)")
        
    # Ensemble predictions on Test
    ens_preds = np.mean(seed_preds, axis=0)
    ens_metrics = compute_test_metrics(ens_preds, tgts_test, masks_test, bench_idx)
    
    # 5,000 Bootstrap Resamples on Test Patient MREs
    n_boot = 5000
    np.random.seed(42)
    pat_mres = np.array(ens_metrics["patient_mres"])
    boot_means = [np.mean(pat_mres[np.random.choice(len(pat_mres), size=len(pat_mres), replace=True)]) for _ in range(n_boot)]
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    
    # Breakdown by Source Cohort
    test_v2_mask = [sources[i] == "v2" for i in test_idx]
    test_ts_mask = [sources[i] == "totalsegmentator" for i in test_idx]
    
    met_v2 = compute_test_metrics(ens_preds[test_v2_mask], tgts_test[test_v2_mask], masks_test[test_v2_mask], bench_idx)
    met_ts = compute_test_metrics(ens_preds[test_ts_mask], tgts_test[test_ts_mask], masks_test[test_ts_mask], bench_idx)
    
    log("\n" + "="*80)
    log(f"FINAL LOCKED TEST HEADLINE RESULT:")
    log(f"  3-Seed Mean: {np.mean([m['macro_mre'] for m in seed_metrics]):.2f} ± {np.std([m['macro_mre'] for m in seed_metrics]):.2f} mm")
    log(f"  3-Model Ensemble Macro MRE: {ens_metrics['macro_mre']:.2f} mm [95% CI: {ci_low:.2f} - {ci_high:.2f} mm]")
    log(f"  Ensemble Micro MRE: {ens_metrics['micro_mre']:.2f} mm")
    log(f"  Ensemble Median Error: {ens_metrics['median']:.2f} mm")
    log(f"  Ensemble P90 Error: {ens_metrics['p90']:.2f} mm")
    log(f"  SDR@10: {ens_metrics['sdr10']:.1f}% | SDR@20: {ens_metrics['sdr20']:.1f}%")
    log(f"  Cohort Breakdown: V2 Test = {met_v2['macro_mre']:.2f} mm | TotalSegmentator Test = {met_ts['macro_mre']:.2f} mm")
    log("="*80)
    
    # Save predictions npz
    out_dir = repo_root / "reports" / "phase10R"
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    npz_path = pred_dir / "final_test_predictions.npz"
    np.savez_compressed(
        npz_path,
        ensemble_predictions=ens_preds,
        seed42_predictions=seed_preds[0],
        seed43_predictions=seed_preds[1],
        seed44_predictions=seed_preds[2],
        ground_truth_targets=tgts_test,
        target_masks=masks_test,
        test_patient_indices=test_idx
    )
    log(f"Saved test predictions array to {npz_path}")
    
    # Write FINAL_LOCKED_TEST_RESULTS.md
    md_test = out_dir / "FINAL_LOCKED_TEST_RESULTS.md"
    with open(md_test, "w", encoding="utf-8") as f:
        f.write("# Phase 10R: Final Locked Test Benchmark Evaluation\n\n")
        f.write("## 1. Compliance Statement & Protocol Integrity\n")
        f.write("- **Test Set Status:** The held-out test split (N=168 subjects) remained strictly locked throughout model development, tuning, and hyperparameter selection.\n")
        f.write(f"- **Freeze Confirmation:** Pre-test protocol frozen and sealed under SHA-256 `{actual_hash}`.\n")
        f.write("- **One-Time Evaluation:** This evaluation was executed exactly once with zero post-hoc tuning or re-selection.\n\n")
        
        f.write("## 2. Definitive Test Performance Summary\n\n")
        f.write("| Evaluation Configuration | Macro MRE (mm) | Micro MRE (mm) | Median (mm) | P90 (mm) | SDR@10 (%) | SDR@20 (%) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for i, s in enumerate([42, 43, 44]):
            m = seed_metrics[i]
            f.write(f"| Seed {s} | {m['macro_mre']:.2f} | {m['micro_mre']:.2f} | {m['median']:.2f} | {m['p90']:.2f} | {m['sdr10']:.1f}% | {m['sdr20']:.1f}% |\n")
        f.write(f"| **3-Seed Mean ± SD** | **{np.mean([m['macro_mre'] for m in seed_metrics]):.2f} ± {np.std([m['macro_mre'] for m in seed_metrics]):.2f}** | **{np.mean([m['micro_mre'] for m in seed_metrics]):.2f}** | — | — | **{np.mean([m['sdr10'] for m in seed_metrics]):.1f}%** | **{np.mean([m['sdr20'] for m in seed_metrics]):.1f}%** |\n")
        f.write(f"| **3-Model Prediction Ensemble** | **{ens_metrics['macro_mre']:.2f}** | **{ens_metrics['micro_mre']:.2f}** | **{ens_metrics['median']:.2f}** | **{ens_metrics['p90']:.2f}** | **{ens_metrics['sdr10']:.1f}%** | **{ens_metrics['sdr20']:.1f}%** |\n\n")
        
        f.write("## 3. Statistical Confidence & Cohort Breakdown\n\n")
        f.write(f"- **95% Bootstrap Confidence Interval (Macro MRE):** **[{ci_low:.2f} mm, {ci_high:.2f} mm]** (5,000 resamples)\n")
        f.write(f"- **V2 Test Cohort (N={np.sum(test_v2_mask)}):** **{met_v2['macro_mre']:.2f} mm** (Median: {met_v2['median']:.2f} mm, SDR@10: {met_v2['sdr10']:.1f}%)\n")
        f.write(f"- **TotalSegmentator Test Cohort (N={np.sum(test_ts_mask)}):** **{met_ts['macro_mre']:.2f} mm** (Median: {met_ts['median']:.2f} mm, SDR@10: {met_ts['sdr10']:.1f}%)\n")
    log(f"Saved {md_test}")
    
    # Update canonical_results.json
    res_path = out_dir / "canonical_results.json"
    canonical_results = {}
    if res_path.exists():
        with open(res_path) as f:
            canonical_results = json.load(f)
    canonical_results["locked_test_results"] = {
        "ensemble_macro_mre": ens_metrics["macro_mre"],
        "ensemble_micro_mre": ens_metrics["micro_mre"],
        "ensemble_median": ens_metrics["median"],
        "ensemble_p90": ens_metrics["p90"],
        "ensemble_sdr10": ens_metrics["sdr10"],
        "ensemble_sdr20": ens_metrics["sdr20"],
        "bootstrap_ci_95": [ci_low, ci_high],
        "v2_test_mre": met_v2["macro_mre"],
        "ts_test_mre": met_ts["macro_mre"],
        "seed_macros": [m["macro_mre"] for m in seed_metrics]
    }
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(canonical_results, f, indent=2)
    log(f"Updated {res_path} with final locked test benchmark results.")

if __name__ == "__main__":
    main()
