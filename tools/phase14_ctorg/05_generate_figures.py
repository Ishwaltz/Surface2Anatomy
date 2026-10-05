#!/usr/bin/env python3
"""
tools/phase14_ctorg/05_generate_figures.py
=========================================
Generates publication-quality figures for CT-ORG zero-shot external validation:
1. Target-wise MRE comparison bar chart
2. Cumulative SDR (Success Detection Rate) curve
3. Multi-method baseline benchmark comparison
4. Internal vs External generalization comparison
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent.parent.parent
reports_dir = repo_root / "reports" / "phase14_ctorg"
fig_dir = reports_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6
})

def main():
    print("Generating CT-ORG validation figures...")
    
    # Load data
    targets_csv = reports_dir / "08_CTORG_target_results_FINAL.csv"
    baselines_csv = reports_dir / "10_CTORG_baselines_FINAL.csv"
    pred_csv = reports_dir / "CTORG_predictions_FINAL.csv"
    final_json = reports_dir / "CTORG_EXTERNAL_FINAL.json"
    
    if not targets_csv.exists() or not baselines_csv.exists() or not pred_csv.exists():
        print("Required CSV files not found yet. Figure generation will run after inference.")
        return
        
    df_t = pd.read_csv(targets_csv)
    df_b = pd.read_csv(baselines_csv)
    df_p = pd.read_csv(pred_csv)
    with open(final_json) as f:
        d_final = json.load(f)
        
    # Figure 1: Target-wise MRE
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = ["#2b5c8f", "#3c7bb6", "#4d9ad6", "#5eb7f2", "#7fc6f7"][:len(df_t)]
    bars = ax.bar(df_t["target_name"], df_t["MRE_mm"], color=colors, edgecolor="black", width=0.55, zorder=3)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_ylabel("Mean Radial Error (mm)")
    ax.set_title("CT-ORG Zero-Shot External Validation: Target-Wise Localization Error")
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} mm",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")
                    
    ax.set_ylim(0, max(df_t["MRE_mm"]) * 1.25)
    plt.tight_layout()
    p1 = fig_dir / "ctorg_target_wise_mre.png"
    plt.savefig(p1)
    plt.close()
    print(f"Saved {p1}")
    
    # Figure 2: Cumulative SDR curve
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    errs = sorted(df_p["error_ens_mm"].tolist())
    thresholds = np.linspace(0, 50, 101)
    sdr_vals = [np.mean(np.array(errs) <= t) * 100.0 for t in thresholds]
    
    ax.plot(thresholds, sdr_vals, color="#1b4d3e", label="Proposed Ensemble (3 Seeds)")
    ax.axvline(10, color="gray", linestyle=":", label=f"SDR@10mm: {d_final['SDR10_PERCENT']:.1f}%")
    ax.axvline(20, color="orange", linestyle=":", label=f"SDR@20mm: {d_final['SDR20_PERCENT']:.1f}%")
    ax.axvline(30, color="red", linestyle=":", label=f"SDR@30mm: {d_final['SDR30_PERCENT']:.1f}%")
    
    ax.set_xlabel("Error Distance Threshold (mm)")
    ax.set_ylabel("Success Detection Rate (%)")
    ax.set_title("CT-ORG External Validation: Cumulative Accuracy Profile")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    p2 = fig_dir / "ctorg_sdr_curve.png"
    plt.savefig(p2)
    plt.close()
    print(f"Saved {p2}")
    
    # Figure 3: Baseline Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    models = df_b["model"].tolist()
    mres = df_b["macro_mre_mm"].tolist()
    clean_names = [m.replace("_", " ").replace("C0 ", "").replace("C5 ", "").replace("C2 ", "").replace("C3 ", "").replace("C4 ", "") for m in models]
    b_colors = ["#999999", "#777777", "#4472c4", "#ed7d31", "#2ca02c"][:len(models)]
    
    bars = ax.bar(clean_names, mres, color=b_colors, edgecolor="black", width=0.5, zorder=3)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_ylabel("Macro MRE (mm)")
    ax.set_title("CT-ORG External Benchmark: Proposed Architecture vs Baselines")
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} mm",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")
                    
    ax.set_ylim(0, max(mres) * 1.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    p3 = fig_dir / "ctorg_baseline_comparison.png"
    plt.savefig(p3)
    plt.close()
    print(f"Saved {p3}")
    
    # Figure 4: Cross-cohort Comparison
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cohorts = ["Internal V3 (104 tgts)", "Internal Matched", "CT-ORG External", "FLARE22 Corrected"]
    vals = [
        d_final["INTERNAL_104_TARGET_MRE_MM"],
        d_final["INTERNAL_MATCHED_TARGET_MRE_MM"],
        d_final["CTORG_MATCHED_TARGET_MRE_MM"],
        d_final["FLARE_CORRECTED_MRE_MM"]
    ]
    c_cols = ["#1f77b4", "#aec7e8", "#2ca02c", "#ff7f0e"]
    bars = ax.bar(cohorts, vals, color=c_cols, edgecolor="black", width=0.5, zorder=3)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_ylabel("Macro MRE (mm)")
    ax.set_title("Cross-Cohort Generalization Benchmark")
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} mm",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")
                    
    ax.set_ylim(0, max(vals) * 1.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p4 = fig_dir / "ctorg_cross_cohort_comparison.png"
    plt.savefig(p4)
    plt.close()
    print(f"Saved {p4}")

if __name__ == "__main__":
    main()
