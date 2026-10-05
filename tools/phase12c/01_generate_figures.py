#!/usr/bin/env python3
"""
Phase 12C: Manuscript Figure Generation
========================================
Generates:
1. Figure 1: Cross-Cohort Consistency (System 0 vs System 1 vs System 3) across V2, V3, and AMOS cohorts with 95% bootstrap CIs.
2. Figure 2: Target-Wise Consistency (8 semantically equivalent targets across V2, V3, AMOS) for deployable System 1.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': False
})

out_dir = repo_root / "reports" / "phase12c"
tab_dir = out_dir / "tables"
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

def generate_figure_1():
    t2_path = tab_dir / "TABLE2_system0_vs_system1.csv"
    if not t2_path.exists():
        print(f"Table 2 not found at {t2_path}")
        return

    df2 = pd.read_csv(t2_path)
    
    # Load summary for error bars
    with open(out_dir / "PHASE12C_SUMMARY.json") as f:
        summary = json.load(f)

    cohorts = ["V2 locked test\n(Internal Reference Domain)", "Dataset V3 locked test\n(Source Domain)", "AMOS-22 External\n(Zero-Shot External)"]
    sys0_means = [summary["v2_sys0"]["macro_mre"], summary["v3_sys0"]["macro_mre"], summary["amos_sys0"]["macro_mre"]]
    sys0_cis = [
        [summary["v2_sys0"]["macro_mre"] - summary["v2_sys0"]["ci"][0], summary["v2_sys0"]["ci"][1] - summary["v2_sys0"]["macro_mre"]],
        [summary["v3_sys0"]["macro_mre"] - summary["v3_sys0"]["ci"][0], summary["v3_sys0"]["ci"][1] - summary["v3_sys0"]["macro_mre"]],
        [summary["amos_sys0"]["macro_mre"] - summary["amos_sys0"]["ci"][0], summary["amos_sys0"]["ci"][1] - summary["amos_sys0"]["macro_mre"]],
    ]

    sys1_means = [summary["v2_sys1"]["macro_mre"], summary["v3_sys1"]["macro_mre"], summary["amos_sys1"]["macro_mre"]]
    sys1_cis = [
        [summary["v2_sys1"]["macro_mre"] - summary["v2_sys1"]["ci"][0], summary["v2_sys1"]["ci"][1] - summary["v2_sys1"]["macro_mre"]],
        [summary["v3_sys1"]["macro_mre"] - summary["v3_sys1"]["ci"][0], summary["v3_sys1"]["ci"][1] - summary["v3_sys1"]["macro_mre"]],
        [summary["amos_sys1"]["macro_mre"] - summary["amos_sys1"]["ci"][0], summary["amos_sys1"]["ci"][1] - summary["amos_sys1"]["macro_mre"]],
    ]

    yerr_sys0 = np.array(sys0_cis).T
    yerr_sys1 = np.array(sys1_cis).T

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)
    x = np.arange(len(cohorts))
    width = 0.35

    rects1 = ax.bar(x - width/2, sys0_means, width, yerr=yerr_sys0, capsize=5,
                    label="System 0 (Internally-Referenced / Retrospective)", color="#94a3b8", edgecolor="#475569", linewidth=1.2)
    rects2 = ax.bar(x + width/2, sys1_means, width, yerr=yerr_sys1, capsize=5,
                    label="System 1 (Phase10R + External Frame — Deployable)", color="#2563eb", edgecolor="#1e40af", linewidth=1.2)

    # Threshold lines
    ax.axhline(32.0, color="#f59e0b", linestyle="--", linewidth=1.5, alpha=0.8, label="Consistency Target Threshold (32.0 mm)")
    ax.axhline(25.0, color="#10b981", linestyle=":", linewidth=1.5, alpha=0.8, label="Strict Clinical Utility Bound (25.0 mm)")

    ax.set_ylabel("Macro Mean Radial Error (mm, Exact 8 Targets)", fontweight="bold")
    ax.set_title("Phase 12C: Cross-Cohort Localization Consistency Audit\nRetrospective Internal Reference vs. Fully External Deployable Pipeline", fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, fontweight="bold")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="upper right")
    ax.set_ylim(0, 68)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Add data labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f} mm", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#334155")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f} mm", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1e3a8a")

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_dir / f"fig_cross_cohort_system0_vs_system3.{ext}", dpi=300)
    print(f"Figure 1 saved to {fig_dir / 'fig_cross_cohort_system0_vs_system3.png'}")
    plt.close()

def generate_figure_2():
    t3_path = tab_dir / "TABLE3_target_wise_system1.csv"
    if not t3_path.exists():
        print(f"Table 3 not found at {t3_path}")
        return

    df3 = pd.read_csv(t3_path)
    # df3 has: Target, V2_Sys1_mm, V3_Sys1_mm, AMOS_Sys1_mm

    targets = df3["Target"].tolist()
    # Clean display names
    target_labels = [t.replace("_", " ").title() for t in targets]

    x = np.arange(len(targets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300)

    rects1 = ax.bar(x - width, df3["V2_Sys1_mm"], width, label="V2 Locked Test", color="#0ea5e9", edgecolor="#0284c7")
    rects2 = ax.bar(x, df3["V3_Sys1_mm"], width, label="Dataset V3 Locked Test", color="#6366f1", edgecolor="#4f46e5")
    rects3 = ax.bar(x + width, df3["AMOS_Sys1_mm"], width, label="AMOS-22 External Cohort", color="#ec4899", edgecolor="#db2777")

    ax.set_ylabel("Mean Radial Error (mm)", fontweight="bold")
    ax.set_title("Phase 12C: Target-Wise Cross-Cohort Consistency (System 1 Deployable Pipeline)", fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=25, ha="right", fontweight="bold")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(0, max(df3[["V2_Sys1_mm", "V3_Sys1_mm", "AMOS_Sys1_mm"]].max()) * 1.25)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_dir / f"fig_target_wise_consistency.{ext}", dpi=300)
    print(f"Figure 2 saved to {fig_dir / 'fig_target_wise_consistency.png'}")
    plt.close()

if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
