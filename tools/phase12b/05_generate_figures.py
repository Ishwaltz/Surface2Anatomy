#!/usr/bin/env python3
"""
Task 11: Phase 12B Publication Figures
======================================
Generates:
  - Figure 1: reports/phase12b/figures/fig1_v2_amos_system_comparison.png
  - Figure 2: reports/phase12b/figures/fig2_signed_axis_bias_comparison.png
  - Figure 3: reports/phase12b/figures/fig3_v2_fov_crop_drift_mre.png
  - Figure 4: reports/phase12b/figures/fig4_exact_8_target_comparison.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

repo_root = Path(__file__).resolve().parent.parent.parent
fig_dir = repo_root / "reports" / "phase12b" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.autolayout': False
})

def main():
    print("Generating Phase 12B Publication Figures...")

    # -------------------------------------------------------------
    # Figure 1: V2 vs AMOS Multi-System Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    systems = ["System 0\n(Phase10R Original)", "System 1\n(New Ext Frame)", "System 3\n(FOV + New Frame)"]
    v2_mre = [18.68, 27.23, 27.23]
    amos_mre = [55.72, 27.25, 23.98]

    x = np.arange(len(systems))
    width = 0.35

    rects1 = ax.bar(x - width/2, v2_mre, width, label='V2 In-Domain Sanity (Locked Test N=41)', color='#2b5c8f', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, amos_mre, width, label='AMOS-22 External Validation (N=180)', color='#d95f02', edgecolor='black', linewidth=0.8)

    ax.axhline(30.0, color='red', linestyle='--', linewidth=1.2, label='Clinical Acceptability Threshold (30 mm)')
    ax.set_ylabel('Macro MRE (mm)', fontweight='bold')
    ax.set_title('Figure 1: V2 In-Domain Sanity vs AMOS-22 External Cohort Across Systems', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontweight='bold')
    ax.set_ylim(0, 65)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f} mm', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f} mm', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_v2_amos_system_comparison.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Signed Axis Bias (V2 vs AMOS Raw)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    axes_labels = ["dx (Lateral)", "dy (Anterior-Posterior)", "dz (Superior-Inferior)"]
    amos_raw_bias = [+0.02, +46.22, -15.40]
    v2_sys0_bias = [+0.89, +0.90, +1.36]

    x = np.arange(len(axes_labels))
    width = 0.35

    r1 = ax.bar(x - width/2, v2_sys0_bias, width, label='V2 System 0 (Original)', color='#2b5c8f', edgecolor='black', linewidth=0.8)
    r2 = ax.bar(x + width/2, amos_raw_bias, width, label='AMOS Raw Phase 11', color='#e7298a', edgecolor='black', linewidth=0.8)

    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_ylabel('Mean Signed Error (Pred - GT, mm)', fontweight='bold')
    ax.set_title('Figure 2: Signed Coordinate Bias: V2 Baseline vs AMOS Raw Phase 11', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(axes_labels, fontweight='bold')
    ax.set_ylim(-25, 55)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')

    for rect in r1:
        h = rect.get_height()
        offset = 3 if h >= 0 else -12
        ax.annotate(f'{h:+.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, offset),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in r2:
        h = rect.get_height()
        offset = 3 if h >= 0 else -12
        ax.annotate(f'{h:+.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, offset),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(fig_dir / "fig2_signed_axis_bias_comparison.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: V2 FOV Cropping Origin Drift & MRE
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    df_stress = pd.read_csv(repo_root / "reports/phase12b/tables/V2_FOV_STRESS_RESULTS.csv")

    conds = df_stress["Condition"].tolist()
    x = np.arange(len(conds))

    # Panel A: Origin Drift
    ax1.plot(x, df_stress["Old_Frame_Drift_mm"], marker='o', color='#d95f02', linewidth=2.0, label='Old Frame (Naive Midpoint)')
    ax1.plot(x, df_stress["New_Frame_Drift_mm"], marker='s', color='#2b5c8f', linewidth=2.0, label='New External Frame')
    ax1.set_title('A: Origin Drift Under Axial FOV Cropping', fontweight='bold')
    ax1.set_ylabel('Origin Drift vs Spine Reference (mm)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conds, rotation=25, ha='right')
    ax1.set_ylim(0, 135)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True)

    for i in range(len(conds)):
        ax1.annotate(f"{df_stress['Drift_Reduction_Pct'].iloc[i]:.0f}% red", 
                     xy=(x[i], df_stress['New_Frame_Drift_mm'].iloc[i]),
                     xytext=(0, 6), textcoords="offset points", ha='center', fontsize=8, color='#2b5c8f', fontweight='bold')

    # Panel B: Macro MRE
    ax2.plot(x, df_stress["Old_Frame_MRE_mm"], marker='o', color='#d95f02', linewidth=2.0, label='Old Frame (Naive Centering)')
    ax2.plot(x, df_stress["New_Frame_MRE_mm"], marker='s', color='#2b5c8f', linewidth=2.0, label='New External Frame')
    ax2.axhline(30.0, color='red', linestyle='--', linewidth=1.0, label='30 mm Threshold')
    ax2.set_title('B: Macro MRE Under Axial FOV Cropping', fontweight='bold')
    ax2.set_ylabel('Macro MRE (mm)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conds, rotation=25, ha='right')
    ax2.set_ylim(20, 100)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig3_v2_fov_crop_drift_mre.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 4: Exact 8-Target Comparison
    # -------------------------------------------------------------
    df_exact = pd.read_csv(repo_root / "reports/phase12b/04_EXACT_TARGET_COMPARISON.csv")
    df_exact_sub = df_exact[df_exact["target_name"] != "MACRO_MRE"].copy()

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    names = df_exact_sub["target_name"].tolist()
    x = np.arange(len(names))
    width = 0.20

    ax.bar(x - 1.5*width, df_exact_sub["v2_sys0_mre_mm"], width, label='V2 Sys 0 (Baseline)', color='#2b5c8f')
    ax.bar(x - 0.5*width, df_exact_sub["v3_test_mre_mm"], width, label='V3 Test Sys 0', color='#7570b3')
    ax.bar(x + 0.5*width, df_exact_sub["amos_sys0_mre_mm"], width, label='AMOS Sys 0 (Raw)', color='#e7298a')
    ax.bar(x + 1.5*width, df_exact_sub["amos_sys3_mre_mm"], width, label='AMOS Sys 3 (Combined)', color='#1b9e77')

    ax.axhline(30.0, color='red', linestyle='--', linewidth=1.0, label='30 mm Clinical Target')
    ax.set_ylabel('Target MRE (mm)', fontweight='bold')
    ax.set_title('Figure 4: Exact 8-Target Breakdown Across V2, V3, and AMOS Deployable Systems', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontweight='bold')
    ax.set_ylim(0, 65)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, ncol=3, loc='upper right')

    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_exact_8_target_comparison.png")
    plt.close()

    print(f"All 4 figures saved to: {fig_dir}")

if __name__ == "__main__":
    main()
