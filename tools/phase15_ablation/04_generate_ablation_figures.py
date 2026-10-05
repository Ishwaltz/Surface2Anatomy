#!/usr/bin/env python3
"""
tools/phase15_ablation/04_generate_ablation_figures.py
======================================================
Generates publication-quality figures for Phase 15:
1. FIGURE_surface_occlusion_macro_impact.png (Bar chart of occlusion impact)
2. FIGURE_organ_vulnerability_heatmap.png (Top organ vulnerabilities across covered regions)
3. FIGURE_sota_model_comparison.png (Comparison against PointNet++, DGCNN, Atlas, SSM)
4. FIGURE_literature_matched_subsets.png (1:1 matched anatomy comparison)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent.parent.parent
reports_dir = repo_root / "reports" / "phase15_ablation"
fig_dir = reports_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0

def plot_occlusion_impact():
    df = pd.read_csv(reports_dir / "surface_occlusion_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    modes = [
        "Full_360_Reference", "Central_Drape_Covered", "Posterior_Covered",
        "Right_Flank_Covered", "Left_Flank_Covered", "Anterior_Covered",
        "Inferior_Covered", "Superior_Covered"
    ]
    labels = [
        "Full 360° (Reference)", "Central Drape (Surgical)", "Posterior (Back Hidden)",
        "Right Flank Hidden", "Left Flank Hidden", "Anterior (Front Hidden)",
        "Inferior (Pelvis Hidden)", "Superior (Chest Hidden)"
    ]
    
    sub = df.set_index("mode_id").loc[modes].reset_index()
    y_pos = np.arange(len(modes))
    colors = ["#2e7d32", "#43a047", "#f57c00", "#fb8c00", "#e65100", "#d32f2f", "#c2185b", "#b71c1c"]
    
    bars = ax.barh(y_pos, sub["macro_mre_mm"], color=colors, height=0.65, edgecolor="none")
    ax.axvline(23.22, color="#2e7d32", linestyle="--", linewidth=1.5, alpha=0.7, label="Reference (23.2 mm)")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight="medium")
    ax.invert_yaxis()
    ax.set_xlabel("Macro Mean Radial Error (MRE) in mm [Lower is Better]", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Surface Point Cloud Coverage & Occlusion Ablation Study (N=168)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 100)
    
    for bar, val, delta in zip(bars, sub["macro_mre_mm"], sub["delta_macro_mm"]):
        w = bar.get_width()
        d_str = f" (+{delta:.1f} mm)" if delta > 0 else " (Ref)"
        ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{val:.1f} mm{d_str}", 
                va="center", ha="left", fontsize=10, fontweight="bold", color="#222222")
        
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    p = fig_dir / "FIGURE_surface_occlusion_macro_impact.png"
    plt.savefig(p)
    plt.close()
    print(f"Saved: {p}")

def plot_organ_vulnerability():
    df = pd.read_csv(reports_dir / "surface_occlusion_per_target.csv")
    key_organs = [
        "liver", "spleen", "kidney_right", "kidney_left", "urinary_bladder",
        "stomach", "pancreas", "aorta", "trachea", "vertebrae_L1", "vertebrae_T1"
    ]
    key_modes = [
        "Posterior_Covered", "Anterior_Covered", "Superior_Covered",
        "Inferior_Covered", "Right_Flank_Covered", "Left_Flank_Covered", "Central_Drape_Covered"
    ]
    mode_labels = ["Back Hidden", "Front Hidden", "Chest Hidden", "Pelvis Hidden", "Right Hidden", "Left Hidden", "Drape (Belly)"]
    
    matrix = np.zeros((len(key_organs), len(key_modes)))
    for i, org in enumerate(key_organs):
        for j, m in enumerate(key_modes):
            row = df[(df["target_name"] == org) & (df["mode_id"] == m)]
            if len(row) > 0:
                matrix[i, j] = row.iloc[0]["delta_mre_mm"]
                
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cax = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(mode_labels)))
    ax.set_xticklabels(mode_labels, rotation=30, ha="right", fontsize=11, fontweight="medium")
    ax.set_yticks(np.arange(len(key_organs)))
    ax.set_yticklabels([o.replace("_", " ").title() for o in key_organs], fontsize=11, fontweight="medium")
    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Localization Error Increase Δ (mm)", fontsize=11, fontweight="bold")
    
    # Annotate values
    for i in range(len(key_organs)):
        for j in range(len(key_modes)):
            val = matrix[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"+{val:.1f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")
            
    ax.set_title("Organ Vulnerability Heatmap: Sensitivity to Surface Occlusion (Δ MRE in mm)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    p = fig_dir / "FIGURE_organ_vulnerability_heatmap.png"
    plt.savefig(p)
    plt.close()
    print(f"Saved: {p}")

def plot_sota_comparison():
    df = pd.read_csv(reports_dir / "sota_deep_learning_comparison.csv")
    models = ["Population Atlas", "Linear Ridge", "SSM / PCA", "DGCNN Decoder", "PointNet++ Direct", "Proposed Point Transformer"]
    mres = [65.97, 63.74, 52.56, 39.31, 41.24, 23.22] # ensemble / locked test
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    y_pos = np.arange(len(models))
    colors = ["#757575", "#78909c", "#8d6e63", "#1976d2", "#0288d1", "#2e7d32"]
    
    bars = ax.barh(y_pos, mres, color=colors, height=0.6, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11, fontweight="medium")
    ax.invert_yaxis()
    ax.set_xlabel("Macro MRE in mm [Lower is Better]", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Model Architecture Comparison: Deep Learning vs Analytical Baselines", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 80)
    
    for bar, val in zip(bars, mres):
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{val:.2f} mm", va="center", ha="left", fontsize=11, fontweight="bold")
        
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    p = fig_dir / "FIGURE_sota_model_comparison.png"
    plt.savefig(p)
    plt.close()
    print(f"Saved: {p}")

def plot_matched_literature():
    df = pd.read_csv(reports_dir / "matched_literature_benchmarks.csv")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    labels = [
        "SAMe (arXiv 2026)\n[11 Visceral Organs]",
        "Atici et al. (MIDL 2026)\n[14 Visceral Organs]",
        "FLARE22 Zero-Shot\n[13 Abdominal Organs]",
        "CT-ORG Benchmark\n[4 Abdominal Organs]"
    ]
    our_vals = df["our_model_mre_on_matched_subset_mm"].tolist()
    pub_vals = df["published_value_on_their_data"].tolist()
    
    x = np.arange(len(labels))
    w = 0.35
    
    b1 = ax.bar(x - w/2, pub_vals, width=w, label="Published Benchmark (Own Cohort)", color="#78909c")
    b2 = ax.bar(x + w/2, our_vals, width=w, label="Our Frozen Model (Exact Matched Organs)", color="#2e7d32")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="medium")
    ax.set_ylabel("Mean Localization Error (mm)", fontsize=12, fontweight="bold")
    ax.set_title("Open-Source Competitive Literature: Exact 1:1 Matched Anatomy Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}", ha="center", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f} mm", ha="center", fontsize=9, fontweight="bold", color="#1b5e20")
        
    ax.set_ylim(0, 55)
    plt.tight_layout()
    p = fig_dir / "FIGURE_literature_matched_subsets.png"
    plt.savefig(p)
    plt.close()
    print(f"Saved: {p}")

def main():
    plot_occlusion_impact()
    plot_organ_vulnerability()
    plot_sota_comparison()
    plot_matched_literature()
    print("All 4 Phase 15 figures generated successfully!")

if __name__ == "__main__":
    main()
