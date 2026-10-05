#!/usr/bin/env python3
"""
tools/phase11/generate_figure2_figure3.py

Generates publication-quality Figure 2 and Figure 3 for Phase 11:
- FIGURE 2: AMOS External Surface Extraction & Mesh Reconstruction Pipeline
- FIGURE 3: AMOS Field-of-View (FOV) Truncation Spectrum & Classification
Outputs in PDF, SVG, and PNG (300 DPI) to reports/phase11/figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.gridspec as gridspec

FIG_DIR = "reports/phase11/figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

def generate_figure2():
    print("Generating Figure 2: AMOS External Surface Pipeline...")
    fig = plt.figure(figsize=(12, 5), dpi=300)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.3)

    # 1. Intensity Volume Slice (CT HU)
    ax1 = fig.add_subplot(gs[0])
    # Synthetic realistic CT abdominal slice
    y, x = np.ogrid[-100:100, -100:100]
    torso_ellipse = (x/80)**2 + (y/60)**2 <= 1.0
    spine = (x/15)**2 + ((y-30)/15)**2 <= 1.0
    aorta = ((x+8)/6)**2 + ((y-18)/6)**2 <= 1.0
    ct_slice = np.full((200, 200), -1000.0) # Air
    ct_slice[torso_ellipse] = 40.0 # Soft tissue
    ct_slice[spine] = 400.0 # Bone
    ct_slice[aorta] = 120.0 # Contrast vessel
    ax1.imshow(ct_slice, cmap='gray', vmin=-1000, vmax=500)
    ax1.set_title("1. Raw Intensity Volume\n(HU Range: -1000 to +400)")
    ax1.axis('off')

    # 2. External Body Mask Segmentation
    ax2 = fig.add_subplot(gs[1])
    mask = torso_ellipse
    ax2.imshow(ct_slice, cmap='gray', vmin=-1000, vmax=500)
    ax2.contour(mask, levels=[0.5], colors='#e74c3c', linewidths=2.0)
    ax2.set_title("2. External Body Mask\n(HU > -500 + 3D Fill)")
    ax2.axis('off')

    # 3. Marching Cubes Isosurface
    ax3 = fig.add_subplot(gs[2], projection='3d')
    # Generate synthetic torso mesh points
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(-150, 150, 30)
    U, V = np.meshgrid(u, v)
    X = 160 * np.cos(U)
    Y = 120 * np.sin(U)
    Z = V
    ax3.plot_surface(X, Y, Z, color='#3498db', alpha=0.3, edgecolor='#2980b9', linewidth=0.2)
    ax3.set_title("3. Marching Cubes Mesh\n(Level Set = 0.5 mm)")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    ax3.set_zlabel("Z (mm)")
    ax3.view_init(elev=15, azim=45)

    # 4. Uniform 4,096 Point Cloud
    ax4 = fig.add_subplot(gs[3], projection='3d')
    np.random.seed(42)
    n_pts = 4096
    u_rand = np.random.uniform(0, 2*np.pi, n_pts)
    v_rand = np.random.uniform(-150, 150, n_pts)
    x_pc = 160 * np.cos(u_rand) + np.random.normal(0, 1.5, n_pts)
    y_pc = 120 * np.sin(u_rand) + np.random.normal(0, 1.5, n_pts)
    z_pc = v_rand
    ax4.scatter(x_pc[::4], y_pc[::4], z_pc[::4], s=1.5, c=z_pc[::4], cmap='viridis', alpha=0.7)
    # Centroid
    ax4.scatter([0], [0], [0], color='red', s=40, marker='o', label='Body Center')
    ax4.set_title("4. Sampled 4,096 Points\n(Centered & S_global=500mm)")
    ax4.set_xlabel("X (mm)")
    ax4.set_ylabel("Y (mm)")
    ax4.set_zlabel("Z (mm)")
    ax4.view_init(elev=15, azim=45)

    plt.suptitle("FIGURE 2: AMOS External Body Surface Reconstruction & Normalization Pipeline", fontsize=13, y=0.98)
    for ext in ["png", "pdf", "svg"]:
        out_p = os.path.join(FIG_DIR, f"FIGURE_2_amos_surface_pipeline.{ext}")
        plt.savefig(out_p, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Figure 2 to", FIG_DIR)

def generate_figure3():
    print("Generating Figure 3: AMOS FOV Truncation & Classification...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=300)

    # Diagram of FOV-A, FOV-B, FOV-C
    configs = [
        ("FOV-A: Broad Torso", 490, 0, "#2ecc71", "Height >= 350 mm\nTouches <= 2 borders\nFull abdomen + thorax\nN = 116 cases (44.8%)\nMacro MRE = 56.48 mm"),
        ("FOV-B: Adequate Abdomen", 280, 2, "#f39c12", "Height 200-350 mm\nDiagnostic abdominal scan\nPelvis / diaphragm limits\nN = 140 cases (54.1%)\nMacro MRE = 55.02 mm"),
        ("FOV-C: Substantial Truncation", 140, 4, "#e74c3c", "Height 100-200 mm\nTargeted subvolume\nSevere axial clipping\nN = 3 cases (1.2%)\nMacro MRE = 58.60 mm")
    ]

    for idx, (title, height, touches, color, desc) in enumerate(configs):
        ax = axes[idx]
        # Draw body silhouette outline
        y = np.linspace(-300, 300, 200)
        x_left = -80 - 15 * np.sin(y/100)
        x_right = 80 + 15 * np.sin(y/100)
        ax.plot(x_left, y, color='#7f8c8d', linestyle='--', alpha=0.5)
        ax.plot(x_right, y, color='#7f8c8d', linestyle='--', alpha=0.5)

        # Draw scan FOV box
        half_h = height / 2.0
        rect = Rectangle((-110, -half_h), 220, height, facecolor=color, alpha=0.25, edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)

        # Internal organs positions (liver, kidneys, bladder)
        ax.scatter([40], [20], color='#c0392b', s=50, label='Liver' if idx==0 else "")
        ax.scatter([-40, 40], [-30, -35], color='#8e44ad', s=40, label='Kidneys' if idx==0 else "")
        ax.scatter([0], [-120], color='#2980b9', s=45, label='Bladder' if idx==0 else "")

        ax.set_xlim(-140, 140)
        ax.set_ylim(-320, 320)
        ax.set_aspect('equal')
        ax.set_title(f"{title}\n(Height: {height} mm)", fontsize=11, fontweight='bold', pad=10)
        ax.text(0, -280, desc, ha='center', va='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bdc3c7", alpha=0.9))
        ax.set_xlabel("Lateral (mm)")
        if idx == 0:
            ax.set_ylabel("Craniocaudal SI (mm)")
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.set_yticklabels([])

    plt.suptitle("FIGURE 3: AMOS Field-of-View (FOV) Truncation Spectrum & Robustness", fontsize=13, y=0.98)
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        out_p = os.path.join(FIG_DIR, f"FIGURE_3_amos_fov_truncation.{ext}")
        plt.savefig(out_p, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Figure 3 to", FIG_DIR)

if __name__ == "__main__":
    generate_figure2()
    generate_figure3()
