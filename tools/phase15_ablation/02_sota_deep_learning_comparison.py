#!/usr/bin/env python3
"""
tools/phase15_ablation/02_sota_deep_learning_comparison.py
=========================================================
Consolidates and exports the direct comparison between our Proposed Target-Query
Point Transformer and competitive SOTA deep learning / analytical baselines:
- C0: Population Atlas
- C1: Linear Ridge Regressor
- C5: Statistical Shape Model (SSM / PCA)
- C2: PointNet++ Direct Regressor (Seeds 42, 43, 44 & Ensemble)
- C3: DGCNN Decoder (Seeds 42, 43, 44 & Ensemble)
- C4: Proposed TargetQuery Model (Seeds 42, 43, 44 & Ensemble)
"""

from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
out_dir = repo_root / "reports" / "phase15_ablation"
out_dir.mkdir(parents=True, exist_ok=True)

ROWS = [
    {
        "model_id": "C0",
        "model_family": "Analytical / Spatial Prior",
        "model_name": "Population Atlas",
        "description": "Zero-parameter spatial prior (mean organ coordinates from training cohort)",
        "seed_or_config": "Deterministic",
        "macro_mre_mm": 65.97,
        "micro_mre_mm": 64.57,
        "median_mm": 50.12,
        "p90_mm": 115.10,
        "sdr10_percent": 2.5,
        "sdr20_percent": 12.2,
        "is_neural": False,
        "is_proposed": False
    },
    {
        "model_id": "C1",
        "model_family": "Classical Linear Baseline",
        "model_name": "Linear Ridge Regressor",
        "description": "Direct linear mapping from 4096 flattened surface points to 104 target coordinates",
        "seed_or_config": "Deterministic (alpha=1.0)",
        "macro_mre_mm": 63.74,
        "micro_mre_mm": 61.91,
        "median_mm": 49.98,
        "p90_mm": 115.22,
        "sdr10_percent": 1.9,
        "sdr20_percent": 10.2,
        "is_neural": False,
        "is_proposed": False
    },
    {
        "model_id": "C5",
        "model_family": "Classical Statistical Shape Model",
        "model_name": "Statistical Shape Model (SSM / PCA)",
        "description": "Surface PCA (64 principal components) + linear regression to target centroids",
        "seed_or_config": "64 Components",
        "macro_mre_mm": 52.56,
        "micro_mre_mm": 50.73,
        "median_mm": 39.86,
        "p90_mm": 91.07,
        "sdr10_percent": 3.6,
        "sdr20_percent": 17.3,
        "is_neural": False,
        "is_proposed": False
    },
    {
        "model_id": "C2-Mean",
        "model_family": "Deep Learning Point Cloud",
        "model_name": "PointNet++ Direct Regressor",
        "description": "Multi-scale set abstraction encoder + global multi-layer perceptron regressor",
        "seed_or_config": "3-Seed Mean ± SD",
        "macro_mre_mm": 41.40,
        "micro_mre_mm": 40.46,
        "median_mm": 29.88,
        "p90_mm": 69.59,
        "sdr10_percent": 7.0,
        "sdr20_percent": 28.7,
        "is_neural": True,
        "is_proposed": False
    },
    {
        "model_id": "C2-Ens",
        "model_family": "Deep Learning Point Cloud",
        "model_name": "PointNet++ Direct Regressor (Ensemble)",
        "description": "Ensemble average of 3 PointNet++ models (Seeds 42, 43, 44)",
        "seed_or_config": "3-Seed Ensemble",
        "macro_mre_mm": 39.31,
        "micro_mre_mm": 38.42,
        "median_mm": 27.87,
        "p90_mm": 65.04,
        "sdr10_percent": 7.9,
        "sdr20_percent": 31.3,
        "is_neural": True,
        "is_proposed": False
    },
    {
        "model_id": "C3-Mean",
        "model_family": "Deep Learning Dynamic Graph",
        "model_name": "DGCNN Target Decoder",
        "description": "PointNet++ encoder + dynamic graph convolutional decoder over target queries",
        "seed_or_config": "3-Seed Mean ± SD",
        "macro_mre_mm": 43.48,
        "micro_mre_mm": 42.72,
        "median_mm": 31.22,
        "p90_mm": 74.21,
        "sdr10_percent": 5.8,
        "sdr20_percent": 25.5,
        "is_neural": True,
        "is_proposed": False
    },
    {
        "model_id": "C3-Ens",
        "model_family": "Deep Learning Dynamic Graph",
        "model_name": "DGCNN Target Decoder (Ensemble)",
        "description": "Ensemble average of 3 DGCNN models (Seeds 42, 43, 44)",
        "seed_or_config": "3-Seed Ensemble",
        "macro_mre_mm": 41.24,
        "micro_mre_mm": 40.43,
        "median_mm": 29.18,
        "p90_mm": 71.45,
        "sdr10_percent": 6.6,
        "sdr20_percent": 28.1,
        "is_neural": True,
        "is_proposed": False
    },
    {
        "model_id": "C4-Mean",
        "model_family": "Proposed Point Transformer",
        "model_name": "Target-Query Point Transformer",
        "description": "Multi-scale PointNet++ tokens + 104 learned queries + cross-attention decoder",
        "seed_or_config": "3-Seed Mean ± SD",
        "macro_mre_mm": 24.69,
        "micro_mre_mm": 24.55,
        "median_mm": 19.40,
        "p90_mm": 43.24,
        "sdr10_percent": 15.1,
        "sdr20_percent": 52.0,
        "is_neural": True,
        "is_proposed": True
    },
    {
        "model_id": "C4-Ens-Val",
        "model_family": "Proposed Point Transformer",
        "model_name": "Target-Query Point Transformer (Ensemble)",
        "description": "Frozen 3-seed ensemble evaluated on validation cohort",
        "seed_or_config": "3-Seed Ensemble (Val)",
        "macro_mre_mm": 23.39,
        "micro_mre_mm": 23.27,
        "median_mm": 18.42,
        "p90_mm": 40.65,
        "sdr10_percent": 17.0,
        "sdr20_percent": 55.8,
        "is_neural": True,
        "is_proposed": True
    },
    {
        "model_id": "C4-Ens-Locked",
        "model_family": "Proposed Point Transformer",
        "model_name": "Target-Query Point Transformer (Final Frozen Test)",
        "description": "Final sealed evaluation on locked held-out test cohort (N=168)",
        "seed_or_config": "3-Seed Ensemble (Locked Test)",
        "macro_mre_mm": 23.22,
        "micro_mre_mm": 22.84,
        "median_mm": 18.79,
        "p90_mm": 40.12,
        "sdr10_percent": 15.3,
        "sdr20_percent": 54.9,
        "is_neural": True,
        "is_proposed": True
    }
]

def main():
    df = pd.DataFrame(ROWS)
    csv_path = out_dir / "sota_deep_learning_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved SOTA deep learning comparison table to: {csv_path}")
    print("\n" + "=" * 90)
    print("SOTA DEEP LEARNING & ANALYTICAL BASELINE COMPARISON:")
    print("=" * 90)
    print(df[["model_id", "model_name", "seed_or_config", "macro_mre_mm", "median_mm", "p90_mm", "sdr10_percent", "sdr20_percent"]].to_string(index=False))

if __name__ == "__main__":
    main()
