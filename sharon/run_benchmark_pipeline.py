import os
import sys
import time
import subprocess
from pathlib import Path

# Add paths
sharon_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(sharon_dir))
sys.path.insert(0, str(sharon_dir.parent))

def run_step(cmd_list, desc):
    print("\n" + "=" * 70)
    print(f" [STEP] {desc}")
    print(f" [CMD]  {' '.join(cmd_list)}")
    print("=" * 70)
    t0 = time.time()
    res = subprocess.run(cmd_list)
    if res.returncode != 0:
        print(f"[ERROR] Step failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"✓ Step completed in {time.time()-t0:.1f}s")

def main():
    py_exec = sys.executable
    out_dir = Path("sharon/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "benchmark_pipeline.log"

    print("=" * 70)
    print(" STARTING END-TO-END BENCHMARK PIPELINE")
    print(f" Python: {py_exec}")
    print(f" Output: {out_dir}")
    print("=" * 70)

    # 1. Point Cloud Sampling if not yet done
    pt_path = Path("sharon/dataset/pointclouds_450.pt")
    if not pt_path.exists():
        run_step(
            [py_exec, "sharon/pointcloud_sampler.py", "--dataset_dir", "sharon/dataset", "--output_file", str(pt_path), "--num_workers", "16"],
            "1. Parallel Point Cloud Sampling (N=4096, K=121, 450 cases)"
        )
    else:
        print(f"✓ {pt_path} already exists. Skipping sampling.")

    # 2. Train All 3 Benchmark Models
    run_step(
        [py_exec, "sharon/train_benchmarks.py", "--data_file", str(pt_path), "--output_dir", "sharon/outputs", "--epochs", "200", "--batch_size", "16", "--lr", "2e-4"],
        "2. Train 3 Benchmark Models (PointNet++, DGCNN, Hierarchical Sex-GNN) for 200 Epochs"
    )

    # 3. Quantitative Test Evaluation & Report Generation
    run_step(
        [py_exec, "sharon/evaluate_gnn.py", "--data_file", str(pt_path), "--splits_file", "sharon/outputs/splits_pointcloud.json", "--output_report", "sharon/outputs/GNN_BENCHMARK_REPORT.md", "--output_json", "sharon/outputs/GNN_BENCHMARK_RESULTS.json"],
        "3. Evaluate Held-Out Test Set (45 cases) and Generate GNN_BENCHMARK_REPORT.md"
    )

    print("\n" + "=" * 70)
    print(" PIPELINE FULLY COMPLETED")
    print(f" Report: {out_dir / 'GNN_BENCHMARK_REPORT.md'}")
    print(f" Results JSON: {out_dir / 'GNN_BENCHMARK_RESULTS.json'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
