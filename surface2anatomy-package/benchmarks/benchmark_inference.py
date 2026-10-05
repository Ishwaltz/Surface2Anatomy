"""
Surface2Anatomy Latency, Throughput, and Memory Profiler.
"""

import time
import numpy as np
import torch
from surface2anatomy import SurfaceAnatomyModel

def run_benchmark():
    print("=" * 60)
    print("SURFACE2ANATOMY INFERENCE LATENCY & MEMORY BENCHMARK")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on device: {device.upper()}")

    model = SurfaceAnatomyModel.from_pretrained(model_variant="phase16_brain", device=device)
    pts = np.random.randn(4096, 3).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = model.predict(pts, target="spleen", units="mm")

    # Measure latency for single target
    runs = 30
    latencies_single = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict(pts, target="spleen", units="mm")
        latencies_single.append((time.perf_counter() - t0) * 1000.0)

    # Measure latency for multi-target query (10 targets)
    latencies_multi = []
    targets_10 = ["liver", "spleen", "kidney_left", "kidney_right", "heart", "brain", "pancreas", "stomach", "aorta", "trachea"]
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict_multiple(pts, targets=targets_10, units="mm")
        latencies_multi.append((time.perf_counter() - t0) * 1000.0)

    print("\nBenchmark Results (N=30 iterations):")
    print(f"  Single Target Mean Latency:  {np.mean(latencies_single):.2f} ± {np.std(latencies_single):.2f} ms")
    print(f"  10-Target Mean Latency:      {np.mean(latencies_multi):.2f} ± {np.std(latencies_multi):.2f} ms")
    print(f"  Throughput:                  {1000.0 / np.mean(latencies_multi):.1f} queries/sec")

if __name__ == "__main__":
    run_benchmark()
