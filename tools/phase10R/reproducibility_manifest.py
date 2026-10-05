import os
import sys
import json
import hashlib
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

def compute_sha256(filepath):
    p = Path(filepath)
    if not p.exists():
        return f"FILE_NOT_FOUND: {filepath}"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Computing Phase 10R Reproducibility Manifest...")
    
    files_to_hash = {
        "dataset_pointclouds_v3": repo_root / "sharon/dataset_v3/pointclouds_v3.pt",
        "dataset_v3_frozen_spec": repo_root / "sharon/dataset_v3/DATASET_V3_FROZEN.json",
        "dataset_v3_version_spec": repo_root / "sharon/dataset_v3/DATASET_V3_VERSION.json",
        "splits_v3_iid": repo_root / "sharon/dataset_v3/splits_v3_iid.json",
        "target_ontology_v3": repo_root / "sharon/dataset_v3/target_ontology_v3.csv",
        "benchmark_primary_targets_v3": repo_root / "sharon/dataset_v3/benchmark_primary_targets_v3.json",
        "dataset_manifest_v3": repo_root / "sharon/dataset_v3/manifest_v3.csv",
        "model_source_target_query": repo_root / "sharon/model_target_query.py",
        "model_source_gnn": repo_root / "sharon/model_gnn.py",
        "training_source_train": repo_root / "sharon/train.py",
        "scaling_experiments_runner": repo_root / "tools/phase9/run_phase9b_scaling_experiments.py",
        "phase10_master_runner": repo_root / "tools/phase10/run_phase10_master.py"
    }
    
    hashes = {}
    for name, path in files_to_hash.items():
        h = compute_sha256(path)
        hashes[name] = {
            "path": str(path.relative_to(repo_root)),
            "sha256": h,
            "exists": Path(path).exists(),
            "size_bytes": Path(path).stat().st_size if Path(path).exists() else 0
        }
        print(f"  {name}: {h[:16]}... ({hashes[name]['size_bytes'] / (1024*1024):.2f} MB)")
        
    manifest = {
        "manifest_name": "PHASE10R_REPRODUCIBILITY_MANIFEST",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixed_scientific_context": {
            "dataset_version": "Dataset V3.0.0-FROZEN",
            "total_cases": 1668,
            "split_counts": {
                "train": 1334,
                "validation": 166,
                "locked_test": 168
            },
            "primary_benchmark_targets_count": 104,
            "ontology_total_targets": 117,
            "global_normalization_scale_mm": 500.0,
            "canonical_coordinate_frame": "+X Right, +Y Anterior, +Z Superior",
            "canonical_training_recipe": {
                "epochs": 65,
                "batch_size": 16,
                "optimizer": "AdamW",
                "encoder_lr": 2e-4,
                "decoder_lr": 5e-4,
                "weight_decay": 1e-4,
                "lr_scheduler": "CosineAnnealingLR(T_max=65)",
                "surface_jitter_train_sigma_mm": 0.5,
                "checkpoint_selection_rule": "Lowest validation macro MRE on primary 104 benchmark targets"
            },
            "hardware_context": {
                "gpu": "NVIDIA GeForce RTX 4070 Ti SUPER",
                "vram_gb": 16,
                "driver_version": "595.84",
                "cuda_version": "13.2"
            }
        },
        "file_hashes": hashes
    }
    
    out_dir = repo_root / "reports" / "phase10R"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "00_reproducibility_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nSaved manifest to {out_path}")

if __name__ == "__main__":
    main()
