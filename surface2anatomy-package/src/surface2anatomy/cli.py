"""
Surface2Anatomy Command-Line Interface (CLI).
Provides rich terminal diagnostics, model management, surface validation, and inference.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

from surface2anatomy._version import __version__
from surface2anatomy.constants import (
    NUM_POINTS, COORDINATE_FRAME_DESC, RUNTIME_IMAGING_DESC,
    TOTAL_CATALOG_TARGETS, NUM_PRIMARY_BENCHMARK_TARGETS
)
from surface2anatomy.targets import list_targets, TARGET_CATEGORIES
from surface2anatomy.io import load_surface
from surface2anatomy.model import SurfaceAnatomyModel
from surface2anatomy.download import ensure_model_artifact, get_default_cache_dir
from surface2anatomy.manifest import MODEL_MANIFEST

def print_header():
    print(f"Surface2Anatomy {__version__}")
    print("=" * 60)

def cmd_info(args):
    print_header()
    print("Scientific Task:")
    print("  Target-conditioned 3D Internal Anatomy Localization from External Body Surface Geometry")
    print("\nModel Specifications:")
    print(f"  Input Geometry:        {NUM_POINTS} surface points (XYZ)")
    print(f"  Coordinate Convention: {COORDINATE_FRAME_DESC}")
    print(f"  Runtime Imaging:       {RUNTIME_IMAGING_DESC}")
    print(f"  Supported Targets:     {TOTAL_CATALOG_TARGETS} targets ({NUM_PRIMARY_BENCHMARK_TARGETS} primary benchmark)")
    print("\nLocal Cache Directory:")
    print(f"  {get_default_cache_dir()}")
    print("\nEnsemble:")
    print("  Unweighted average of 3 frozen seeds (Seed 42, Seed 43, Seed 44)")

def cmd_list_targets(args):
    print_header()
    if args.category:
        targets = list_targets(category=args.category, benchmark_only=args.benchmark_only)
        print(f"Category: {args.category} ({len(targets)} targets)")
        for t in sorted(targets):
            print(f"  - {t}")
    else:
        print(f"Supported Targets Directory ({TOTAL_CATALOG_TARGETS} landmarks):\n")
        for cat, items in TARGET_CATEGORIES.items():
            if args.benchmark_only:
                items = [x for x in items if x in list_targets(benchmark_only=True)]
            print(f"[{cat}] ({len(items)} targets)")
            for it in items:
                print(f"  - {it}")
            print()

def cmd_validate_surface(args):
    print_header()
    path = Path(args.surface_file)
    print(f"Validating Surface: {path.name}")
    try:
        pts, mesh = load_surface(path)
        p_min = pts.min(axis=0)
        p_max = pts.max(axis=0)
        span = p_max - p_min
        print("  Status:               PASSED VALIDATION")
        print(f"  Raw Point Count:      {len(pts)}")
        print(f"  Geometry Type:        {'Mesh (' + str(len(mesh.faces)) + ' faces)' if mesh is not None and len(mesh.faces) > 0 else 'Point Cloud'}")
        print(f"  Bounding Box Min:     [{p_min[0]:.2f}, {p_min[1]:.2f}, {p_min[2]:.2f}]")
        print(f"  Bounding Box Max:     [{p_max[0]:.2f}, {p_max[1]:.2f}, {p_max[2]:.2f}]")
        print(f"  Physical Extents:     Width={span[0]:.2f}, Depth={span[1]:.2f}, Height={span[2]:.2f}")
    except Exception as e:
        print(f"  Status:               FAILED VALIDATION")
        print(f"  Error:                {e}")
        sys.exit(1)

def cmd_download_models(args):
    print_header()
    print("Checking and downloading pretrained model weights...\n")
    variant = args.variant or "phase16_brain"
    cache_dir = args.cache_dir

    # Download alignment
    align_meta = MODEL_MANIFEST["canonical_alignment"]["ridge_model"]
    print("1. Canonical Alignment Regressor:")
    p = ensure_model_artifact(align_meta, cache_dir=cache_dir)
    print(f"   ✓ Verified: {p.name}\n")

    # Download checkpoints
    var_meta = MODEL_MANIFEST[variant]
    print(f"2. {var_meta['description']}:")
    for s in [42, 43, 44]:
        s_meta = var_meta[f"seed{s}"]
        p = ensure_model_artifact(s_meta, cache_dir=cache_dir)
        print(f"   ✓ Verified: {p.name}")

    print("\nAll pretrained assets are downloaded and verified successfully!")

def cmd_predict(args):
    if getattr(args, "remote", False):
        from surface2anatomy.model import RemoteSurfaceAnatomyModel
        model = RemoteSurfaceAnatomyModel(endpoint=getattr(args, "endpoint", "https://sharonmelhi-surface2anatomy-backend.hf.space"))
    else:
        model = SurfaceAnatomyModel.from_pretrained(
            model_variant=args.variant,
            device=args.device,
            cache_dir=args.cache_dir,
            remote_endpoint=getattr(args, "endpoint", "https://sharonmelhi-surface2anatomy-backend.hf.space")
        )

    targets = args.targets if args.targets else [args.target]
    result = model.predict_multiple(
        surface_input=args.surface_file,
        targets=targets,
        units=args.units
    )

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() == ".json":
            with open(out_path, "w") as f:
                f.write(result.to_json())
            print(f"Saved results to {out_path}")
        elif out_path.suffix.lower() == ".csv":
            result.to_dataframe().to_csv(out_path, index=False)
            print(f"Saved results to {out_path}")
        else:
            with open(out_path, "w") as f:
                f.write(result.to_json())
            print(f"Saved results to {out_path}")
        return

    # Formatted terminal output
    print(f"\nSurface2Anatomy {__version__}\n")
    for t_name, p in result.items():
        print(f"Target: {p.target}")
        print("\nPredicted centroid:")
        print(f"X = {p.centroid_mm[0]:+8.2f} mm")
        print(f"Y = {p.centroid_mm[1]:+8.2f} mm")
        print(f"Z = {p.centroid_mm[2]:+8.2f} mm")
        print(f"\nEnsemble disagreement:\n{p.ensemble_disagreement_mm:.2f} mm")
        print(f"\nInput:\n{result.preprocessing.sampled_point_count} sampled surface points")
        print(f"\nRuntime imaging:\n{RUNTIME_IMAGING_DESC}\n")
        if len(result) > 1:
            print("-" * 40 + "\n")

def cmd_batch(args):
    dir_path = Path(args.surfaces_dir)
    if not dir_path.is_dir():
        print(f"Error: '{dir_path}' is not a valid directory.")
        sys.exit(1)

    surface_files = []
    for ext in (".ply", ".pcd", ".obj", ".stl", ".xyz", ".npy"):
        surface_files.extend(list(dir_path.glob(f"*{ext}")))

    if not surface_files:
        print(f"No supported 3D surfaces found in {dir_path}")
        sys.exit(1)

    print_header()
    print(f"Processing batch of {len(surface_files)} surfaces from {dir_path}...")
    if getattr(args, "remote", False):
        from surface2anatomy.model import RemoteSurfaceAnatomyModel
        model = RemoteSurfaceAnatomyModel(endpoint=getattr(args, "endpoint", "https://sharonmelhi-surface2anatomy-backend.hf.space"))
    else:
        model = SurfaceAnatomyModel.from_pretrained(
            model_variant=args.variant,
            device=args.device,
            cache_dir=args.cache_dir,
            remote_endpoint=getattr(args, "endpoint", "https://sharonmelhi-surface2anatomy-backend.hf.space")
        )

    targets = args.targets if args.targets else [args.target]
    all_dfs = []
    for f in surface_files:
        try:
            res = model.predict_multiple(surface_input=f, targets=targets, units=args.units)
            df = res.to_dataframe()
            df["filename"] = f.name
            all_dfs.append(df)
            print(f"  ✓ Processed: {f.name}")
        except Exception as e:
            print(f"  ✗ Failed: {f.name} ({e})")

    if all_dfs:
        import pandas as pd
        combined_df = pd.concat(all_dfs, ignore_index=True)
        out_path = Path(args.output) if args.output else dir_path / "batch_results.csv"
        combined_df.to_csv(out_path, index=False)
        print(f"\nBatch processing complete! Saved results to {out_path}")

def main():
    parser = argparse.ArgumentParser(
        prog="surface2anatomy",
        description="Surface2Anatomy: 3D Internal Anatomy Localization from External Body Surface Geometry"
    )
    parser.add_argument("-v", "--version", action="version", version=f"Surface2Anatomy {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # info
    p_info = subparsers.add_parser("info", help="Display model architecture, coordinate systems, and cache info")
    p_info.set_defaults(func=cmd_info)

    # list-targets
    p_targets = subparsers.add_parser("list-targets", help="List all supported internal anatomical landmarks")
    p_targets.add_argument("--category", type=str, default=None, help="Filter by category")
    p_targets.add_argument("--benchmark-only", action="store_true", help="Show only primary 104 benchmark targets")
    p_targets.set_defaults(func=cmd_list_targets)

    # validate-surface
    p_val = subparsers.add_parser("validate-surface", help="Validate 3D surface file format, vertex count, and scale")
    p_val.add_argument("surface_file", type=str, help="Path to 3D surface geometry file")
    p_val.set_defaults(func=cmd_validate_surface)

    # download-models
    p_dl = subparsers.add_parser("download-models", help="Pre-download and verify pretrained model weights")
    p_dl.add_argument("--variant", type=str, default="phase16_brain", choices=["phase10r", "phase16_brain"])
    p_dl.add_argument("--cache-dir", type=str, default=None, help="Custom cache directory")
    p_dl.set_defaults(func=cmd_download_models)

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict 3D centroid coordinates of internal organs")
    p_pred.add_argument("surface_file", type=str, help="Path to 3D surface geometry (.PLY, .PCD, .OBJ, .STL, etc.)")
    p_pred.add_argument("--target", type=str, default="spleen", help="Single target query")
    p_pred.add_argument("--targets", nargs="+", help="Multiple target queries")
    p_pred.add_argument("--units", type=str, default="auto", choices=["auto", "mm", "cm", "m"], help="Coordinate physical units")
    p_pred.add_argument("--variant", type=str, default="phase16_brain", choices=["phase10r", "phase16_brain"])
    p_pred.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p_pred.add_argument("--cache-dir", type=str, default=None)
    p_pred.add_argument("--remote", action="store_true", help="Route inference directly to live Hugging Face Space API without downloading local weights")
    p_pred.add_argument("--endpoint", type=str, default="https://sharonmelhi-surface2anatomy-backend.hf.space", help="Live API backend URL")
    p_pred.add_argument("--output", "-o", type=str, default=None, help="Output file (.json or .csv)")
    p_pred.set_defaults(func=cmd_predict)

    # batch
    p_batch = subparsers.add_parser("batch", help="Batch process a cohort of 3D patient surfaces")
    p_batch.add_argument("surfaces_dir", type=str, help="Folder containing 3D surface files")
    p_batch.add_argument("--target", type=str, default="liver", help="Single target")
    p_batch.add_argument("--targets", nargs="+", help="Multiple targets")
    p_batch.add_argument("--units", type=str, default="auto")
    p_batch.add_argument("--variant", type=str, default="phase16_brain")
    p_batch.add_argument("--device", type=str, default="auto")
    p_batch.add_argument("--cache-dir", type=str, default=None)
    p_batch.add_argument("--remote", action="store_true", help="Route batch inference to live Hugging Face Space API")
    p_batch.add_argument("--endpoint", type=str, default="https://sharonmelhi-surface2anatomy-backend.hf.space", help="Live API backend URL")
    p_batch.add_argument("--output", "-o", type=str, default="batch_results.csv")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
