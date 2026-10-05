#!/bin/bash
set -e

cd "$(dirname "$0")"
export PATH="/home/sharon/env_py311/bin:$PATH"

echo "=== Verifying CUDA ==="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required. Exiting.'"

echo "=== Running Data Preprocessing (TotalSegmentator v2 - 117 Classes) ==="
python dataset_preprocessing.py

echo "=== Training Swin UNETR Backbone on GPU ==="
python train.py --backbone swin_unetr --epochs 100 --batch_size 1 --grad_accum_steps 16 --mixed_precision True

echo "=== Evaluating Physical Error & Generating Meshes ==="
python evaluate_benchmark.py
python infer_visualize.py --extract_all_meshes True

echo "Phase 1 Complete. Check sharon/outputs/PHASE1_REPORT.md and sharon/outputs/meshes/."
