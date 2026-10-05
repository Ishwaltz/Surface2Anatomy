#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Verifying CUDA Environment ==="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting execution.'"

BACKBONES=("densenet121" "resnet50" "efficientnet_b0" "swin_unetr")
EPOCHS=150

for BB in "${BACKBONES[@]}"; do
    echo "=================================================="
    echo " Training Backbone: $BB on CUDA Device"
    echo "=================================================="
    
    if [ "$BB" == "swin_unetr" ]; then
        BATCH_SIZE=2
        GRAD_ACCUM=8
    else
        BATCH_SIZE=4
        GRAD_ACCUM=4
    fi

    python train.py \
        --data_root dataset \
        --backbone "$BB" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --grad_accum_steps $GRAD_ACCUM \
        --mixed_precision True
done

echo "=================================================="
echo " Executing GPU Validation & Benchmark Analysis..."
echo "=================================================="

python evaluate_benchmark.py

echo "Benchmarking process complete. Results stored in sharon/outputs/BENCHMARK_REPORT.md."