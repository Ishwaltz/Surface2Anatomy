#!/bin/bash
# sharon/run_benchmark.sh

set -e

# Navigate to sharon folder
cd "$(dirname "$0")"

BACKBONES=("densenet121" "resnet50" "efficientnet_b0" "swin_unetr")
EPOCHS=150

echo "=== Starting 3D Organ Backbone Benchmarking Suite in sharon/ ==="

for BB in "${BACKBONES[@]}"; do
    echo "=================================================="
    echo " Training Backbone: $BB"
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
        --grad_accum_steps $GRAD_ACCUM
done

echo "=================================================="
echo " All Backbones Trained. Evaluating Real Metrics..."
echo "=================================================="

python evaluate_benchmark.py

echo "Benchmarking complete. Real evaluation report generated at sharon/outputs/BENCHMARK_REPORT.md."
