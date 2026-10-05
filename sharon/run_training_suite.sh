#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/home/sharon/env_py311/bin/python"
LOG_DIR="$SCRIPT_DIR/outputs/logs"
CKPT_DIR="$SCRIPT_DIR/outputs/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "=========================================================="
echo " Starting Full 150-Epoch 3D Organ Location Training Suite "
echo " Date: $(date)"
echo " CUDA Device: $($PYTHON -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')")"
echo "=========================================================="

BACKBONES=("swin_unetr" "densenet121" "resnet50" "efficientnet_b0")
EPOCHS=150

for BB in "${BACKBONES[@]}"; do
    echo ""
    echo "=========================================================="
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] Starting Backbone: $BB ($EPOCHS Epochs)"
    echo "=========================================================="
    
    mkdir -p "$CKPT_DIR/$BB" "$LOG_DIR/$BB"
    
    if [ "$BB" == "swin_unetr" ]; then
        BATCH_SIZE=1
        GRAD_ACCUM=16
    elif [ "$BB" == "resnet50" ]; then
        BATCH_SIZE=2
        GRAD_ACCUM=8
    else
        BATCH_SIZE=2
        GRAD_ACCUM=8
    fi

    $PYTHON -u train.py \
        --data_root "$SCRIPT_DIR/dataset" \
        --output_dir "$SCRIPT_DIR/outputs" \
        --backbone "$BB" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --grad_accum_steps $GRAD_ACCUM \
        --patience $EPOCHS \
        --mixed_precision True 2>&1 | tee "$LOG_DIR/${BB}_train.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished training for backbone: $BB"
done

echo ""
echo "=========================================================="
echo " All 4 Backbones Trained for $EPOCHS Epochs. Running Benchmark Evaluation..."
echo "=========================================================="

$PYTHON evaluate_benchmark.py --backbone all

echo ""
echo "=========================================================="
echo " Running Spatial Verification Audit..."
echo "=========================================================="

$PYTHON audit_model.py

echo ""
echo "=========================================================="
echo " Training Suite Successfully Completed! "
echo " Date: $(date)"
echo " Reports: outputs/BENCHMARK_REPORT.md, outputs/PHASE1_REPORT.md"
echo " Audits:  outputs/audits/audit_organ_5.png"
echo "=========================================================="
