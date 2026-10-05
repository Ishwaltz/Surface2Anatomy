#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/outputs/train.pid"
LOG_FILE="$SCRIPT_DIR/outputs/logs/train_bg.log"

echo "=========================================================="
echo "          3D ORGAN MODEL BACKGROUND TRAINING STATUS       "
echo "=========================================================="

IS_RUNNING=0
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        IS_RUNNING=1
        echo "[STATUS] Master Runner is ACTIVE (PID: $PID)"
    fi
fi

if pgrep -f "train.py" > /dev/null 2>&1; then
    IS_RUNNING=1
    TRAIN_PIDS=$(pgrep -f "train.py" | tr '\n' ' ')
    echo "[STATUS] Active PyTorch Training Process (PIDs: $TRAIN_PIDS)"
fi

if [ $IS_RUNNING -eq 1 ]; then
    echo ""
    echo "[GPU Resource Usage]"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    echo ""
    echo "[Saved Checkpoints in outputs/checkpoints/]"
    find "$SCRIPT_DIR/outputs/checkpoints" -type f -name "*.pth" -exec ls -lh {} + 2>/dev/null | awk '{print $9, "(" $5 ")"}'
    echo ""
    echo "[Recent Output Log ($LOG_FILE)]"
    echo "----------------------------------------------------------"
    tail -n 25 "$LOG_FILE"
    echo "----------------------------------------------------------"
else
    echo "[STATUS] Training is currently NOT running."
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "[Last Log Entries ($LOG_FILE)]"
        tail -n 15 "$LOG_FILE"
    fi
fi
echo "=========================================================="
