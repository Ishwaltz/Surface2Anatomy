#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/outputs/train.pid"

echo "=========================================================="
echo "          STOPPING 3D ORGAN BACKGROUND TRAINING           "
echo "=========================================================="

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[INFO] Terminating master process group (PID: $PID)..."
        kill -TERM -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
        sleep 2
        kill -9 -"$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

echo "[INFO] Cleaning up any remaining PyTorch training worker processes..."
pkill -9 -f "train.py" 2>/dev/null || true
pkill -9 -f "run_training_suite.sh" 2>/dev/null || true

sleep 1
echo "[SUCCESS] All training processes stopped. GPU VRAM released."
echo "=========================================================="
