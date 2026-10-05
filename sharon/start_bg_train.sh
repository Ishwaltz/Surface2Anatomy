#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p outputs/logs outputs/checkpoints

LOG_FILE="$SCRIPT_DIR/outputs/logs/train_bg.log"
PID_FILE="$SCRIPT_DIR/outputs/train.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[INFO] Training is already running in background with PID: $PID"
        echo "[INFO] Check status: bash status_bg_train.sh"
        echo "[INFO] Stream logs:   tail -f $LOG_FILE"
        exit 0
    fi
fi

echo "=========================================================="
echo " Launching Detached Background Training (Full 150 Epochs)"
echo " Workspace: $SCRIPT_DIR"
echo " Log file:  $LOG_FILE"
echo "=========================================================="

nohup setsid bash "$SCRIPT_DIR/run_training_suite.sh" > "$LOG_FILE" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

sleep 2

if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "=========================================================="
    echo " [SUCCESS] Background training is actively running!"
    echo " PID: $TRAIN_PID"
    echo " Even if you close this window/IDE, training will continue."
    echo " Monitor with: bash status_bg_train.sh"
    echo " Stop with:    bash stop_bg_train.sh"
    echo "=========================================================="
else
    echo "[ERROR] Background training failed to start. Check $LOG_FILE:"
    cat "$LOG_FILE"
    exit 1
fi
