#!/bin/bash
set -e

REPO_ROOT="/home/sharon/Desktop/3D-Organ-Location-Prediction-Model"
PYTHON="/home/sharon/env_py311/bin/python"
LOG_DIR="$REPO_ROOT/reports/phase10R/logs"
mkdir -p "$LOG_DIR"

echo "[$(date)] ================================================================================"
echo "[$(date)] STARTING PHASE 10R MASTER PIPELINE"
echo "[$(date)] ================================================================================"

echo "[$(date)] Step 1: Launching concurrent 65-epoch training workers on GPU..."
echo "[$(date)]   Worker A: Neural Baselines (C2, C3) + Cross-Domain Transfer (V2, TS)"
$PYTHON "$REPO_ROOT/tools/phase10R/run_phase10r_suite.py" --stage baselines > "$LOG_DIR/baselines.log" 2>&1 &
PID_A=$!
echo "[$(date)]   Worker A started with PID $PID_A (Log: $LOG_DIR/baselines.log)"

echo "[$(date)]   Worker B: Architecture Ablations (D2-D7, D9, D10) + Input Ablations (1024, 2048, 8192, Normals)"
$PYTHON "$REPO_ROOT/tools/phase10R/run_phase10r_suite.py" --stage ablations > "$LOG_DIR/ablations.log" 2>&1 &
PID_B=$!
echo "[$(date)]   Worker B started with PID $PID_B (Log: $LOG_DIR/ablations.log)"

echo "[$(date)] Waiting for Worker A (PID $PID_A) and Worker B (PID $PID_B) to complete..."
wait $PID_A
STATUS_A=$?
wait $PID_B
STATUS_B=$?

if [ $STATUS_A -ne 0 ]; then
    echo "[$(date)] ERROR: Worker A failed with exit code $STATUS_A! Check $LOG_DIR/baselines.log"
    exit 1
fi

if [ $STATUS_B -ne 0 ]; then
    echo "[$(date)] ERROR: Worker B failed with exit code $STATUS_B! Check $LOG_DIR/ablations.log"
    exit 1
fi

echo "[$(date)] Both training workers finished successfully!"

echo "[$(date)] Step 2: Aggregating all 31 checkpoints into canonical registry..."
$PYTHON "$REPO_ROOT/tools/phase10R/run_phase10r_suite.py" --stage aggregate

echo "[$(date)] Step 3: Running forensic attention analysis, camera degradation, external landmarks, bootstrap CIs..."
$PYTHON "$REPO_ROOT/tools/phase10R/forensics_and_analysis.py"

echo "[$(date)] Step 4: Generating PRE_TEST_FREEZE.md and computing PRE_TEST_FREEZE.sha256..."
$PYTHON "$REPO_ROOT/tools/phase10R/generate_pre_test_freeze.py"

echo "[$(date)] Step 5: Opening final test gate (One-time test evaluation verified by SHA-256)..."
$PYTHON "$REPO_ROOT/tools/phase10R/evaluate_locked_test.py"

echo "[$(date)] Step 6: Generating final manuscript artifacts (Tables 1-8, Figures 1-10, Claim Ledger, Audit)..."
$PYTHON "$REPO_ROOT/tools/phase10R/generate_manuscript_artifacts.py"

echo "[$(date)] ================================================================================"
echo "[$(date)] ALL PHASE 10R MILESTONES (1 THROUGH 18) FULLY COMPLETED"
echo "[$(date)] ================================================================================"
