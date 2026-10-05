#!/bin/bash
set -e

REPO_ROOT="/home/sharon/Desktop/3D-Organ-Location-Prediction-Model"
PYTHON="/home/sharon/env_py311/bin/python"
PID=1674712

echo "[$(date)] Post-training chain started. Waiting for training PID $PID to finish..."

while kill -0 $PID 2>/dev/null; do
    sleep 15
done

echo "[$(date)] Training PID $PID completed! Launching post-training forensic, test, and synthesis pipeline..."

echo "[$(date)] Step 1: Running Forensics and Scientific Audit..."
$PYTHON "$REPO_ROOT/tools/phase10R/forensics_and_analysis.py"

echo "[$(date)] Step 2: Running Locked Test Evaluation..."
$PYTHON "$REPO_ROOT/tools/phase10R/evaluate_locked_test.py"

echo "[$(date)] Step 3: Generating Manuscript Artifacts (Tables, Figures, Audit, Claim Ledger)..."
$PYTHON "$REPO_ROOT/tools/phase10R/generate_manuscript_artifacts.py"

echo "[$(date)] ================================================================================"
echo "[$(date)] ALL PHASE 10R MILESTONES (1 THROUGH 18) FULLY COMPLETED"
echo "[$(date)] ================================================================================"
