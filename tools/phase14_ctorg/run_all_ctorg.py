#!/usr/bin/env python3
"""
tools/phase14_ctorg/run_all_ctorg.py
===================================
Master runner for Phase 14 CT-ORG evaluation suite.
Executes stages in strict sequential dependency order.
"""

import sys
import subprocess
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent

stages = [
    ("Orientation & Left-Right Audit", "tools/phase14_ctorg/01_orientation_qc.py"),
    ("Duplicate & Overlap Audit", "tools/phase14_ctorg/02_duplicate_audit.py"),
    ("Eligibility & Geometry Preprocessing", "tools/phase14_ctorg/03_eligibility_and_preprocessing.py"),
    ("Inference, Baselines & Scientific Metrics", "tools/phase14_ctorg/04_run_inference_and_metrics.py"),
    ("Visualization & Figure Generation", "tools/phase14_ctorg/05_generate_figures.py")
]

for name, script in stages:
    print(f"\n{'='*80}\nRUNNING: {name}\n{'='*80}")
    cmd = [sys.executable, str(repo_root / script)]
    ret = subprocess.run(cmd, cwd=str(repo_root))
    if ret.returncode != 0:
        print(f"STAGE FAILED: {name} (exit code {ret.returncode})")
        sys.exit(ret.returncode)

print(f"\n{'='*80}\nALL CT-ORG PIPELINE STAGES COMPLETED SUCCESSFULLY\n{'='*80}")
