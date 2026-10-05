#!/usr/bin/env python3
"""
tools/upload_sex_aware_to_hf.py
===============================
Uploads newly trained Sex-Aware 121-organ ensemble weights and PointNet sex classifier
to the official Hugging Face repository: SharonMelhi/S2A-Net-Weights
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "SharonMelhi/S2A-Net-Weights"

repo_root = Path(__file__).resolve().parent.parent
ckpt_dir = repo_root / "experiments" / "sex_aware" / "checkpoints"

def main():
    print(f"Connecting to Hugging Face Hub ({REPO_ID})...")
    api = HfApi(token=HF_TOKEN)

    files_to_upload = [
        "S2A_SexAware_seed42.pt",
        "S2A_SexAware_seed43.pt",
        "S2A_SexAware_seed44.pt",
        "S2A_SexClassifier.pt"
    ]

    for fname in files_to_upload:
        fpath = ckpt_dir / fname
        if not fpath.exists():
            print(f"Skipping {fname} (not found in {ckpt_dir})")
            continue

        print(f"Uploading {fname} ({fpath.stat().st_size / (1024*1024):.2f} MB) to {REPO_ID}...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=f"sex_aware/{fname}",
            repo_id=REPO_ID,
            repo_type="model"
        )
        print(f"✓ Successfully uploaded sex_aware/{fname} to Hugging Face!")

    print("\nAll Sex-Aware weights successfully synchronized to Hugging Face!")

if __name__ == "__main__":
    main()
