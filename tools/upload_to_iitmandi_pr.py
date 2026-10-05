import os, sys
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", None)
api = HfApi(token=token)
repo_id = "IITMandiResearch/Surface2Anatomy-Weights"
ckpt_dir = Path("experiments/sex_aware/checkpoints")

files = [
    "S2A_SexAware_seed42.pt",
    "S2A_SexAware_seed43.pt",
    "S2A_SexAware_seed44.pt"
]

for f in files:
    p = ckpt_dir / f
    if not p.exists():
        print(f"Skipping {f}, not found")
        continue
    print(f"Uploading {f} ({p.stat().st_size / (1024*1024):.2f} MB) to {repo_id} (refs/pr/2)...")
    res = api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=f"sex_aware/{f}",
        repo_id=repo_id,
        revision="refs/pr/2",
        repo_type="model"
    )
    print(f"✓ Uploaded {f} to PR #2!")

print("\nAll Sex-Aware weights successfully added to PR #2 on IITMandiResearch/Surface2Anatomy-Weights!")
