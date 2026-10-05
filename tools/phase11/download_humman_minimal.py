#!/usr/bin/env python3
"""
tools/phase11/download_humman_minimal.py

Downloads the minimal real-sensor suite from Hugging Face repo 'caizhongang/HuMMan':
1. point_cameras.7z (~58 KB)
2. point_iphone_depth.7z (~3.2 GB)
3. smpl_params.7z (~22 MB)

Saves download manifest and optional RGB decision.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "caizhongang/HuMMan"
REPO_TYPE = "dataset"
RAW_DIR = "data_external/HuMMan/raw"
MANIFEST_PATH = "reports/phase11/downloads/HuMMan_download_manifest.json"
RGB_DECISION_PATH = "reports/phase11/downloads/HuMMan_optional_RGB_decision.md"

TARGET_FILES = [
    "humman_release_v1.0_point/point_cameras.7z",
    "humman_release_v1.0_point/smpl_params.7z",
    "humman_release_v1.0_point/point_iphone_depth.7z"
]

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1048576): # 1MB chunks
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Phase 11B: HuMMan Minimal Sensor Suite Download ===")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

    api = HfApi()
    print(f"Connecting to Hugging Face repo: {REPO_ID} (type={REPO_TYPE})...")
    ds_info = api.dataset_info(repo_id=REPO_ID, files_metadata=True)
    repo_commit = ds_info.sha
    print(f"Repo commit SHA: {repo_commit}")

    # Build target metadata map
    sibling_map = {}
    for s in ds_info.siblings:
        if s.rfilename in TARGET_FILES:
            sibling_map[s.rfilename] = s

    downloaded_records = []

    for rfilename in TARGET_FILES:
        if rfilename not in sibling_map:
            raise FileNotFoundError(f"Required HuMMan file '{rfilename}' not found in repo!")
        
        meta = sibling_map[rfilename]
        expected_size = meta.size
        expected_sha = meta.lfs.sha256 if meta.lfs else None
        local_filename = os.path.basename(rfilename)
        dest_path = os.path.join(RAW_DIR, local_filename)

        print(f"\nTarget: {rfilename}")
        print(f"Expected Size: {expected_size} bytes ({expected_size/1024/1024:.2f} MB)")
        print(f"Expected LFS SHA256: {expected_sha}")

        # Check if already present and matches size/sha
        already_valid = False
        if os.path.exists(dest_path) and os.path.getsize(dest_path) == expected_size:
            print(f"File exists locally with matching size. Verifying SHA-256...")
            actual_sha = sha256_file(dest_path)
            if expected_sha is None or actual_sha == expected_sha:
                print(f"SHA-256 verified: {actual_sha}")
                already_valid = True
                downloaded_records.append({
                    "rfilename": rfilename,
                    "local_path": dest_path,
                    "size_bytes": expected_size,
                    "sha256": actual_sha,
                    "expected_sha256": expected_sha,
                    "status": "VERIFIED_EXISTING"
                })

        if not already_valid:
            print(f"Downloading {rfilename} via hf_hub_download...")
            downloaded_file = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=rfilename,
                local_dir=RAW_DIR,
                local_dir_use_symlinks=False
            )
            # If nested in subdir, ensure it's accessible or moved to RAW_DIR flat or kept
            print(f"Downloaded to: {downloaded_file}")
            actual_size = os.path.getsize(downloaded_file)
            print(f"Computing SHA-256 for {downloaded_file}...")
            actual_sha = sha256_file(downloaded_file)
            print(f"Actual SHA-256: {actual_sha}")
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(f"Checksum mismatch for {rfilename}! Expected {expected_sha}, got {actual_sha}")

            downloaded_records.append({
                "rfilename": rfilename,
                "local_path": downloaded_file,
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "expected_sha256": expected_sha,
                "status": "DOWNLOADED_AND_VERIFIED"
            })

    # Save manifest
    manifest = {
        "dataset_name": "HuMMan",
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "repo_commit": repo_commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "license": "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
        "files": downloaded_records
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote HuMMan manifest to {MANIFEST_PATH}")

    # Write optional RGB decision
    with open(RGB_DECISION_PATH, "w") as f:
        f.write("# HuMMan Optional RGB Download Decision\n\n")
        f.write(f"- **Decision Date:** {datetime.now(timezone.utc).isoformat()}\n")
        f.write("- **Status:** NOT DOWNLOADED (BYPASS CONFIRMED)\n")
        f.write("- **Rationale:** The Phase-10R model architecture (`MultiScaleSurfacePointNet2Encoder` + `TargetQueryTransformerDecoder`) strictly ingests 3D coordinates $(N, 3)$, without requiring RGB color channels or photometric textures. The minimal sensor suite downloaded contains real iPhone depth maps, intrinsic/extrinsic calibration parameters, and SMPL body references, which fully satisfy all geometric surface reconstruction, temporal jitter, cross-view consistency, and domain gap evaluation criteria without the unnecessary download overhead of multi-gigabyte color archives.\n")
    print(f"Wrote RGB decision to {RGB_DECISION_PATH}")
    print("=== HuMMan Minimal Download Complete ===")

if __name__ == "__main__":
    main()
