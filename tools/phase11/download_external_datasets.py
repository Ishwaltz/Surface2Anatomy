#!/usr/bin/env python3
"""
tools/phase11/download_external_datasets.py

Downloads AMOS22 official dataset from Zenodo (DOI 10.5281/zenodo.7262581):
- amos22.zip (~24.2 GB), official expected MD5: 67717b2a483ac0744c89c3016b7aaef7
- labeled_data_meta_0000_0599.csv (~35.2 KB)

Features:
- Streamed chunked downloads with progress
- HTTP Range resume support
- Checksum verification (MD5 & SHA256)
- Manifest generation in reports/phase11/downloads/AMOS22_download_manifest.json
"""

import os
import sys
import json
import time
import hashlib
import requests
from datetime import datetime, timezone

ZENODO_API_URL = "https://zenodo.org/api/records/7262581"
DEST_DIR = "data_external/AMOS22/raw"
MANIFEST_PATH = "reports/phase11/downloads/AMOS22_download_manifest.json"
EXPECTED_MD5_AMOS22 = "67717b2a483ac0744c89c3016b7aaef7"

def compute_checksums(filepath):
    print(f"Computing MD5 and SHA-256 for {filepath}...")
    t0 = time.time()
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    total_bytes = 0
    with open(filepath, 'rb') as f:
        while chunk := f.read(4 * 1024 * 1024): # 4MB chunks
            md5.update(chunk)
            sha256.update(chunk)
            total_bytes += len(chunk)
    elapsed = time.time() - t0
    rate = (total_bytes / 1024 / 1024) / max(elapsed, 0.001)
    print(f"Checksums computed in {elapsed:.2f}s ({rate:.1f} MB/s)")
    return md5.hexdigest(), sha256.hexdigest()

def download_file_with_resume(url, dest_path, expected_size, expected_md5=None):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    # Check if final file already exists and is complete
    if os.path.exists(dest_path):
        actual_size = os.path.getsize(dest_path)
        if actual_size == expected_size:
            print(f"File {dest_path} already exists with exact size ({actual_size} bytes). Verifying checksum...")
            md5_val, sha256_val = compute_checksums(dest_path)
            if expected_md5 and md5_val != expected_md5:
                print(f"WARNING: MD5 mismatch for existing file ({md5_val} != {expected_md5}). Re-downloading...")
            else:
                print(f"File {dest_path} is verified! MD5: {md5_val}, SHA256: {sha256_val}")
                return {
                    "local_path": dest_path,
                    "size_bytes": actual_size,
                    "md5": md5_val,
                    "sha256": sha256_val,
                    "status": "VERIFIED_EXISTING"
                }

    existing_size = 0
    if os.path.exists(temp_path):
        existing_size = os.path.getsize(temp_path)
        print(f"Found partial download: {existing_size} / {expected_size} bytes ({existing_size/expected_size*100:.2f}%)")

    headers = {}
    mode = "wb"
    if existing_size > 0:
        if existing_size > expected_size:
            print("Partial file larger than expected size. Truncating...")
            existing_size = 0
            mode = "wb"
        else:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
            print(f"Resuming download from byte {existing_size}...")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=5)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    t0 = time.time()
    last_log_time = t0
    bytes_downloaded_this_session = 0

    with session.get(url, headers=headers, stream=True, timeout=30) as r:
        if headers and r.status_code == 416: # Range not satisfiable
            print("Server returned 416 (Range not satisfiable). Restarting download...")
            existing_size = 0
            mode = "wb"
            r = session.get(url, stream=True, timeout=30)
            r.raise_for_status()
        elif headers and r.status_code != 206 and r.status_code != 200:
            r.raise_for_status()
        elif not headers and r.status_code != 200:
            r.raise_for_status()

        total_target = expected_size
        current_bytes = existing_size

        with open(temp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024): # 4MB
                if chunk:
                    f.write(chunk)
                    current_bytes += len(chunk)
                    bytes_downloaded_this_session += len(chunk)
                    now = time.time()
                    if now - last_log_time >= 5.0 or current_bytes == total_target:
                        elapsed_total = now - t0
                        speed_mb = (bytes_downloaded_this_session / 1024 / 1024) / max(elapsed_total, 0.001)
                        pct = (current_bytes / total_target) * 100.0 if total_target else 0
                        remaining_bytes = max(0, total_target - current_bytes)
                        eta_s = remaining_bytes / max(speed_mb * 1024 * 1024, 1)
                        print(f"Progress: {current_bytes}/{total_target} bytes ({pct:.1f}%) | Speed: {speed_mb:.2f} MB/s | ETA: {eta_s/60:.1f} min")
                        last_log_time = now

    # Move temp to final
    os.rename(temp_path, dest_path)
    print(f"Download complete: {dest_path}")
    md5_val, sha256_val = compute_checksums(dest_path)

    if expected_md5 and md5_val != expected_md5:
        raise ValueError(f"FATAL: Downloaded MD5 {md5_val} does not match expected {expected_md5}!")

    print(f"Verified MD5: {md5_val}")
    print(f"Verified SHA256: {sha256_val}")

    return {
        "local_path": dest_path,
        "size_bytes": os.path.getsize(dest_path),
        "md5": md5_val,
        "sha256": sha256_val,
        "status": "DOWNLOADED_AND_VERIFIED"
    }

def main():
    print("=== Phase 11A: AMOS22 Official Dataset Downloader ===")
    os.makedirs(DEST_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

    print(f"Querying Zenodo API: {ZENODO_API_URL}...")
    r = requests.get(ZENODO_API_URL, timeout=15)
    r.raise_for_status()
    record = r.json()

    metadata = record.get("metadata", {})
    files_list = record.get("files", [])
    print(f"Record Title: {metadata.get('title')}")
    print(f"Record DOI: {metadata.get('doi')}")
    print(f"Found {len(files_list)} files in record:")

    file_meta_map = {}
    for f in files_list:
        fname = f.get("key")
        fsize = f.get("size")
        fchecksum = f.get("checksum")
        flink = f.get("links", {}).get("self")
        print(f"  - {fname} ({fsize} bytes, {fchecksum})")
        file_meta_map[fname] = {
            "size": fsize,
            "checksum": fchecksum,
            "url": flink
        }

    download_results = []

    # 1. Download metadata CSV if present
    meta_csv_name = "labeled_data_meta_0000_0599.csv"
    if meta_csv_name in file_meta_map:
        csv_info = file_meta_map[meta_csv_name]
        csv_dest = os.path.join(DEST_DIR, meta_csv_name)
        exp_md5 = csv_info["checksum"].replace("md5:", "") if "md5:" in csv_info["checksum"] else None
        print(f"\nDownloading metadata CSV: {meta_csv_name}...")
        res = download_file_with_resume(csv_info["url"], csv_dest, csv_info["size"], exp_md5)
        res["filename"] = meta_csv_name
        download_results.append(res)

    # 2. Download amos22.zip
    zip_name = "amos22.zip"
    if zip_name in file_meta_map:
        zip_info = file_meta_map[zip_name]
        zip_dest = os.path.join(DEST_DIR, zip_name)
        print(f"\nDownloading main archive: {zip_name} (~24.2 GB)...")
        res = download_file_with_resume(zip_info["url"], zip_dest, zip_info["size"], EXPECTED_MD5_AMOS22)
        res["filename"] = zip_name
        download_results.append(res)
    else:
        raise FileNotFoundError(f"amos22.zip not found in Zenodo record {ZENODO_API_URL}!")

    # Write manifest
    manifest = {
        "dataset_name": "AMOS22",
        "doi": metadata.get("doi", "10.5281/zenodo.7262581"),
        "zenodo_record_id": 7262581,
        "title": metadata.get("title"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "license": metadata.get("license", {}).get("id", "cc-by-4.0"),
        "files": download_results
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote AMOS22 download manifest to {MANIFEST_PATH}")
    print("=== AMOS22 Download Complete & Verified ===")

if __name__ == "__main__":
    main()
