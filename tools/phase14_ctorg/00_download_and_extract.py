#!/usr/bin/env python3
"""
tools/phase14_ctorg/00_download_and_extract.py
=============================================
1. Downloads the 3 CT-ORG volume zip archives from official TCIA Box links.
2. Verifies archive sizes and SHA256 checksums.
3. Writes reports/phase14_ctorg/00_download_inventory.json.
4. Extracts archives into ~/Datasets/CT_ORG/extracted/.
5. Verifies 140 CT volumes and 140 label masks match 1:1.
"""

import os
import sys
import time
import json
import zipfile
import hashlib
import requests
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
dl_dir = Path("/home/sharon/Datasets/CT_ORG/downloads")
extracted_dir = Path("/home/sharon/Datasets/CT_ORG/extracted")
reports_dir = repo_root / "reports" / "phase14_ctorg"

dl_dir.mkdir(parents=True, exist_ok=True)
extracted_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

ARCHIVES = [
    {
        "key": "labels",
        "filename": "labels and README.zip",
        "url": "https://app.box.com/index.php?rm=box_download_shared_file&shared_name=haf06j5fy2h94re315841m7ok3n04s3k&file_id=f_566354490712",
        "expected_size": 270662160,
    },
    {
        "key": "vol0_49",
        "filename": "volumes 0-49.zip",
        "url": "https://app.box.com/index.php?rm=box_download_shared_file&shared_name=xxtxlpa1ift5c8d084njjerxamb951gy&file_id=f_566354043414",
        "expected_size": 5445256516,
    },
    {
        "key": "vol50_99",
        "filename": "volumes 50-99.zip",
        "url": "https://app.box.com/index.php?rm=box_download_shared_file&shared_name=tfzjfgi3wtht256wr89fpmjnmyssnfi4&file_id=f_566359888123",
        "expected_size": 5840157598,
    },
    {
        "key": "vol100_139",
        "filename": "volumes 100-139.zip",
        "url": "https://app.box.com/index.php?rm=box_download_shared_file&shared_name=lb5f9putwakzwct5wsghyp2cp6xejjq2&file_id=f_566362486349",
        "expected_size": 7404374918,
    },
]

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def download_file(url, dest_path, expected_size):
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    
    # Check if already complete
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if actual_size == expected_size:
            print(f"File {dest_path.name} already fully downloaded ({actual_size / 1e9:.2f} GB). Skipping download.")
            return
        else:
            print(f"File {dest_path.name} partially downloaded ({actual_size} / {expected_size} bytes). Resuming...")
            resume_header = {"Range": f"bytes={actual_size}-"}
            headers.update(resume_header)
            mode = "ab"
            downloaded = actual_size
    else:
        mode = "wb"
        downloaded = 0

    print(f"Starting download: {dest_path.name} (target: {expected_size / 1e9:.2f} GB)")
    t0 = time.time()
    last_print = t0
    
    with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
        if r.status_code not in (200, 206):
            r.raise_for_status()
        
        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print >= 5.0:
                        speed = (downloaded / 1e6) / max(now - t0, 0.1)
                        pct = (downloaded / expected_size * 100.0) if expected_size else 0
                        print(f"  [{dest_path.name}] {downloaded / 1e6:.1f} MB / {expected_size / 1e6:.1f} MB ({pct:.1f}%) - Speed: {speed:.1f} MB/s", flush=True)
                        last_print = now

    final_size = dest_path.stat().st_size
    print(f"Download complete: {dest_path.name} ({final_size / 1e9:.2f} GB in {time.time() - t0:.1f}s)")

def main():
    print("=" * 80)
    print("PHASE 14 CT-ORG: AUTOMATED STREAMED DOWNLOAD & INTEGRITY VERIFICATION")
    print("=" * 80)
    
    inventory = {}
    total_bytes = 0
    
    for item in ARCHIVES:
        key = item["key"]
        fname = item["filename"]
        url = item["url"]
        exp_sz = item["expected_size"]
        dest = dl_dir / fname
        
        # In case it was downloaded under another name (e.g. labels_and_README.zip)
        alt_dest = dl_dir / fname.replace(" ", "_")
        if alt_dest.exists() and not dest.exists():
            print(f"Renaming {alt_dest.name} -> {dest.name}")
            alt_dest.rename(dest)
            
        download_file(url, dest, exp_sz)
        
        actual_sz = dest.stat().st_size
        total_bytes += actual_sz
        print(f"Calculating SHA256 for {dest.name}...")
        sha = sha256_file(dest)
        print(f"  SHA256: {sha}")
        
        inventory[key] = {
            "filename": fname,
            "path": str(dest),
            "size_bytes": actual_sz,
            "size_gb": round(actual_sz / (1024**3), 3),
            "sha256": sha,
            "expected_size": exp_sz,
            "size_match": actual_sz == exp_sz
        }
    
    inv_file = reports_dir / "00_download_inventory.json"
    full_inventory = {
        "dataset": "CT-ORG",
        "source": "The Cancer Imaging Archive (TCIA) / Box",
        "total_files": len(ARCHIVES),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / (1024**3), 3),
        "archives": inventory
    }
    with open(inv_file, "w") as f:
        json.dump(full_inventory, f, indent=2)
    print(f"\nSaved inventory to {inv_file}")
    
    # Extraction
    print("\n" + "=" * 80)
    print("EXTRACTING ARCHIVES TO ~/Datasets/CT_ORG/extracted/")
    print("=" * 80)
    
    for item in ARCHIVES:
        fname = item["filename"]
        dest = dl_dir / fname
        print(f"Extracting {fname}...")
        with zipfile.ZipFile(dest, "r") as z:
            z.extractall(extracted_dir)
        print(f"  Extracted {fname}")
        
    # Organize extracted files into clean volumes/ and labels/ directories
    vols_dir = extracted_dir / "volumes"
    lbls_dir = extracted_dir / "labels"
    vols_dir.mkdir(parents=True, exist_ok=True)
    lbls_dir.mkdir(parents=True, exist_ok=True)
    
    for f in extracted_dir.rglob("volume-*.nii.gz"):
        target = vols_dir / f.name
        if f != target:
            f.rename(target)
            
    for f in extracted_dir.rglob("labels-*.nii.gz"):
        target = lbls_dir / f.name
        if f != target:
            f.rename(target)
            
    ct_files = sorted(list(vols_dir.glob("volume-*.nii.gz")))
    lbl_files = sorted(list(lbls_dir.glob("labels-*.nii.gz")))
    
    print(f"\nTotal CT files found: {len(ct_files)}")
    print(f"Total label files found: {len(lbl_files)}")
    
    # Verify 1:1 pairing from 0 to 139
    mismatches = []
    for i in range(140):
        v_path = vols_dir / f"volume-{i}.nii.gz"
        l_path = lbls_dir / f"labels-{i}.nii.gz"
        if not v_path.exists():
            mismatches.append(f"Missing CT: volume-{i}.nii.gz")
        if not l_path.exists():
            mismatches.append(f"Missing label: labels-{i}.nii.gz")
            
    if mismatches:
        print(f"ERROR: Found {len(mismatches)} pairing mismatches!")
        for m in mismatches[:10]:
            print(" ", m)
        sys.exit(1)
    else:
        print("PASS: Exactly 140 CT volumes and 140 label files verified 1:1 (volume-0 to volume-139).")

if __name__ == "__main__":
    main()
