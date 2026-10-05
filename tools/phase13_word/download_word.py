#!/usr/bin/env python3
"""
tools/phase13_word/download_word.py
Downloads official WORD-V0.1.0.zip from Google Drive directly.
"""

import os
import sys
import time
import hashlib
from pathlib import Path
import requests
from bs4 import BeautifulSoup

out_dir = Path.home() / "Datasets/WORD/downloads"
out_dir.mkdir(parents=True, exist_ok=True)
dest_path = out_dir / "WORD-V0.1.0.zip"

print(f"Destination: {dest_path}")
file_id = "19OWCXZGrimafREhXm8O8w2HBHZTfxEgU"
url = f"https://drive.google.com/uc?id={file_id}&export=download"

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=5)
session.mount("https://", adapter)

print("Fetching confirmation page...")
r = session.get(url, timeout=30)
soup = BeautifulSoup(r.text, "html.parser")
form = soup.find("form", {"id": "download-form"})
if not form:
    print("[ERROR] Confirmation form not found on Google Drive page:")
    print(r.text[:500])
    sys.exit(1)

action = form.get("action")
inputs = form.find_all("input")
params = {inp.get("name"): inp.get("value") for inp in inputs if inp.get("name")}
print(f"Action: {action}, Params: {params}")

# If file already exists and is complete (6809125051 bytes), verify SHA256
expected_size = 6809125051
if dest_path.exists() and dest_path.stat().st_size == expected_size:
    print(f"File already downloaded and matches expected size ({expected_size} bytes). Computing SHA256...")
    h = hashlib.sha256()
    with open(dest_path, "rb") as f:
        while chunk := f.read(1048576 * 8):
            h.update(chunk)
    sha256 = h.hexdigest()
    print(f"Verified existing file SHA256: {sha256}")
    sys.exit(0)

# Resume / fresh download
start_byte = 0
headers = {}
if dest_path.exists():
    start_byte = dest_path.stat().st_size
    if start_byte < expected_size:
        headers["Range"] = f"bytes={start_byte}-"
        print(f"Resuming download from byte {start_byte} / {expected_size} ({start_byte/expected_size*100:.1f}%)...")
    else:
        start_byte = 0

mode = "ab" if start_byte > 0 else "wb"

t0 = time.time()
with session.get(action, params=params, headers=headers, stream=True, timeout=60) as resp:
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0)) + start_byte
    print(f"Streaming {total_size / (1024**3):.2f} GB to {dest_path}...")
    
    downloaded = start_byte
    last_log = time.time()
    last_downloaded = downloaded
    
    with open(dest_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=1048576 * 4): # 4MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                now = time.time()
                if now - last_log >= 10.0 or downloaded == total_size:
                    elapsed = now - t0
                    speed = (downloaded - last_downloaded) / (now - last_log) / (1024 * 1024)
                    pct = (downloaded / total_size) * 100.0 if total_size > 0 else 0
                    print(f"  [{pct:5.1f}%] {downloaded / (1024**3):.2f} / {total_size / (1024**3):.2f} GB | Speed: {speed:5.1f} MB/s | Elapsed: {elapsed:.0f}s")
                    last_log = now
                    last_downloaded = downloaded

print(f"\nDownload complete! Total size: {dest_path.stat().st_size} bytes.")
print("Computing SHA256 checksum...")
h = hashlib.sha256()
with open(dest_path, "rb") as f:
    while chunk := f.read(1048576 * 8):
        h.update(chunk)
sha256 = h.hexdigest()
print(f"SHA256: {sha256}")

# Save checksum info
info_file = out_dir / "WORD-V0.1.0.zip.sha256"
with open(info_file, "w") as f:
    f.write(f"{sha256}  WORD-V0.1.0.zip\n")
print(f"Saved checksum to {info_file}")
