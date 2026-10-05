#!/bin/bash
set -e

DL_DIR="/home/sharon/Datasets/CT_ORG/downloads"

# 1. Complete volumes 50-99.zip
FILE50="$DL_DIR/volumes 50-99.zip"
URL50="https://app.box.com/index.php?rm=box_download_shared_file&shared_name=tfzjfgi3wtht256wr89fpmjnmyssnfi4&file_id=f_566359888123"
EXP50=5840157598

echo "=== Downloading remaining bytes for volumes 50-99.zip ==="
while [ ! -f "$FILE50" ] || [ $(stat -c%s "$FILE50") -lt $EXP50 ]; do
    echo "Current size: $(stat -c%s "$FILE50" 2>/dev/null || echo 0) / $EXP50 bytes. Resuming..."
    curl -L -C - --speed-limit 50000 --speed-time 15 --connect-timeout 20 \
        -o "$FILE50" "$URL50" || true
    sleep 2
done
echo "volumes 50-99.zip completed: $(stat -c%s "$FILE50") bytes"

# 2. Complete volumes 100-139.zip
FILE100="$DL_DIR/volumes 100-139.zip"
URL100="https://app.box.com/index.php?rm=box_download_shared_file&shared_name=lb5f9putwakzwct5wsghyp2cp6xejjq2&file_id=f_566362486349"
EXP100=7404374918

echo "=== Downloading remaining bytes for volumes 100-139.zip ==="
while [ ! -f "$FILE100" ] || [ $(stat -c%s "$FILE100") -lt $EXP100 ]; do
    echo "Current size: $(stat -c%s "$FILE100" 2>/dev/null || echo 0) / $EXP100 bytes. Resuming..."
    curl -L -C - --speed-limit 50000 --speed-time 15 --connect-timeout 20 \
        -o "$FILE100" "$URL100" || true
    sleep 2
done
echo "volumes 100-139.zip completed: $(stat -c%s "$FILE100") bytes"

# 3. Verify all 4 archives and extract
echo "=== Running extraction and verification ==="
/home/sharon/env_py311/bin/python tools/phase14_ctorg/00_download_and_extract.py

# 4. Run entire evaluation suite
echo "=== Running all evaluation stages ==="
/home/sharon/env_py311/bin/python tools/phase14_ctorg/run_all_ctorg.py

echo "=== ALL PHASES COMPLETED SUCCESSFULLY ==="
