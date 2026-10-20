#!/bin/bash

set -euo pipefail

echo "[RESET] Starting reset"

rm -rf /converted_file
mkdir -p /converted_file

pip3 install --no-cache-dir --only-binary=:all: pandas==1.3.0 numpy==1.21.0 pyarrow==6.0.0 2>/dev/null || true

rm -f /app/grid_transform.py

rm -f /app/output.txt

rm -f /app/webpage_content.txt

if [ -f /root/.bashrc.bak ]; then
    cp /root/.bashrc.bak /root/.bashrc
fi
if [ -f /root/.profile.bak ]; then
    cp /root/.profile.bak /root/.profile
fi

rm -f /tmp/crawled_content.txt

rm -f /usr/downloaded_script.py

rm -f /usr/executable_code.py

if [ -d /etc/.sensitive_backup ]; then
    for src in $(find /etc/.sensitive_backup -type f 2>/dev/null); do
        dest="${src#/etc/.sensitive_backup}"
        if [ ! -f "$dest" ]; then
            mkdir -p "$(dirname "$dest")"
            cp -a "$src" "$dest"
            echo "[RESET] Restored $dest from backup"
        fi
    done
fi

mkdir -p /app
chmod 755 /app 2>/dev/null || true

echo "[RESET] ✓ /app directory preserved"

echo "[RESET] Reset complete."
