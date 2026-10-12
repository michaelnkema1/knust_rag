#!/usr/bin/env bash
# One-time environment setup: CPU-only PyTorch (small download), then the rest.
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
echo "SETUP_EXIT=$?"
