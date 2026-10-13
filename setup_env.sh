#!/usr/bin/env bash
# One-time local setup: virtualenv + everything for running, ingesting, evaluating and testing.
# (No PyTorch needed: embeddings use fastembed / ONNX Runtime.)
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m src.embed   # download the embedding model once
echo "SETUP_EXIT=$?"
