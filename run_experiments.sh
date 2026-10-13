#!/usr/bin/env bash
# Week 3 experiments, run one after another to stay within free-tier rate limits.
cd "$(dirname "$0")"
PROMPT_VERSION=v2 CHUNK_SIZE=300 TOP_K=10 .venv/bin/python -u -m eval.run_eval --tag vector-300-k10-promptv2
PROMPT_VERSION=v2 .venv/bin/python -u -m eval.run_eval --tag vector-600-promptv2-run2
echo EXP_EXIT=$?
