#!/usr/bin/env bash
# Eval experiments, run one after another to stay within free-tier rate limits.
# Each run is resumable: re-running skips questions already scored for that tag.
# The judge is always JUDGE_MODEL (default gemini-3.1-flash-lite), whatever model answers.
cd "$(dirname "$0")"
run() { .venv/bin/python -u -m eval.run_eval "$@"; }

# 300-word chunks, top 10, prompt v2, gemini-3.1-flash-lite (current best config)
CHUNK_SIZE=300 TOP_K=10 PROMPT_VERSION=v2 run --tag vector-300-k10-promptv2
# Repeat of the 600-word v2 run, to measure run-to-run variation
CHUNK_SIZE=600 TOP_K=5 PROMPT_VERSION=v2 run --tag vector-600-promptv2-run2
# Same best config answered by gemini-3.6-flash. Disabled: its free tier allows only 20 requests/day.
# CHUNK_SIZE=300 TOP_K=10 PROMPT_VERSION=v2 LLM_MODEL=gemini-3.6-flash run --tag vector-300-k10-promptv2-flash36
echo EXP_EXIT=$?
