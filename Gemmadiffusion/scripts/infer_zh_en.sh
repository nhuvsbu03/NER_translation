#!/bin/bash
# Usage: bash scripts/infer_zh_en.sh [MAX_LINES]
source /root/venv_diffgemma/bin/activate
pkill -f "infer_diffgemma" 2>/dev/null || true; sleep 2
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.hf_token" ]; then export HF_TOKEN=$(cat "$HOME/.hf_token"); fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_DIR="/root/Gemmadiffusion"
MAX_ARG=""; [ -n "$1" ] && MAX_ARG="--max_lines $1"

python3 "$REPO_DIR/analysis/infer_diffgemma.py" \
    --src_file "$REPO_DIR/data/zh-en/test.zh" \
    --ref_file "$REPO_DIR/data/zh-en/test.en" \
    --out_dir  "$REPO_DIR/results/zh-en" \
    --src_lang "Chinese" --tgt_lang "English" $MAX_ARG
