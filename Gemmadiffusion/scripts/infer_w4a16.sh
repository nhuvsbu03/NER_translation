#!/bin/bash
# Inference script for GoedelMachines/diffusiongemma-26B-A4B-w4a16 (W4A16 quantized).
# Fits in ~11.5 GB GPU — runs on any 16 GB+ GPU; 2-3× faster than BF16 version.
#
# Usage:
#   bash scripts/infer_w4a16.sh en-ru           → full test set
#   bash scripts/infer_w4a16.sh en-ru 5         → smoke test (5 lines)
#   bash scripts/infer_w4a16.sh en-ru 5 150     → smoke test, explicit max_new_tokens

set -e
source /root/venv_diffgemma/bin/activate

pkill -f "infer_diffgemma_w4a16" 2>/dev/null || true
sleep 1

if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN=$(cat "$HOME/.hf_token")
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

REPO_DIR="/root/Gemmadiffusion"
PAIR=${1:-en-ru}
MAX_LINES=${2:-""}
MAX_NEW_TOKENS=${3:-150}
MAX_ARG=""
[ -n "$MAX_LINES" ] && MAX_ARG="--max_lines $MAX_LINES"

case "$PAIR" in
    en-ru) SRC_LANG="English";  TGT_LANG="Russian" ;;
    en-zh) SRC_LANG="English";  TGT_LANG="Chinese" ;;
    zh-en) SRC_LANG="Chinese";  TGT_LANG="English" ;;
    en-ja) SRC_LANG="English";  TGT_LANG="Japanese" ;;
    ja-en) SRC_LANG="Japanese"; TGT_LANG="English" ;;
    *)
        echo "Unknown pair '$PAIR'. Supported: en-ru en-zh zh-en en-ja ja-en"
        exit 1
        ;;
esac

SRC_EXT="${PAIR%%-*}"
TGT_EXT="${PAIR##*-}"

echo "=== DiffusionGemma W4A16 inference: $PAIR ($SRC_LANG → $TGT_LANG) ==="
[ -n "$MAX_LINES" ] && echo "    Smoke-test mode: $MAX_LINES lines"

python3 "$REPO_DIR/analysis/infer_diffgemma_w4a16.py" \
    --src_file "$REPO_DIR/data/$PAIR/test.$SRC_EXT" \
    --ref_file "$REPO_DIR/data/$PAIR/test.$TGT_EXT" \
    --out_dir  "$REPO_DIR/results/${PAIR}_w4a16" \
    --src_lang "$SRC_LANG" \
    --tgt_lang "$TGT_LANG" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    $MAX_ARG
