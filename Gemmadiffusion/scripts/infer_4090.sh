#!/bin/bash
# Generic inference script optimised for RTX 4090 (24 GB).
# Usage:
#   bash scripts/infer_4090.sh en-ru          → full test set
#   bash scripts/infer_4090.sh en-ru 20       → smoke test (20 lines)
#   bash scripts/infer_4090.sh en-zh 20
#   bash scripts/infer_4090.sh zh-en 20
#   bash scripts/infer_4090.sh en-ja 20
#   bash scripts/infer_4090.sh ja-en 20

set -e
source /root/venv_diffgemma/bin/activate

# Kill any leftover inference processes holding VRAM
pkill -f "infer_diffgemma" 2>/dev/null || true
sleep 2

# Load HF token from file if not already set in environment
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN=$(cat "$HOME/.hf_token")
fi

# Must be set before Python starts (CUDA allocator reads it at init time)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

REPO_DIR="/root/Gemmadiffusion"
PAIR=${1:-en-ru}
MAX_LINES=${2:-""}
# Smoke test uses short outputs to stay within 48 GB VRAM; full run uses 150 tokens
MAX_NEW_TOKENS=${3:-150}
MAX_ARG=""
[ -n "$MAX_LINES" ] && MAX_ARG="--max_lines $MAX_LINES"

# Map language pair to human-readable names for the prompt
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

# Infer source/target file extensions from the pair code
SRC_EXT="${PAIR%%-*}"   # everything before the dash
TGT_EXT="${PAIR##*-}"   # everything after the dash

echo "=== DiffusionGemma inference: $PAIR ($SRC_LANG → $TGT_LANG) ==="
[ -n "$MAX_LINES" ] && echo "    Smoke-test mode: $MAX_LINES lines"

python3 "$REPO_DIR/analysis/infer_diffgemma.py" \
    --src_file "$REPO_DIR/data/$PAIR/test.$SRC_EXT" \
    --ref_file "$REPO_DIR/data/$PAIR/test.$TGT_EXT" \
    --out_dir  "$REPO_DIR/results/$PAIR" \
    --src_lang "$SRC_LANG" \
    --tgt_lang "$TGT_LANG" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    $MAX_ARG
