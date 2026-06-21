#!/bin/bash
# Usage:
#   bash scripts/infer_en_ru.sh          → full test set (3003 lines)
#   bash scripts/infer_en_ru.sh 20       → smoke test (20 lines, ~5 min)
source /root/venv_diffgemma/bin/activate

# Kill any leftover python inference processes that may be holding VRAM
pkill -f "infer_diffgemma" 2>/dev/null || true
sleep 2

# Load HF token from file if not already set in environment
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN=$(cat "$HOME/.hf_token")
fi

# Must be set before Python starts (CUDA allocator reads it at init time)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_DIR="/root/Gemmadiffusion"
MAX_LINES=${1:-""}
MAX_ARG=""
[ -n "$MAX_LINES" ] && MAX_ARG="--max_lines $MAX_LINES"

python3 "$REPO_DIR/analysis/infer_diffgemma.py" \
    --src_file "$REPO_DIR/data/en-ru/test.en" \
    --ref_file "$REPO_DIR/data/en-ru/test.ru" \
    --out_dir  "$REPO_DIR/results/en-ru" \
    --src_lang "English" \
    --tgt_lang "Russian" \
    $MAX_ARG
