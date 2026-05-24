#!/usr/bin/env bash
# Run SeqDiffuSeq inference on the ZH-EN test set (newstest2017, 2,001 pairs).
# Supports targeting a specific checkpoint step for staged evaluation.
# Usage (from /root/NER_translation):
#   bash scripts/infer_zh_en.sh           → uses latest EMA checkpoint
#   bash scripts/infer_zh_en.sh 50000     → targets ema_0.9999_050000.pt
#   bash scripts/infer_zh_en.sh 100000    → targets ema_0.9999_100000.pt
#   bash scripts/infer_zh_en.sh 200000    → targets ema_0.9999_200000.pt
set -euo pipefail

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
CKPT_DIR="$REPO_DIR/ckpts/zh-en"

STEP="${1:-}"

# ── Find EMA checkpoint ───────────────────────────────────────────────────────
if [ -n "$STEP" ]; then
    STEP_PAD=$(printf '%06d' "$STEP")
    EMA_CKPT="$CKPT_DIR/ema_0.9999_${STEP_PAD}.pt"
    if [ ! -f "$EMA_CKPT" ]; then
        echo "ERROR: Checkpoint not found: $EMA_CKPT"
        echo "       Available EMA checkpoints:"
        ls "$CKPT_DIR"/ema_0.9999_*.pt 2>/dev/null || echo "       (none)"
        exit 1
    fi
    OUT_DIR="$CKPT_DIR/inference_out_step${STEP_PAD}"
else
    EMA_CKPT=$(ls "$CKPT_DIR"/ema_0.9999_*.pt 2>/dev/null | sort | tail -1 || true)
    if [ -z "$EMA_CKPT" ]; then
        echo "ERROR: No EMA checkpoint found in $CKPT_DIR"
        echo "       Training must complete at least one save_interval (10,000 steps)."
        exit 1
    fi
    OUT_DIR="$CKPT_DIR/inference_out"
fi

echo "==> Checkpoint: $(basename "$EMA_CKPT")"
echo "==> Output dir: $OUT_DIR"
mkdir -p "$OUT_DIR"

# ── Find alpha_cumprod schedule ───────────────────────────────────────────────
SCHEDULE=$(ls "$CKPT_DIR"/alpha_cumprod_step_*.npy 2>/dev/null | sort | tail -1 || true)
if [ -z "$SCHEDULE" ]; then
    echo "ERROR: No alpha_cumprod_step_*.npy found in $CKPT_DIR"
    exit 1
fi
echo "==> Schedule:   $(basename "$SCHEDULE")"

# ── Count test sentences ──────────────────────────────────────────────────────
TEST_FILE="$REPO_DIR/data/zh-en/test.zh"
NUM_TEST=$(wc -l < "$TEST_FILE")
echo "==> Test set:   $NUM_TEST sentences (newstest2017)"
echo ""

# ── Run inference ─────────────────────────────────────────────────────────────
cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 \
python3 -u inference_main.py \
    --model_name_or_path "$EMA_CKPT" \
    --val_txt_path       ./data/zh-en/test \
    --out_dir            "$OUT_DIR" \
    --time_schedule_path "$SCHEDULE" \
    --diffusion_steps    2000 \
    --num_samples        -1 \
    --batch_size         50 \
    --sequence_len       128 \
    --sequence_len_src   128 \
    --top_p              -1 \
    --clamp              no_clamp \
    --use_ddim           True \
    --seed               42 \
    --generate_by_q      False \
    --generate_by_mix    False

echo ""
echo "==> Inference complete. Output in: $OUT_DIR"
echo "    Pull results with: .\\scripts\\pull_results.ps1 -Pair zh-en (from Windows)"
