#!/usr/bin/env bash
# Run SeqDiffuSeq inference on the full EN-RU newstest2014 test set.
# Auto-finds the latest EMA checkpoint and alpha_cumprod schedule.
# Usage (from /root/NER_translation): bash scripts/infer_en_ru.sh
set -euo pipefail

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
CKPT_DIR="$REPO_DIR/ckpts/en-ru"
OUT_DIR="$CKPT_DIR/inference_out"

mkdir -p "$OUT_DIR"

# ── Find latest EMA checkpoint ────────────────────────────────────────────────
EMA_CKPT=$(ls "$CKPT_DIR"/ema_0.9999_*.pt 2>/dev/null | sort | tail -1 || true)
if [ -z "$EMA_CKPT" ]; then
    echo "ERROR: No EMA checkpoint found in $CKPT_DIR"
    echo "       Training must complete at least one save_interval (10,000 steps)."
    exit 1
fi
echo "==> Checkpoint: $(basename "$EMA_CKPT")"

# ── Find latest alpha_cumprod schedule ───────────────────────────────────────
SCHEDULE=$(ls "$CKPT_DIR"/alpha_cumprod_step_*.npy 2>/dev/null | sort | tail -1 || true)
if [ -z "$SCHEDULE" ]; then
    echo "ERROR: No alpha_cumprod_step_*.npy found in $CKPT_DIR"
    exit 1
fi
echo "==> Schedule:   $(basename "$SCHEDULE")"

# ── Count test sentences ──────────────────────────────────────────────────────
TEST_FILE="$REPO_DIR/data/en-ru/test.en"
NUM_TEST=$(wc -l < "$TEST_FILE")
echo "==> Test set:   $NUM_TEST sentences (num_samples=-1 → full set)"
echo ""

# ── Run inference ─────────────────────────────────────────────────────────────
cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 \
python3 -u inference_main.py \
    --model_name_or_path "$EMA_CKPT" \
    --val_txt_path       ./data/en-ru/test \
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
echo "    Pull results with: .\\scripts\\pull_results.ps1 (from Windows)"
