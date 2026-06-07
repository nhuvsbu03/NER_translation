#!/usr/bin/env bash
# Watch for new EMA checkpoints and run 10-sentence mini-inference at each one.
# Designed to run alongside training in a second tmux window.
#
# Usage (from /root/NER_translation):
#   tmux new-window -t train -n watch "bash scripts/watch_checkpoints.sh en-ru"
#
# Or manually:
#   bash scripts/watch_checkpoints.sh en-ru
#
# Arg 1: language pair (default: en-ru)
set -euo pipefail

PAIR="${1:-en-ru}"
PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
CKPT_DIR="$REPO_DIR/ckpts/$PAIR"
SAMPLE_DIR="$CKPT_DIR/samples"
DATA_DIR="$REPO_DIR/data/$PAIR"

mkdir -p "$SAMPLE_DIR"

echo "==> Watching $CKPT_DIR for new EMA checkpoints..."
echo "    Samples will be saved to $SAMPLE_DIR/step_<N>.txt"
echo "    (Ctrl+C to stop)"
echo ""

SEEN=""

while true; do
    # Find latest EMA checkpoint
    LATEST=$(ls "$CKPT_DIR"/ema_0.9999_*.pt 2>/dev/null | sort | tail -1 || true)

    if [ -n "$LATEST" ] && [ "$LATEST" != "$SEEN" ]; then
        STEP=$(basename "$LATEST" | grep -o '[0-9]*' | head -1 | sed 's/^0*//')
        OUT_FILE="$SAMPLE_DIR/step_$(printf '%06d' "$STEP").txt"

        # Find matching alpha_cumprod schedule
        SCHEDULE=$(ls "$CKPT_DIR"/alpha_cumprod_step_*.npy 2>/dev/null | sort | tail -1 || true)

        if [ -z "$SCHEDULE" ]; then
            echo "[$(date '+%H:%M:%S')] step $STEP — no schedule file yet, skipping"
        else
            echo "[$(date '+%H:%M:%S')] New checkpoint: step $STEP — running 10-sentence sample..."

            cd "$REPO_DIR"
            CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 \
            python3 -u inference_main.py \
                --model_name_or_path "$LATEST" \
                --val_txt_path       "$DATA_DIR/test" \
                --out_dir            "$SAMPLE_DIR" \
                --time_schedule_path "$SCHEDULE" \
                --diffusion_steps    2000 \
                --num_samples        10 \
                --batch_size         10 \
                --sequence_len       128 \
                --sequence_len_src   128 \
                --top_p              -1 \
                --clamp              no_clamp \
                --use_ddim           True \
                --seed               42 \
                --generate_by_q      False \
                --generate_by_mix    False \
                2>&1 | tee "$OUT_FILE"

            echo ""
            echo "--- Step $STEP sample (first 5 lines) ---"
            head -5 "$OUT_FILE" 2>/dev/null || true
            echo "-----------------------------------------"
            echo ""

            SEEN="$LATEST"
        fi
    fi

    sleep 60
done
