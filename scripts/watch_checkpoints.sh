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

# Activate the image's torch venv (vast pytorch image: /venv/main)
source /venv/main/bin/activate 2>/dev/null || true

PAIR="${1:-en-ru}"
SEQ_LEN="${2:-128}"   # target sequence_len (Run A paper=64, Run B/S08=128)
PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
CKPT_DIR="$REPO_DIR/ckpts/$PAIR"
SAMPLE_DIR="$CKPT_DIR/samples"
# Strip trailing experiment suffix (en-ru-A → en-ru, en-ru → en-ru)
LANG_PAIR=$(echo "$PAIR" | sed 's/-[A-Z]$//')
DATA_DIR="$REPO_DIR/data/$LANG_PAIR"

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
        STEP=$(basename "$LATEST" | sed 's/ema_0\.[0-9]*_\([0-9]*\)\.pt/\1/' | sed 's/^0*//')
        OUT_FILE="$SAMPLE_DIR/step_$(printf '%06d' "$STEP").txt"

        # Find matching alpha_cumprod schedule (sort -V for numeric order: 66000 > 8000)
        SCHEDULE=$(ls "$CKPT_DIR"/alpha_cumprod_step_*.npy 2>/dev/null | sort -V | tail -1 || true)

        if [ -z "$SCHEDULE" ]; then
            echo "[$(date '+%H:%M:%S')] step $STEP — no schedule file yet, skipping"
        else
            echo "[$(date '+%H:%M:%S')] New checkpoint: step $STEP — running 10-sentence sample..."

            cd "$REPO_DIR"
            CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 DATALOADER_NUM_WORKERS=0 \
            python3 -u inference_main.py \
                --model_name_or_path "$LATEST" \
                --val_txt_path       "$DATA_DIR/test" \
                --out_dir            "$SAMPLE_DIR" \
                --time_schedule_path "$SCHEDULE" \
                --diffusion_steps    2000 \
                --num_samples        10 \
                --batch_size         10 \
                --sequence_len       $SEQ_LEN \
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

            # ── 200-sentence BLEU eval (2000 DDIM steps) ─────────────────────
            BLEU_OUT="$SAMPLE_DIR/bleu_step_$(printf '%06d' "$STEP").txt"
            BLEU_INFER="$SAMPLE_DIR/bleu_infer_step_$(printf '%06d' "$STEP").txt"
            echo "[$(date '+%H:%M:%S')] Running 200-sentence BLEU eval (2000 DDIM steps)..."

            CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 DATALOADER_NUM_WORKERS=0 \
            python3 -u inference_main.py \
                --model_name_or_path "$LATEST" \
                --val_txt_path       "$DATA_DIR/test" \
                --out_dir            "$SAMPLE_DIR" \
                --time_schedule_path "$SCHEDULE" \
                --diffusion_steps    2000 \
                --num_samples        200 \
                --batch_size         10 \
                --sequence_len       $SEQ_LEN \
                --sequence_len_src   128 \
                --top_p              -1 \
                --clamp              no_clamp \
                --use_ddim           True \
                --seed               42 \
                --generate_by_q      False \
                --generate_by_mix    False \
                2>&1 | tee "$BLEU_INFER"

            # Parse [hyp, ref] pairs and compute BLEU
            DECODED200=$(ls "$CKPT_DIR"/ema_0.9999_$(printf '%06d' "$STEP").pt.samples_200.steps-2000*clamp-no_clamp-normal_42.txt 2>/dev/null | sort | tail -1 || true)

            if [ -n "$DECODED200" ]; then
                python3 -c "
import json, sys
from sacrebleu.metrics import BLEU
decoded_file, step = sys.argv[1], sys.argv[2]
hyps, refs = [], []
with open(decoded_file) as f:
    for line in f:
        line = line.strip()
        if not line.startswith('['):
            continue
        try:
            pair = json.loads(line)
            if isinstance(pair, list) and len(pair) == 2:
                hyps.append(pair[0])
                refs.append(pair[1])
        except Exception:
            pass
if hyps:
    result = BLEU(tokenize='13a').corpus_score(hyps, [refs])
    print(f'Step {step} | n={len(hyps)} | BLEU={result.score:.2f} | {result}')
else:
    print(f'Step {step} | no valid pairs parsed')
" "$DECODED200" "$STEP" 2>/dev/null | tee "$BLEU_OUT"
            else
                echo "[$(date '+%H:%M:%S')] Could not find 200-sample decoded file for BLEU"
            fi

            SEEN="$LATEST"
        fi
    fi

    sleep 60
done
