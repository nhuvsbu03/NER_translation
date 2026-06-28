#!/usr/bin/env bash
# Run 200-sentence BLEU eval on one or more specific EMA checkpoints (Run A).
# Usage (from /root/NER_translation):
#   bash scripts/infer_en_ru_ckpt.sh 600000 625000
set -euo pipefail

source /venv/main/bin/activate 2>/dev/null || true

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
CKPT_DIR="$REPO_DIR/ckpts/en-ru-A"
SAMPLE_DIR="$CKPT_DIR/samples"
DATA_DIR="$REPO_DIR/data/en-ru"

mkdir -p "$SAMPLE_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/infer_en_ru_ckpt.sh 600000 625000"
    exit 1
fi

# Find alpha_cumprod schedule (use latest available, fall back gracefully)
SCHEDULE=$(ls "$CKPT_DIR"/alpha_cumprod_step_*.npy 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$SCHEDULE" ]; then
    echo "WARNING: No alpha_cumprod schedule found in $CKPT_DIR — using empty string"
    SCHEDULE=""
fi
echo "==> Schedule: ${SCHEDULE:-'(none, using default sqrt)'}"

for STEP in "$@"; do
    STEP_PAD=$(printf '%06d' "$STEP")
    EMA_CKPT="$CKPT_DIR/ema_0.9999_${STEP_PAD}.pt"

    if [ ! -f "$EMA_CKPT" ]; then
        echo "ERROR: $EMA_CKPT not found, skipping"
        continue
    fi

    BLEU_OUT="$SAMPLE_DIR/bleu_step_${STEP_PAD}.txt"
    BLEU_INFER="$SAMPLE_DIR/bleu_infer_step_${STEP_PAD}.txt"
    echo ""
    echo "==> Step $STEP: running 200-sentence eval..."

    SCHED_ARG=""
    [ -n "$SCHEDULE" ] && SCHED_ARG="--time_schedule_path $SCHEDULE"

    cd "$REPO_DIR"
    CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 DATALOADER_NUM_WORKERS=0 \
    python3 -u inference_main.py \
        --model_name_or_path "$EMA_CKPT" \
        --val_txt_path       "$DATA_DIR/test" \
        --out_dir            "$SAMPLE_DIR" \
        $SCHED_ARG \
        --diffusion_steps    2000 \
        --num_samples        200 \
        --batch_size         10 \
        --sequence_len       64 \
        --sequence_len_src   128 \
        --top_p              -1 \
        --clamp              no_clamp \
        --use_ddim           True \
        --seed               42 \
        --generate_by_q      False \
        --generate_by_mix    False \
        2>&1 | tee "$BLEU_INFER"

    DECODED=$(ls "$CKPT_DIR/ema_0.9999_${STEP_PAD}.pt.samples_200.steps-2000"*"clamp-no_clamp-normal_42.txt" 2>/dev/null | sort | tail -1 || true)

    if [ -n "$DECODED" ]; then
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
" "$DECODED" "$STEP" 2>/dev/null | tee "$BLEU_OUT"
        echo "==> BLEU saved to $BLEU_OUT"
    else
        echo "WARNING: decoded output file not found for step $STEP"
    fi
done

echo ""
echo "==> Done. Results in $SAMPLE_DIR"
