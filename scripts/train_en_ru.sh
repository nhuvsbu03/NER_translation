#!/usr/bin/env bash
# Launch SeqDiffuSeq EN→RU training in a tmux session named 'train'.
# Auto-resumes from the latest checkpoint in ckpts/en-ru/.
# Returns immediately — training runs in the background.
# Usage (from /root/NER_translation): bash scripts/train_en_ru.sh
set -euo pipefail

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
CKPT_DIR="$REPO_DIR/ckpts/en-ru"
LOG_DIR="$CKPT_DIR/log"
LOG_FILE="$LOG_DIR/train.log"
PRETRAINED="$REPO_DIR/pretrained/bart-base"

mkdir -p "$LOG_DIR"

# ── Auto-resume: find latest non-EMA checkpoint ──────────────────────────────
RESUME_CKPT=""
if ls "$CKPT_DIR"/model*.pt 2>/dev/null | grep -v ema | sort | tail -1 > /dev/null 2>&1; then
    RESUME_CKPT=$(ls "$CKPT_DIR"/model*.pt 2>/dev/null | grep -v ema | sort | tail -1)
fi

if [ -n "$RESUME_CKPT" ]; then
    STEP=$(basename "$RESUME_CKPT" | grep -o '[0-9]*' | head -1 | sed 's/^0*//')
    echo "==> Resuming from step $STEP: $(basename "$RESUME_CKPT")"
else
    echo "==> No checkpoint found — starting from scratch."
fi

# ── Build training command ────────────────────────────────────────────────────
TRAIN_CMD="cd $REPO_DIR && CUDA_VISIBLE_DEVICES=0 DIFFUSION_BLOB_LOGDIR=$LOG_DIR TRANSFORMERS_OFFLINE=1 \
python3 -u main.py \
  --checkpoint_path $CKPT_DIR \
  --src en \
  --tgt ru \
  --train_txt_path ./data/en-ru/train \
  --val_txt_path   ./data/en-ru/valid \
  --dataset        en-ru \
  --config_name    $PRETRAINED \
  --diffusion_steps    2000 \
  --noise_schedule     sqrt \
  --sequence_len       128 \
  --sequence_len_src   128 \
  --batch_size         64 \
  --lr                 1e-4 \
  --lr_anneal_steps    200000 \
  --warmup             10000 \
  --save_interval      10000 \
  --eval_interval      5000 \
  --log_interval       100 \
  --schedule_update_stride 2000 \
  --loss_update_granu  20 \
  --schedule_sampler   uniform \
  --encoder_layers     6 \
  --decoder_layers     6 \
  --num_heads          12 \
  --in_channel         768 \
  --out_channel        768 \
  --num_channels       3072 \
  --vocab_size         32005 \
  --dropout            0.3 \
  --predict_xstart     True \
  --seed               42 \
  --init_pretrained    True \
  --freeze_embeddings  False \
  --use_pretrained_embeddings False \
  --resume_checkpoint  '$RESUME_CKPT' \
  2>&1 | tee $LOG_FILE"

# ── Launch in tmux ────────────────────────────────────────────────────────────
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "bash -c \"$TRAIN_CMD\""

echo "==> Training launched in tmux session 'train'."
echo ""
echo "    Monitor:   tmux attach -t train   (detach: Ctrl+B then D)"
echo "    Log tail:  tail -f $LOG_FILE"
echo "    Check loss: grep 'loss:' $LOG_FILE | tail -20"
