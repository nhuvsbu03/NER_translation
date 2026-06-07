#!/usr/bin/env bash
# Launch SeqDiffuSeq JA→EN training (mBART-large-cc25 backbone).
# Auto-resumes from the latest checkpoint in ckpts/ja-en/.
# Returns immediately — training runs in the background (vast.ai/tmux).
#
# On Google Colab, download mBART first with this Python cell:
#   from transformers import MBartForConditionalGeneration
#   model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
#   model.save_pretrained('SeqDiffuSeq/pretrained/mbart-large', safe_serialization=False)
#
# Usage (from /root/NER_translation): bash scripts/train_ja_en.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/NER_translation}"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
CKPT_DIR="$REPO_DIR/ckpts/ja-en"
LOG_DIR="$CKPT_DIR/log"
LOG_FILE="$LOG_DIR/train.log"
PRETRAINED="$REPO_DIR/pretrained/mbart-large"

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
  --src ja \
  --tgt en \
  --train_txt_path ./data/ja-en/train \
  --val_txt_path   ./data/ja-en/valid \
  --dataset        ja-en \
  --config_name    $PRETRAINED \
  --diffusion_steps    2000 \
  --noise_schedule     sqrt \
  --sequence_len       128 \
  --sequence_len_src   128 \
  --batch_size         32 \
  --lr                 1e-4 \
  --lr_anneal_steps    25000 \
  --warmup             2500 \
  --save_interval      5000 \
  --eval_interval      2500 \
  --log_interval       100 \
  --schedule_update_stride 2000 \
  --loss_update_granu  20 \
  --schedule_sampler   uniform \
  --encoder_layers     12 \
  --decoder_layers     12 \
  --num_heads          16 \
  --in_channel         1024 \
  --out_channel        1024 \
  --num_channels       4096 \
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
