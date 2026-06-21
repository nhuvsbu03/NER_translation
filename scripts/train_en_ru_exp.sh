#!/usr/bin/env bash
# Session-09 cause-finding experiments (run ONE per instance).
#   bash scripts/train_en_ru_exp.sh A   # Run A — faithful paper recipe (512/2048, batch128, seq64, stride20000)
#   bash scripts/train_en_ru_exp.sh B   # Run B — capacity-only change from S08 (512/2048 but S08 batch32/seq128/stride2000)
# Launches training + checkpoint watcher (per-step 200-sentence BLEU) + collapse monitor in tmux.
set -euo pipefail

EXP="${1:-}"
if [ "$EXP" != "A" ] && [ "$EXP" != "B" ]; then
    echo "Usage: bash scripts/train_en_ru_exp.sh A|B"; exit 1
fi

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
CKPT_DIR="$REPO_DIR/ckpts/en-ru-$EXP"
LOG_DIR="$CKPT_DIR/log"
LOG_FILE="$LOG_DIR/train.log"
PRETRAINED="$REPO_DIR/pretrained/bart-base"
mkdir -p "$LOG_DIR"

# ── Per-experiment hyperparameters ───────────────────────────────────────────
# Common (both): 512/2048 capacity, 6+6 layers, 8 heads, init_pretrained False, sqrt, 2000 steps
if [ "$EXP" = "A" ]; then
    echo "==> Run A — faithful paper recipe (iwslt_en_de.sh values on en-ru)"
    BATCH=128 ; LR=1e-4    ; SEQ_LEN=64  ; STRIDE=20000
else
    echo "==> Run B — capacity-only change from the S08 baseline"
    BATCH=32  ; LR=3.75e-5 ; SEQ_LEN=128 ; STRIDE=2000
fi
SEQ_LEN_SRC=128

# ── Auto-resume (fresh start expected: ckpts/en-ru empty) ────────────────────
RESUME_CKPT=""
if ls "$CKPT_DIR"/model*.pt 2>/dev/null | grep -v ema | sort | tail -1 > /dev/null 2>&1; then
    RESUME_CKPT=$(ls "$CKPT_DIR"/model*.pt 2>/dev/null | grep -v ema | sort | tail -1)
    echo "==> Resuming from $(basename "$RESUME_CKPT")"
else
    echo "==> Starting from scratch."
fi

TRAIN_CMD="source /venv/main/bin/activate && cd $REPO_DIR && CUDA_VISIBLE_DEVICES=0 DIFFUSION_BLOB_LOGDIR=$LOG_DIR TRANSFORMERS_OFFLINE=1 \
python3 -u main.py \
  --checkpoint_path $CKPT_DIR \
  --src en --tgt ru \
  --train_txt_path ./data/en-ru/train \
  --val_txt_path   ./data/en-ru/valid \
  --dataset        en-ru \
  --config_name    $PRETRAINED \
  --diffusion_steps    2000 \
  --noise_schedule     sqrt \
  --sequence_len       $SEQ_LEN \
  --sequence_len_src   $SEQ_LEN_SRC \
  --batch_size         $BATCH \
  --lr                 $LR \
  --lr_anneal_steps    1000000 \
  --warmup             10000 \
  --save_interval      25000 \
  --eval_interval      5000 \
  --log_interval       100 \
  --schedule_update_stride $STRIDE \
  --loss_update_granu  20 \
  --schedule_sampler   uniform \
  --encoder_layers     6 \
  --decoder_layers     6 \
  --num_heads          8 \
  --in_channel         512 \
  --out_channel        512 \
  --num_channels       2048 \
  --vocab_size         32005 \
  --dropout            0.3 \
  --predict_xstart     True \
  --seed               42 \
  --init_pretrained    False \
  --freeze_embeddings  False \
  --use_pretrained_embeddings False \
  --resume_checkpoint  '$RESUME_CKPT' \
  2>&1 | tee $LOG_FILE"

tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "bash -c \"$TRAIN_CMD\""

WATCH_CMD="cd $PROJECT_ROOT && bash scripts/watch_checkpoints.sh en-ru-$EXP $SEQ_LEN"
tmux new-window -t train -n watch "bash -c \"$WATCH_CMD\""

MON_CMD="cd $PROJECT_ROOT && bash scripts/monitor_training.sh en-ru-$EXP $EXP"
tmux kill-session -t monitor 2>/dev/null || true
tmux new-session -d -s monitor "bash -c \"$MON_CMD\""

echo "==> Run $EXP launched: batch=$BATCH lr=$LR seq_len=$SEQ_LEN stride=$STRIDE (512/2048, init_pretrained=False)"
echo "    Checkpoints: $CKPT_DIR"
echo "    Attach:      tmux attach -t train     (Ctrl+B n to switch to watch window)"
echo "    Train log:   tail -f $LOG_FILE"
echo "    BLEU/step:   ls $CKPT_DIR/samples/bleu_step_*.txt"
