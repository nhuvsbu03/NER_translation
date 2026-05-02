#!/usr/bin/env bash
# Push local code/data to a vast.ai instance.
# Usage:
#   ./scripts/push_vastai.sh [ssh-alias]         # code + data + tokenizer + checkpoint
#   ./scripts/push_vastai.sh [ssh-alias] --pull  # pull results back from instance
#
# Add an entry in ~/.ssh/config:
#   Host vastai
#       HostName <instance-IP>
#       Port     <instance-PORT>
#       User     root
#       IdentityFile ~/.ssh/id_ed25519
#
set -euo pipefail

REMOTE="${1:-vastai}"
MODE="${2:-}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOLUME_DIR="/workspace"

# ── CONFIGURE THESE PATHS ─────────────────────────────────────────────────
# Folder on Google Drive that contains vocab.json and merges.txt
GDRIVE_TOK_DIR="$HOME/Library/CloudStorage/GoogleDrive-nhuvsbu@gmail.com/My Drive/BUyen_Qnhu++/src/SeqDiffuSeq/data/en-zh"

# Folder on Google Drive that contains model*.pt and alpha_cumprod_step_*.npy
GDRIVE_CKPT_DIR="$HOME/Library/CloudStorage/GoogleDrive-nhuvsbu@gmail.com/My Drive/BUyen_Qnhu++/src/SeqDiffuSeq/ckpts/en-zh"
# ─────────────────────────────────────────────────────────────────────────

# ── Code ──────────────────────────────────────────────────────────────────
echo "==> Pushing SeqDiffuSeq/ → $REMOTE:/root/SeqDiffuSeq/"
rsync -avz --progress \
  --exclude='*.pt' \
  --exclude='*.npy' \
  --exclude='out/' \
  --exclude='ckpts/' \
  --exclude='data/' \
  --exclude='__pycache__/' \
  --exclude='.DS_Store' \
  "$PROJECT_ROOT/SeqDiffuSeq/" \
  "$REMOTE:/root/SeqDiffuSeq/"

echo "==> Pushing train_dataset/ → $REMOTE:/root/train_dataset/"
rsync -avz --progress \
  --exclude='.DS_Store' \
  "$PROJECT_ROOT/train_dataset/" \
  "$REMOTE:/root/train_dataset/"

echo "==> Pushing notebook…"
rsync -avz --progress \
  "$PROJECT_ROOT/vastai_training.ipynb" \
  "$REMOTE:/root/vastai_training.ipynb"

# ── Tokenizer (vocab.json + merges.txt) ───────────────────────────────────
echo "==> Pushing tokenizer…"
ssh "$REMOTE" "mkdir -p /root/SeqDiffuSeq/data/en-zh"

for fname in vocab.json merges.txt; do
  # prefer local copy; fall back to Google Drive
  LOCAL_TOK="$PROJECT_ROOT/SeqDiffuSeq/data/en-zh/$fname"
  DRIVE_TOK="$GDRIVE_TOK_DIR/$fname"

  if [[ -f "$LOCAL_TOK" ]]; then
    echo "    $fname (local)"
    rsync -avz --progress "$LOCAL_TOK" "$REMOTE:/root/SeqDiffuSeq/data/en-zh/$fname"
  elif [[ -f "$DRIVE_TOK" ]]; then
    echo "    $fname (Google Drive)"
    rsync -avz --progress "$DRIVE_TOK" "$REMOTE:/root/SeqDiffuSeq/data/en-zh/$fname"
  else
    echo "    WARN: $fname not found locally or on Drive — skipping"
  fi
done

# ── Latest checkpoint + schedule ──────────────────────────────────────────
echo "==> Pushing latest checkpoint + schedule → $REMOTE:$VOLUME_DIR/ckpts/en-zh/"
ssh "$REMOTE" "mkdir -p $VOLUME_DIR/ckpts/en-zh"

# Find latest model*.pt (non-EMA), prefer local then Drive
CKPT_SEARCH_DIRS=("$PROJECT_ROOT/SeqDiffuSeq/ckpts/en-zh" "$GDRIVE_CKPT_DIR")
LATEST_CKPT=""
for dir in "${CKPT_SEARCH_DIRS[@]}"; do
  if [[ -d "$dir" ]]; then
    found=$(find "$dir" -maxdepth 1 -name "model*.pt" ! -name "*ema*" 2>/dev/null | sort | tail -1)
    if [[ -n "$found" ]]; then
      LATEST_CKPT="$found"
      break
    fi
  fi
done

if [[ -n "$LATEST_CKPT" ]]; then
  echo "    $(basename "$LATEST_CKPT")"
  rsync -avz --progress "$LATEST_CKPT" "$REMOTE:$VOLUME_DIR/ckpts/en-zh/"
else
  echo "    WARN: no model*.pt found — skipping"
fi

# Find all alpha_cumprod_step_*.npy, prefer local then Drive
NPY_SEARCH_DIRS=("$PROJECT_ROOT/SeqDiffuSeq/ckpts/en-zh" "$GDRIVE_CKPT_DIR")
NPY_PUSHED=0
for dir in "${NPY_SEARCH_DIRS[@]}"; do
  npys=$(find "$dir" -maxdepth 1 -name "alpha_cumprod_step_*.npy" 2>/dev/null | sort | tail -1)
  if [[ -n "$npys" ]]; then
    echo "    $(basename "$npys")"
    rsync -avz --progress "$npys" "$REMOTE:$VOLUME_DIR/ckpts/en-zh/"
    NPY_PUSHED=1
    break
  fi
done
if [[ $NPY_PUSHED -eq 0 ]]; then
  echo "    WARN: no alpha_cumprod_step_*.npy found — skipping"
fi

if [[ "$MODE" == "--pull" ]]; then
  LOCAL_RESULTS="$PROJECT_ROOT/SeqDiffuSeq/results/"
  mkdir -p "$LOCAL_RESULTS"
  echo "==> Pulling results ← $REMOTE:$VOLUME_DIR/ckpts/en-zh/inference_out/"
  rsync -avz --progress \
    "$REMOTE:$VOLUME_DIR/ckpts/en-zh/inference_out/" \
    "$LOCAL_RESULTS"
fi

echo "==> Done."
