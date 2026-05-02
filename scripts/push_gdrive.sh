#!/usr/bin/env bash
# Push local code changes to Google Drive (requires Google Drive desktop app).
# Usage:  ./scripts/push_gdrive.sh [gdrive-root]
#   gdrive-root defaults to "~/Library/CloudStorage/GoogleDrive-nhuvsbu@gmail.com/My Drive/BUyen_Qnhu++/src"
#
set -euo pipefail

GDRIVE_ROOT="${1:-$HOME/Library/CloudStorage/GoogleDrive-nhuvsbu@gmail.com/My Drive/BUyen_Qnhu++/src}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Syncing SeqDiffuSeq/ → $GDRIVE_ROOT/SeqDiffuSeq/"
rsync -av --delete \
  --exclude='*.pt' \
  --exclude='*.npy' \
  --exclude='out/' \
  --exclude='data/' \
  --exclude='ckpts/' \
  --exclude='train_dataset/' \
  --exclude='__pycache__/' \
  --exclude='.DS_Store' \
  "$PROJECT_ROOT/SeqDiffuSeq/" "$GDRIVE_ROOT/SeqDiffuSeq/"

echo "==> Syncing train_dataset/ → $GDRIVE_ROOT/train_dataset/"
rsync -av --delete \
  --exclude='.DS_Store' \
  "$PROJECT_ROOT/train_dataset/" "$GDRIVE_ROOT/train_dataset/"

echo "==> Pushing notebook…"
cp "$PROJECT_ROOT/zh_vi_translation.ipynb" "$GDRIVE_ROOT/zh_vi_translation.ipynb"

echo "==> Done."
