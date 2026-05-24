#!/usr/bin/env bash
# Run once per new vast.ai instance.
# Installs Python dependencies and downloads BART-base weights.
# Usage (from /root/NER_translation): bash scripts/vastai_setup.sh
set -euo pipefail

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
PRETRAINED="$REPO_DIR/pretrained/bart-base"

echo "==> Downloading facebook/bart-base weights via wget (no HF library dependencies)..."
mkdir -p "$PRETRAINED"

if [ -f "$PRETRAINED/pytorch_model.bin" ]; then
    echo "    Already cached — skipping download."
else
    HF_BASE="https://huggingface.co/facebook/bart-base/resolve/main"
    for fname in config.json tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json; do
        if [ ! -f "$PRETRAINED/$fname" ]; then
            wget -q -L "$HF_BASE/$fname" -O "$PRETRAINED/$fname"
            echo "    Downloaded $fname"
        fi
    done
    echo "    Downloading pytorch_model.bin (~560MB)..."
    wget -L "$HF_BASE/pytorch_model.bin" -O "$PRETRAINED/pytorch_model.bin"
    echo "    BART weights saved to $PRETRAINED"
fi

echo "==> Installing Python dependencies..."
pip install -q \
    bert-score blobfile datasets \
    "huggingface-hub==0.4.0" \
    mpi4py nltk numpy pandas protobuf \
    rouge-score sacrebleu sacremoses \
    scikit-learn scipy spacy \
    tokenizers torchmetrics tqdm \
    "transformers==4.18.0"
echo "    Done."

echo ""
echo "==> Setup complete. Next step:"
echo "    bash scripts/data_en_ru.sh"
