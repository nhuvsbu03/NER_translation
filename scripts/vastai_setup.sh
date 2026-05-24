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
    # Required: config + model weights
    wget -nv -L "$HF_BASE/config.json"       -O "$PRETRAINED/config.json"
    echo "    Downloading pytorch_model.bin (~560MB)..."
    wget -nv -L "$HF_BASE/pytorch_model.bin" -O "$PRETRAINED/pytorch_model.bin"
    # Optional: tokenizer support files (|| true = skip if missing from HF repo)
    for fname in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json; do
        wget -nv -L "$HF_BASE/$fname" -O "$PRETRAINED/$fname" || \
            echo "    WARN: $fname not found in HF repo — skipping"
    done
    echo "    BART weights saved to $PRETRAINED"
fi

echo "==> Installing system dependencies (OpenMPI for mpi4py)..."
apt-get install -y -q libopenmpi-dev
echo "    Done."

echo "==> Installing Python dependencies..."
pip install -q \
    bert-score blobfile "datasets>=2.20" \
    "huggingface-hub>=0.20" \
    mpi4py nltk numpy pandas protobuf \
    rouge-score sacrebleu sacremoses \
    scikit-learn scipy spacy \
    tokenizers torchmetrics tqdm \
    "transformers==4.18.0"
echo "    Done."

echo ""
echo "==> Setup complete. Next step:"
echo "    bash scripts/data_en_ru.sh"
