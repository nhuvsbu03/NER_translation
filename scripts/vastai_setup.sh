#!/usr/bin/env bash
# Run once per new vast.ai instance.
# Installs Python dependencies and downloads BART-base weights.
# Usage (from /root/NER_translation): bash scripts/vastai_setup.sh
set -euo pipefail

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
PRETRAINED="$REPO_DIR/pretrained/bart-base"

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

echo "==> Downloading facebook/bart-base weights..."
mkdir -p "$PRETRAINED"

if [ -f "$PRETRAINED/pytorch_model.bin" ]; then
    echo "    Already cached — skipping download."
else
    PRETRAINED_DIR="$PRETRAINED" python3 - <<'PYEOF'
import os
pretrained = os.environ["PRETRAINED_DIR"]
from transformers import BartConfig, BartTokenizerFast, BartModel
print("    Downloading config...")
BartConfig.from_pretrained("facebook/bart-base").save_pretrained(pretrained)
print("    Downloading tokenizer...")
BartTokenizerFast.from_pretrained("facebook/bart-base").save_pretrained(pretrained)
print("    Downloading model weights (safe_serialization=False for pytorch_model.bin)...")
BartModel.from_pretrained("facebook/bart-base").save_pretrained(
    pretrained, safe_serialization=False
)
print("    Saved to", pretrained)
PYEOF
fi

echo ""
echo "==> Setup complete. Next step:"
echo "    bash scripts/data_en_ru.sh"
