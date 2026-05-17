#!/usr/bin/env bash
# Download WMT14 EN-RU from HuggingFace, save as plain text, train 32k BPE tokenizer.
# Run once for the EN-RU experiment (from /root/NER_translation).
# Usage: bash scripts/data_en_ru.sh
set -euo pipefail

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
RAW_DIR="$PROJECT_ROOT/train_dataset/wmt14_en_ru"
DATA_DIR="$REPO_DIR/data/en-ru"

mkdir -p "$RAW_DIR" "$DATA_DIR"

# ── Step 1: Download WMT14 EN-RU ─────────────────────────────────────────────
echo "==> Downloading WMT14 ru-en from HuggingFace datasets..."
echo "    (train ~2.5M pairs, valid/test ~3,003 pairs each — takes a few minutes)"

python3 - <<PYEOF
import os
from datasets import load_dataset

raw_dir = "$RAW_DIR"
ds = load_dataset("wmt14", "ru-en", cache_dir="/root/.cache/hf")

split_map = {"train": "train", "valid": "validation", "test": "test"}
for name, hf_split in split_map.items():
    en_path = os.path.join(raw_dir, f"{name}.en")
    ru_path = os.path.join(raw_dir, f"{name}.ru")
    if os.path.exists(en_path) and os.path.exists(ru_path):
        n = sum(1 for _ in open(en_path, encoding="utf-8"))
        print(f"  Exists  {name}: {n:,} pairs — skipping")
        continue
    count = 0
    with open(en_path, "w", encoding="utf-8") as f_en, \
         open(ru_path, "w", encoding="utf-8") as f_ru:
        for ex in ds[hf_split]:
            f_en.write(ex["translation"]["en"].strip() + "\n")
            f_ru.write(ex["translation"]["ru"].strip() + "\n")
            count += 1
    print(f"  Wrote   {name}: {count:,} pairs")

print("Download complete.")
PYEOF

# ── Step 2: Copy to SeqDiffuSeq/data/en-ru/ ─────────────────────────────────
echo "==> Copying data to $DATA_DIR..."
for split in train valid test; do
    for lang in en ru; do
        src="$RAW_DIR/$split.$lang"
        dst="$DATA_DIR/$split.$lang"
        if [ ! -f "$dst" ]; then
            cp "$src" "$dst"
            n=$(wc -l < "$dst")
            printf "  Copied  %s.%s  (%s lines)\n" "$split" "$lang" "$n"
        else
            printf "  Exists  %s.%s\n" "$split" "$lang"
        fi
    done
done

# ── Step 3: Train 32k BPE tokenizer ─────────────────────────────────────────
echo "==> Training 32k BPE tokenizer on EN+RU train corpus..."
TOK_VOCAB="$DATA_DIR/vocab.json"
if [ -f "$TOK_VOCAB" ]; then
    echo "    Tokenizer already exists — skipping. Delete $TOK_VOCAB to retrain."
else
    # tokenizer_utils.py looks for files with 'train' in ./data/en-ru/ from cwd=REPO_DIR
    cd "$REPO_DIR"
    python3 tokenizer_utils.py train-byte-level en-ru 32000
    echo "    Tokenizer saved to $DATA_DIR"
fi

# ── Step 4: Sanity check ─────────────────────────────────────────────────────
echo "==> Sanity check: tokenize 'Москва' (should be ≤ 3 tokens)..."
cd "$REPO_DIR"
python3 - <<PYEOF
from tokenizer_utils import read_byte_level
tok = read_byte_level("./data/en-ru")
ids = tok.encode("Москва").ids
print(f"  'Москва' → {len(ids)} tokens: {ids}")
if len(ids) > 5:
    print("  WARN: more tokens than expected — check vocab coverage")
else:
    print("  OK")
PYEOF

echo ""
echo "==> Data ready. Next step:"
echo "    bash scripts/train_en_ru.sh"
