#!/usr/bin/env bash
# Download WMT17 ZH-EN from HuggingFace, save as plain text, train 32k BPE tokenizer.
# Run once for the ZH-EN experiment (from /root/NER_translation).
# Usage: bash scripts/data_zh_en.sh
set -euo pipefail

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
RAW_DIR="$PROJECT_ROOT/train_dataset/wmt17_zh_en"
DATA_DIR="$REPO_DIR/data/zh-en"

mkdir -p "$RAW_DIR" "$DATA_DIR"

# ── Step 1: Download WMT17 ZH-EN ─────────────────────────────────────────────
echo "==> Downloading WMT17 zh-en from HuggingFace datasets..."
echo "    (train ~2.1M pairs, valid ~2k pairs — takes a few minutes)"

python3 - <<PYEOF
import os
from datasets import load_dataset

raw_dir = "$RAW_DIR"
ds = load_dataset("wmt/wmt17", "zh-en", cache_dir="/root/.cache/hf", trust_remote_code=True)

split_map = {"train": "train", "valid": "validation", "test": "test"}
for name, hf_split in split_map.items():
    zh_path = os.path.join(raw_dir, f"{name}.zh")
    en_path = os.path.join(raw_dir, f"{name}.en")
    if os.path.exists(zh_path) and os.path.exists(en_path):
        n = sum(1 for _ in open(zh_path, encoding="utf-8"))
        print(f"  Exists  {name}: {n:,} pairs — skipping")
        continue
    count = 0
    with open(zh_path, "w", encoding="utf-8") as f_zh, \
         open(en_path, "w", encoding="utf-8") as f_en:
        for ex in ds[hf_split]:
            f_zh.write(ex["translation"]["zh"].strip() + "\n")
            f_en.write(ex["translation"]["en"].strip() + "\n")
            count += 1
    print(f"  Wrote   {name}: {count:,} pairs")

print("Download complete.")
PYEOF

# ── Step 2: Copy to SeqDiffuSeq/data/zh-en/ ──────────────────────────────────
echo "==> Copying data to $DATA_DIR..."
for split in train valid test; do
    for lang in zh en; do
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
echo "==> Training 32k BPE tokenizer on ZH+EN train corpus..."
TOK_VOCAB="$DATA_DIR/vocab.json"
if [ -f "$TOK_VOCAB" ]; then
    echo "    Tokenizer already exists — skipping. Delete $TOK_VOCAB to retrain."
else
    cd "$REPO_DIR"
    python3 tokenizer_utils.py train-byte-level zh-en 32000
    echo "    Tokenizer saved to $DATA_DIR"
fi

# ── Step 4: Sanity check ─────────────────────────────────────────────────────
echo "==> Sanity check: tokenize '你好' (should be ≤ 4 tokens)..."
cd "$REPO_DIR"
python3 - <<PYEOF
from tokenizer_utils import read_byte_level
tok = read_byte_level("./data/zh-en")
ids = tok.encode("你好").ids
print(f"  '你好' → {len(ids)} tokens: {ids}")
if len(ids) > 4:
    print("  WARN: more tokens than expected — check vocab coverage")
else:
    print("  OK")
PYEOF

echo ""
echo "==> Data ready. Next step:"
echo "    bash scripts/train_zh_en.sh"
