#!/usr/bin/env bash
# Download WMT17 ZH-EN from HuggingFace, save as plain text, train 32k BPE tokenizer.
# Run once for the ZH-EN experiment (from /root/NER_translation).
# Usage: bash scripts/data_zh_en.sh
set -euo pipefail

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
RAW_DIR="$PROJECT_ROOT/train_dataset/opus100_zh_en"
DATA_DIR="$REPO_DIR/data/zh-en"

mkdir -p "$RAW_DIR" "$DATA_DIR"

# ── Step 1: Download WMT17 ZH-EN ─────────────────────────────────────────────
echo "==> Downloading OPUS-100 zh-en from HuggingFace datasets..."
echo "    (~1M train pairs, 2k valid, 2k test — download ~150MB, takes ~2-3 min)"

python3 - <<PYEOF
import os
from datasets import load_dataset

raw_dir = "$RAW_DIR"
ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", cache_dir="/root/.cache/hf")

split_map = {"train": "train", "valid": "validation", "test": "test"}
for name, hf_split in split_map.items():
    zh_path = os.path.join(raw_dir, f"{name}.zh")
    en_path = os.path.join(raw_dir, f"{name}.en")

    # Validate existing files: line counts must match
    if os.path.exists(zh_path) and os.path.exists(en_path):
        n_zh = sum(1 for _ in open(zh_path, encoding="utf-8"))
        n_en = sum(1 for _ in open(en_path, encoding="utf-8"))
        if n_zh == n_en:
            print(f"  Exists  {name}: {n_zh:,} pairs — skipping")
            continue
        else:
            print(f"  MISMATCH {name}: zh={n_zh:,} en={n_en:,} — re-downloading")
            os.remove(zh_path)
            os.remove(en_path)

    count = 0
    with open(zh_path, "w", encoding="utf-8") as f_zh, \
         open(en_path, "w", encoding="utf-8") as f_en:
        for ex in ds[hf_split]:
            zh_line = ex["translation"]["zh"].replace("\n", " ").strip()
            en_line = ex["translation"]["en"].replace("\n", " ").strip()
            f_zh.write(zh_line + "\n")
            f_en.write(en_line + "\n")
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
# Train directly via tokenizers library (bypasses tokenizer_utils.py which
# imports transformers, which conflicts with the current huggingface_hub version)
echo "==> Training 32k BPE tokenizer on ZH+EN train corpus..."
TOK_VOCAB="$DATA_DIR/vocab.json"
if [ -f "$TOK_VOCAB" ]; then
    echo "    Tokenizer already exists — skipping. Delete $TOK_VOCAB to retrain."
else
    python3 - <<PYEOF
from tokenizers import ByteLevelBPETokenizer
import os

data_dir = "$DATA_DIR"
files = [os.path.join(data_dir, "train.zh"), os.path.join(data_dir, "train.en")]
print(f"  Training ByteLevelBPE on {[os.path.basename(f) for f in files]} ...")

tok = ByteLevelBPETokenizer()
tok.train(
    files=files,
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)
tok.save_model(data_dir)
# vocab_size=32000 + 5 special tokens = 32005 (matches --vocab_size 32005)
vocab_path = os.path.join(data_dir, "vocab.json")
import json
actual_size = len(json.load(open(vocab_path)))
print(f"  Vocab size: {actual_size} tokens")
print(f"  Saved vocab.json + merges.txt to {data_dir}")
PYEOF
    echo "    Tokenizer saved to $DATA_DIR"
fi

# ── Step 4: Sanity check ─────────────────────────────────────────────────────
echo "==> Sanity check: tokenize '你好' (should be ≤ 4 tokens)..."
python3 - <<PYEOF
from tokenizers import ByteLevelBPETokenizer
tok = ByteLevelBPETokenizer("$DATA_DIR/vocab.json", "$DATA_DIR/merges.txt")
ids = tok.encode("你好").ids
print(f"  '你好' → {len(ids)} tokens: {ids}")
print("  OK" if len(ids) <= 4 else "  WARN: more tokens than expected")
PYEOF

echo ""
echo "==> Data ready. Next step:"
echo "    bash scripts/train_zh_en.sh"
