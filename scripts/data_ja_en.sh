#!/usr/bin/env bash
# Download WMT20 JA-EN from HuggingFace, save as plain text, train 32k BPE tokenizer.
# Run once for the JA→EN experiment (from /root/NER_translation).
# Usage: bash scripts/data_ja_en.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/NER_translation}"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
RAW_DIR="$PROJECT_ROOT/train_dataset/wmt20_ja_en"
DATA_DIR="$REPO_DIR/data/ja-en"

mkdir -p "$RAW_DIR" "$DATA_DIR"

# ── Step 1: Download WMT20 JA-EN ─────────────────────────────────────────────
echo "==> Downloading WMT20 ja-en from HuggingFace datasets..."
echo "    (~3.6M train pairs — download ~1GB, takes ~5-10 min)"

python3 - <<PYEOF
import os
from datasets import load_dataset

raw_dir = "$RAW_DIR"
ds = load_dataset("wmt20", "ja-en", cache_dir="/root/.cache/hf")
print(f"  Dataset splits: {list(ds.keys())}")

split_map = {"train": "train", "valid": "validation", "test": "test"}
for name, hf_split in split_map.items():
    if hf_split not in ds:
        print(f"  SKIP {name}: split '{hf_split}' not found in dataset")
        continue

    ja_path = os.path.join(raw_dir, f"{name}.ja")
    en_path = os.path.join(raw_dir, f"{name}.en")

    if os.path.exists(ja_path) and os.path.exists(en_path):
        n_ja = sum(1 for _ in open(ja_path, encoding="utf-8"))
        n_en = sum(1 for _ in open(en_path, encoding="utf-8"))
        if n_ja == n_en:
            print(f"  Exists  {name}: {n_ja:,} pairs — skipping")
            continue
        else:
            print(f"  MISMATCH {name}: ja={n_ja:,} en={n_en:,} — re-downloading")
            os.remove(ja_path); os.remove(en_path)

    count = 0
    with open(ja_path, "w", encoding="utf-8") as f_ja, \
         open(en_path, "w", encoding="utf-8") as f_en:
        for ex in ds[hf_split]:
            ja_line = ex["translation"]["ja"].replace("\n", " ").strip()
            en_line = ex["translation"]["en"].replace("\n", " ").strip()
            f_ja.write(ja_line + "\n")
            f_en.write(en_line + "\n")
            count += 1
    print(f"  Wrote   {name}: {count:,} pairs")

print("Download complete.")
PYEOF

# ── Step 2: Copy to SeqDiffuSeq/data/ja-en/ ──────────────────────────────────
echo "==> Copying data to $DATA_DIR..."
for split in train valid test; do
    for lang in ja en; do
        src="$RAW_DIR/$split.$lang"
        dst="$DATA_DIR/$split.$lang"
        if [ ! -f "$dst" ]; then
            if [ -f "$src" ]; then
                cp "$src" "$dst"
                n=$(wc -l < "$dst")
                printf "  Copied  %s.%s  (%s lines)\n" "$split" "$lang" "$n"
            else
                printf "  SKIP    %s.%s  (not downloaded — check split names above)\n" "$split" "$lang"
            fi
        else
            printf "  Exists  %s.%s\n" "$split" "$lang"
        fi
    done
done

# ── Step 3: Train 32k BPE tokenizer ─────────────────────────────────────────
echo "==> Training 32k BPE tokenizer on JA+EN train corpus..."
TOK_VOCAB="$DATA_DIR/vocab.json"
if [ -f "$TOK_VOCAB" ]; then
    echo "    Tokenizer already exists — skipping. Delete $TOK_VOCAB to retrain."
else
    python3 - <<PYEOF
from tokenizers import ByteLevelBPETokenizer
import os

data_dir = "$DATA_DIR"
files = [os.path.join(data_dir, "train.ja"), os.path.join(data_dir, "train.en")]
print(f"  Training ByteLevelBPE on {[os.path.basename(f) for f in files]} ...")

tok = ByteLevelBPETokenizer()
tok.train(
    files=files,
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)
tok.save_model(data_dir)
import json
actual_size = len(json.load(open(os.path.join(data_dir, "vocab.json"))))
print(f"  Vocab size: {actual_size} tokens")
print(f"  Saved vocab.json + merges.txt to {data_dir}")
PYEOF
    echo "    Tokenizer saved to $DATA_DIR"
fi

# ── Step 4: Sanity check ─────────────────────────────────────────────────────
echo "==> Sanity check: tokenize '東京' (should be ≤ 6 tokens)..."
python3 - <<PYEOF
from tokenizers import ByteLevelBPETokenizer
tok = ByteLevelBPETokenizer("$DATA_DIR/vocab.json", "$DATA_DIR/merges.txt")
ids = tok.encode("東京").ids
print(f"  '東京' → {len(ids)} tokens: {ids}")
print("  OK" if len(ids) <= 6 else "  WARN: more tokens than expected")
PYEOF

echo ""
echo "==> Data ready. Next step:"
echo "    bash scripts/train_ja_en.sh"
