#!/usr/bin/env bash
# Run once per new vast.ai instance.
# Installs Python dependencies and downloads BART-base weights.
# Usage (from /root/NER_translation): bash scripts/vastai_setup.sh
set -euo pipefail

PROJECT_ROOT="/root/NER_translation"
REPO_DIR="$PROJECT_ROOT/SeqDiffuSeq"
PRETRAINED="$REPO_DIR/pretrained/bart-base"

echo "==> Downloading facebook/bart-base weights (using conda transformers before downgrade)..."
mkdir -p "$PRETRAINED"

if [ -f "$PRETRAINED/pytorch_model.bin" ]; then
    echo "    Already cached — skipping download."
else
    PRETRAINED_DIR="$PRETRAINED" python3 - <<'PYEOF'
import os, shutil, glob
pretrained = os.environ["PRETRAINED_DIR"]

# Use snapshot_download — returns cache path (works on all huggingface_hub versions)
from huggingface_hub import snapshot_download
print("    Downloading bart-base snapshot...")
snap = snapshot_download(
    repo_id="facebook/bart-base",
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)
print(f"    Snapshot at: {snap}")

# Copy files to our pretrained dir
for f in os.listdir(snap):
    src = os.path.join(snap, f)
    dst = os.path.join(pretrained, f)
    if os.path.isfile(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"    Copied {f}")

# snapshot_download may give safetensors; SeqDiffuSeq needs pytorch_model.bin
if not os.path.exists(os.path.join(pretrained, "pytorch_model.bin")):
    st = glob.glob(os.path.join(pretrained, "model.safetensors"))
    if st:
        print("    Converting safetensors → pytorch_model.bin ...")
        from safetensors.torch import load_file
        import torch
        torch.save(load_file(st[0]), os.path.join(pretrained, "pytorch_model.bin"))
        print("    Conversion done.")

print("    Saved to", pretrained)
PYEOF
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
