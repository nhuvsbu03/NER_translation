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
    # Use huggingface_hub.snapshot_download (works with current HF API regardless of
    # transformers version — the old transformers 4.18.0 from_pretrained has broken URLs)
    PRETRAINED_DIR="$PRETRAINED" python3 - <<'PYEOF'
import os
pretrained = os.environ["PRETRAINED_DIR"]
from huggingface_hub import snapshot_download
print("    Downloading via huggingface_hub.snapshot_download ...")
snapshot_download(
    repo_id="facebook/bart-base",
    local_dir=pretrained,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)
# snapshot_download saves as safetensors; SeqDiffuSeq needs pytorch_model.bin
# Convert if needed
import os, glob
if not os.path.exists(os.path.join(pretrained, "pytorch_model.bin")):
    st_files = glob.glob(os.path.join(pretrained, "model.safetensors"))
    if st_files:
        print("    Converting safetensors → pytorch_model.bin ...")
        from safetensors.torch import load_file
        import torch
        state_dict = load_file(st_files[0])
        torch.save(state_dict, os.path.join(pretrained, "pytorch_model.bin"))
        print("    Conversion done.")
print("    Saved to", pretrained)
PYEOF
fi

echo ""
echo "==> Setup complete. Next step:"
echo "    bash scripts/data_en_ru.sh"
