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

echo "==> Checking PyTorch CUDA compatibility..."
CUDA_OK=$(python3 -c "import torch; print('ok' if torch.cuda.is_available() else 'no_cuda')" 2>/dev/null || echo "import_fail")
if [ "$CUDA_OK" != "ok" ]; then
    echo "    torch.cuda.is_available()=False — installing torch matched to host CUDA driver..."
    CUDA_VER=$(python3 -c "import subprocess,re; o=subprocess.check_output(['nvidia-smi'],text=True); m=re.search(r'CUDA Version: (\d+)\.(\d+)',o); print(f'cu{m.group(1)}{m.group(2).zfill(1)}') if m else print('cu124')" 2>/dev/null || echo "cu124")
    echo "    Detected CUDA: $CUDA_VER"
    pip install -q torch --index-url "https://download.pytorch.org/whl/$CUDA_VER"
else
    echo "    torch.cuda.is_available()=True — skipping torch reinstall."
fi
echo "    Done."

echo "==> Installing Python dependencies..."
pip install -q \
    bert-score blobfile "datasets>=2.20" \
    "huggingface-hub>=0.20" \
    nltk numpy pandas protobuf \
    rouge-score sacrebleu sacremoses \
    scikit-learn scipy spacy \
    "tokenizers>=0.19" torchmetrics tqdm \
    "transformers>=4.35,<4.45"
echo "    Done."

echo "==> Building mpi4py against HPC-X MPI (fixes MPI_Init_thread on datacenter A100s)..."
# Datacenter nodes ship HPC-X MPI at /usr/local/mpi; pip wheel links against wrong lib.
# Rebuild from source using the correct mpicc so MPI.COMM_WORLD works without mpirun.
MPICC_BIN=$(which mpicc 2>/dev/null || echo /usr/local/mpi/bin/mpicc)
pip uninstall mpi4py -y -q 2>/dev/null || true
MPICC="$MPICC_BIN" pip install -q --no-binary mpi4py mpi4py
echo "    Done."

echo ""
echo "==> Setup complete. Next step:"
echo "    bash scripts/data_en_ru.sh"
