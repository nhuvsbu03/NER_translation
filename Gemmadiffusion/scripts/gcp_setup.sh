#!/bin/bash
# Run ON the GCP instance after gcp_push.sh. Mirrors vastai_setup.sh.
# Usage (from local): gcloud compute ssh <instance> --zone <zone> \
#     --command="bash /root/Gemmadiffusion/scripts/gcp_setup.sh"
set -e
REPO_DIR="/root/Gemmadiffusion"

python3 -m venv /root/venv_diffgemma
source /root/venv_diffgemma/bin/activate

pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers accelerate bitsandbytes sacrebleu tqdm pillow -q

echo "==> GPU check:"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

# Download all 5 test sets via sacrebleu
python3 -c "
import sacrebleu, pathlib, shutil

def save_pair(wmt, pair, src_lang, tgt_lang, out_dir):
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    src = sacrebleu.get_source_file(wmt, pair)
    ref = sacrebleu.get_reference_files(wmt, pair)[0]
    shutil.copy(src, d / f'test.{src_lang}')
    shutil.copy(ref, d / f'test.{tgt_lang}')
    n = len((d / f'test.{src_lang}').read_text().splitlines())
    print(f'{pair} ({wmt}): {n} lines')

save_pair('wmt14', 'en-ru', 'en', 'ru', '$REPO_DIR/data/en-ru')
save_pair('wmt17', 'zh-en', 'zh', 'en', '$REPO_DIR/data/zh-en')

import shutil as sh
d = pathlib.Path('$REPO_DIR/data/en-zh')
d.mkdir(parents=True, exist_ok=True)
zh_en_src = pathlib.Path('$REPO_DIR/data/zh-en/test.zh')
zh_en_ref = pathlib.Path('$REPO_DIR/data/zh-en/test.en')
sh.copy(zh_en_ref, d / 'test.en')
sh.copy(zh_en_src, d / 'test.zh')
print(f'en-zh (flipped wmt17): {len((d / \"test.en\").read_text().splitlines())} lines')

save_pair('wmt20', 'ja-en', 'ja', 'en', '$REPO_DIR/data/ja-en')

d = pathlib.Path('$REPO_DIR/data/en-ja')
d.mkdir(parents=True, exist_ok=True)
ja_en_src = pathlib.Path('$REPO_DIR/data/ja-en/test.ja')
ja_en_ref = pathlib.Path('$REPO_DIR/data/ja-en/test.en')
sh.copy(ja_en_ref, d / 'test.en')
sh.copy(ja_en_src, d / 'test.ja')
print(f'en-ja (flipped wmt20): {len((d / \"test.en\").read_text().splitlines())} lines')
"

echo ""
echo "==> Add your HF token (needed before any infer_*.sh run):"
echo "    gcloud compute ssh $INSTANCE_NAME --zone $ZONE --command='echo YOUR_HF_TOKEN > ~/.hf_token'"
echo "    (accept the Gemma license on both google/diffusiongemma-26B-A4B-it"
echo "     and GoedelMachines/diffusiongemma-26B-A4B-w4a16 on huggingface.co first)"
echo ""
echo "==> Setup complete. Run inference with, e.g.:"
echo "    gcloud compute ssh \$INSTANCE_NAME --zone \$ZONE --command='bash /root/Gemmadiffusion/scripts/infer_w4a16.sh en-ru 5'"
