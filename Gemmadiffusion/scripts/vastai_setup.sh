#!/bin/bash
set -e
REPO_DIR="/root/Gemmadiffusion"

# Separate venv — keeps DiffusionGemma's modern transformers isolated from
# any existing SeqDiffuSeq environment (which pins transformers==4.18)
python3 -m venv /root/venv_diffgemma
source /root/venv_diffgemma/bin/activate

pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers accelerate bitsandbytes sacrebleu tqdm pillow -q

# Download all test sets via sacrebleu
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

# EN-RU: source=EN, ref=RU — WMT14 newstest2014
save_pair('wmt14', 'en-ru', 'en', 'ru', '$REPO_DIR/data/en-ru')

# ZH-EN: source=ZH, ref=EN — WMT17 newstest2017
save_pair('wmt17', 'zh-en', 'zh', 'en', '$REPO_DIR/data/zh-en')

# EN-ZH: flip of zh-en (English source, Chinese ref)
import shutil as sh
d = pathlib.Path('$REPO_DIR/data/en-zh')
d.mkdir(parents=True, exist_ok=True)
zh_en_src = pathlib.Path('$REPO_DIR/data/zh-en/test.zh')
zh_en_ref = pathlib.Path('$REPO_DIR/data/zh-en/test.en')
sh.copy(zh_en_ref, d / 'test.en')  # English is now the source
sh.copy(zh_en_src, d / 'test.zh')  # Chinese is now the reference
print(f'en-zh (flipped wmt17): {len((d / \"test.en\").read_text().splitlines())} lines')

# JA-EN: source=JA, ref=EN — WMT20 newstest2020
save_pair('wmt20', 'ja-en', 'ja', 'en', '$REPO_DIR/data/ja-en')

# EN-JA: flip of ja-en (English source, Japanese ref)
d = pathlib.Path('$REPO_DIR/data/en-ja')
d.mkdir(parents=True, exist_ok=True)
ja_en_src = pathlib.Path('$REPO_DIR/data/ja-en/test.ja')
ja_en_ref = pathlib.Path('$REPO_DIR/data/ja-en/test.en')
sh.copy(ja_en_ref, d / 'test.en')
sh.copy(ja_en_src, d / 'test.ja')
print(f'en-ja (flipped wmt20): {len((d / \"test.en\").read_text().splitlines())} lines')
"

echo "Setup complete. Activate env with: source /root/venv_diffgemma/bin/activate"
