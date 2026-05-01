# Plan: Retrain en_zh SeqDiffuSeq on vast.ai (Fresh Start)

## Context

User is training a SeqDiffuSeq (arXiv 2212.10325) NER-aware English→Chinese translation model in Google Colab using the notebook `zh_vi_translation.ipynb`.

**Problem**: Both previous runs (en_zh @50k steps, zh_vi @30k steps) produce degenerate outputs — the model collapses to repeating a single high-frequency token ("马斯" for ZH, "dài/thủy" for VI) despite training loss being very small (~0.07). BLEU is 4–8 on only 50 test samples.

**Root cause diagnosis**: Low loss + degenerate output is the signature of a **clamping collapse** in diffusion-based LMs:
- The diffusion model denoises continuous embeddings well (low MSE loss)
- But at inference, when the clamping function projects each denoised embedding to the nearest vocabulary token, most embeddings map to the same high-frequency centroid token
- This happens when the embedding space is not well-separated — caused by:
  1. **Byte-level BPE with only 10k vocab is too small** for a bilingual EN+ZH corpus. Chinese characters become 3-byte sequences, creating high-frequency 2-character tokens like "马斯" that dominate the vocabulary centroid
  2. **No pretrained initialization** — the transformer starts from random weights, making embedding geometry slow to converge
  3. **lr_anneal_steps may have been too short** (the saved args don't show what was used in the 50k run — but loss plateau at ~0.07 with poor BLEU suggests premature LR decay)

User confirmed: wants to **start fresh** with fixes to tokenizer and training.

---

## What to Change in the Notebook

### Phase 3 — Tokenizer Fix (most impactful change)

**Current:**
```python
subprocess.run([VENV_PYTHON, "tokenizer_utils.py", "train-byte-level", "en_zh", "10000"], ...)
```

**Change to vocab_size=32000:**
```python
subprocess.run([VENV_PYTHON, "tokenizer_utils.py", "train-byte-level", "en_zh", "32000"], ...)
```

Why: 32k vocab is the standard for multilingual NMT (Helsinki-NLP, mBART, etc.). With 10k, Chinese characters are under-tokenized and high-frequency byte-n-grams dominate the embedding space, causing collapse. Larger vocab → better-separated token embeddings → cleaner clamping.

If the SeqDiffuSeq tokenizer code also accepts `--vocab_size` as a flag, also update the `--vocab_size` arg in the training command from `10005` → `32005`.

### Phase 4 — Training Command Changes

**Change 1: Increase lr_anneal_steps (already set to 100000 in notebook — confirm it's correct)**
```python
"--lr_anneal_steps", "100000",
```
This ensures the LR ramps down slowly over the full 100k steps instead of cutting off early.

**Change 2: Enable pretrained transformer initialization**
```python
"--init_pretrained", "True",   # was False
```
The model architecture (6 layers, 8 heads, 512/2048 dim) matches facebook/bart-base which is already cached at `/content/drive/MyDrive/BUyen_Qnhu++/src/SeqDiffuSeq/pretrained/bart-base`. This initializes the attention and FFN weights from BART — significantly better geometry from step 0.

**Change 3: Fix the data path inconsistency**

The notebook copies data to `data/en_zh/` (underscore) but training references `./data/en-zh/` (hyphen). One of these must be wrong. 

Determine which path the SeqDiffuSeq `main.py` actually reads, then make the copy destination match:
- If code reads `en-zh`: change `DST_DATA_DIR = os.path.join(REPO_DIR, "data", "en-zh")`
- If code reads `en_zh`: change training args to `"--train_txt_path", "./data/en_zh/train"`

**Change 4: Update vocab_size in training args**
```python
"--vocab_size", "32005",   # was 10005 (32000 BPE tokens + 5 special tokens)
```

**Keep unchanged:**
- Architecture (6+6 layers, 8 heads, 512/2048 dim, dropout 0.3)
- Diffusion config (1000 steps, sqrt schedule, predict_xstart=True)
- Batch size 16, LR 1e-4, warmup 500
- sequence_len 64 / sequence_len_src 128
- EMA rate 0.9999

### Phase 5 — Inference Changes (after training)

**Add DDIM + fewer steps:**
```python
"--use_ddim",        "True",    # was False
"--diffusion_steps", "200",     # was 1000 (DDIM needs fewer steps)
```
DDIM (deterministic denoising) produces higher quality outputs than DDPM for text generation. 200 steps is typically enough and is 5× faster.

**Evaluate on more samples:**
Change `num_samples` from 50 → 200 for a more reliable BLEU estimate.

---

## vast.ai Notebook Adaptation (additional changes on top of training fixes)

Instance rented: ID `35960125`, RTX 4090 @ $0.30/hr, `188.36.196.221:5747`
SSH key: `~/.ssh/vastai_key`

### Notebook cells to adapt for vast.ai

**Cell `6d94af1e`** (Drive mount) ✅ Done — replaced with `print("Running on vast.ai")`

**Cell `aa158b95`** (path variables) — change to local paths:
```python
REPO_DIR     = "/root/SeqDiffuSeq"
DRIVE_DATA   = "/root/train_dataset"          # rsync'd from local Mac
DST_DATA_DIR = os.path.join(REPO_DIR, "data", "en_zh")
CKPT_DIR     = os.path.join(REPO_DIR, "ckpts", "en_zh")
OUT_DIR      = os.path.join(CKPT_DIR, "inference_out")
VENV_PYTHON  = "/root/SeqDiffuSeq/venv/bin/python"
```

**Cell `96c7c840`** (pip install PyTorch) — skip old torch==1.11.0; vast.ai image already has PyTorch 2.1.0+cu121. Only install the NLP deps:
```python
# vast.ai already has PyTorch 2.1.0 — skip torch install, just add NLP deps
import subprocess
result = subprocess.run([
    "pip", "install", "-q",
    "bert-score", "blobfile", "datasets", "huggingface-hub==0.4.0",
    "mpi4py", "nltk", "pandas", "protobuf", "rouge-score", "sacrebleu",
    "sacremoses", "scikit-learn", "scipy", "spacy", "tokenizers",
    "torchmetrics", "tqdm", "transformers==4.18.0"
], capture_output=True, text=True)
print(result.stdout[-500:] or "done")
```

**Cell `e75b12a0`** (check data files) — update DRIVE_DATA path check to `/root/train_dataset`

**Cell `99ba7f75`** (copy data) — data already copied via rsync, just verify and skip copy if exists

### Step-by-Step Execution Plan

1. ✅ Cell `6d94af1e` already updated
2. Update cells `aa158b95`, `96c7c840`, `e75b12a0`, `99ba7f75` (see above)
3. Upload notebook + dataset to instance:
   ```bash
   # Upload dataset
   rsync -avz --progress \
     "/Users/holly.nguyen/Documents/My Research/NER_translation/train_dataset/" \
     -e "ssh -i ~/.ssh/vastai_key -p 5747" \
     root@188.36.196.221:~/train_dataset/

   # Upload notebook
   scp -i ~/.ssh/vastai_key -P 5747 \
     "/Users/holly.nguyen/Documents/My Research/NER_translation/zh_vi_translation.ipynb" \
     root@188.36.196.221:~/
   ```
4. On the instance: clone SeqDiffuSeq repo, start JupyterLab
5. Open Jupyter via SSH tunnel and run notebook phases 1→6

---

## Critical Files to Modify

- Notebook: [NER_translation/zh_vi_translation.ipynb](NER_translation/zh_vi_translation.ipynb)
  - Cell `c5d1ff36` (Phase 3): Change `"10000"` → `"32000"`
  - Cell `23c64bfb` (Phase 4 training): Change `init_pretrained → True`, `vocab_size → 32005`, fix data path
  - Cell `607671b2` (Phase 5 inference): Add `use_ddim=True`, `diffusion_steps=200`, increase `num_samples`
  - Cell `aa158b95` (variable setup): Fix `DST_DATA_DIR` path to match training script expectation

---

## Verification

After the first 5k steps: check `progress.csv` — if loss is still above 5.0, something is wrong with the data path or tokenizer.

After 25k steps: run inference. If outputs are no longer "马斯马斯马斯...", the tokenizer fix worked. Expect partially coherent Chinese output.

After 50k–100k steps: target BLEU (13a) > 15 (current best was 7.47 with 50 samples at 50k steps with broken tokenizer).
