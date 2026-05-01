# Project Overview

## Research Goal

Train a **SeqDiffuSeq** (arXiv 2212.10325) diffusion-based NER-aware machine translation model for **English → Chinese (EN→ZH)**.

The NER-aware component means the model must correctly translate and preserve named entities (people, locations, organizations) tagged in the format `[entity:TYPE]`.

---

## Model: SeqDiffuSeq

- **Type**: Diffusion language model operating on continuous token embeddings
- **Architecture**: Transformer encoder-decoder (6 layers, 8 heads, in/out=512, hidden=2048)
- **Diffusion**: 1000 steps, √-noise schedule, predict_xstart=True, EMA rate=0.9999
- **Matches**: facebook/bart-base architecture (used for pretrained init)
- **Paper**: https://arxiv.org/abs/2212.10325

### How inference works (step by step)

1. Start with random Gaussian noise of shape `(seq_len=64, embedding_dim=512)`
2. Run 1000 DDPM denoising steps — transformer predicts clean `x_0` at each step
3. **Rounding**: map continuous `x_0` to nearest vocabulary token via logit argmax
4. Filter special tokens, decode with BPE tokenizer

### The clamping collapse problem

If the embedding space is poorly separated, step 3 maps most positions to the same high-frequency token. Symptoms: low training loss (~0.07) but output = "马斯马斯马斯...". Root causes:
- BPE vocab too small (10k) → Chinese chars share high-frequency byte n-grams → embeddings cluster at centroid
- No pretrained init → model learns to predict statistical mean → rounding hits same token every time

---

## Dataset: EN→ZH parallel corpus

- **Source**: `train_dataset/` — 240k sentence pairs, ~52% with NER tags
- **NER format**: `[entity:TYPE]` inline in text, types = LOC / ORG / PER
- **Clean version**: `train_clean.*` — 233,842 pairs after removing bad pairs (see `session_01_baseline.md`)
- **Sequence lengths**: EN median=21 words (seq_len_src=128 covers 99%); ZH median=43 chars (seq_len=64 truncates 23%)

---

## Key files

| File | Purpose |
|------|---------|
| `zh_vi_translation.ipynb` | Main training + inference notebook (adapted for vast.ai) |
| `train_dataset/clean_dataset.py` | Regenerates `*_clean.*` files from the originals |
| `SeqDiffuSeq/` | Model source code |
| `SeqDiffuSeq/pretrained/bart-base/` | BART tokenizer + config (no weights — loaded at runtime) |
| `plans/` | This folder |

---

## Infrastructure

- **Training**: vast.ai GPU rental (RTX 3090 ~$0.14/hr, RTX 4090 ~$0.32/hr)
- **SSH key**: `~/.ssh/vastai_key`
- **Setup guide**: `SETUP.md` in repo root
- **Upload**: `rsync` SeqDiffuSeq + train_dataset from local Mac → instance
