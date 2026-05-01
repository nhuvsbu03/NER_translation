# Session 01 — Baseline Runs (Failed)

## Run A: EN→ZH (en_zh)

| Parameter | Value |
|-----------|-------|
| Steps | 50,000 |
| Platform | Google Colab |
| BPE vocab | 10,000 |
| init_pretrained | False |
| lr_anneal_steps | 30,000 |
| diffusion_steps (inference) | 1,000 |
| num_samples (eval) | 50 |

**Results**

| Metric | Score |
|--------|-------|
| SacreBLEU (13a) | 7.47 |
| SacreBLEU (char) | 4.52 |
| Training loss | ~0.07 |

**Sample outputs**: "马斯马斯马斯马斯马斯..." — complete clamping collapse.

---

## Run B: ZH→VI (zh_vi)

| Parameter | Value |
|-----------|-------|
| Steps | 30,000 |
| Platform | Google Colab |
| BPE vocab | 10,000 |
| init_pretrained | False |
| lr_anneal_steps | 30,000 |
| diffusion_steps (inference) | 1,000 |
| num_samples (eval) | 50 |

**Results**

| Metric | Score |
|--------|-------|
| SacreBLEU (char) | 8.79 |
| SacreBLEU (13a) | 4.61 |

**Sample outputs**: "dài/thủy" repeating — same collapse pattern.

---

## Root Cause Analysis

Both runs show the same failure: **low training loss + degenerate repeated output = clamping collapse**.

**Why it happens:**

The diffusion model operates in continuous embedding space. It learns to minimize MSE between predicted and true embeddings. With a 10k BPE vocab for bilingual EN+ZH:
- Chinese characters are encoded as multi-byte sequences → common 2-char combos ("马斯", "dài") appear thousands of times
- Their embeddings dominate the centroid of the embedding space
- Model learns "predict the centroid" → minimizes average MSE ✓
- At inference, rounding maps centroid → most frequent token → every position gets same token

**Contributing factors:**
1. BPE vocab=10k too small → high-frequency byte n-grams cluster at center
2. `init_pretrained=False` → embedding geometry starts random, converges slowly to mean
3. `lr_anneal_steps=30k` too short → LR decays before model learns token-specific regions
4. `num_samples=50` → unreliable BLEU estimate

---

## Dataset Issues Discovered (via EDA, 2026-05-01)

Ran EDA on the training data and found 6,349 bad pairs (2.6%):

| Type | Count | Description |
|------|-------|-------------|
| `empty_zh` | 6 | Empty ZH lines causing index drift |
| `bad_lang_zh` | 1,253 | ZH lines with <15% CJK — doc IDs, codes, English mixed in |
| `extreme_ratio` | 5,090 | Short EN header (2 words) paired with long ZH paragraph, or vice versa |

Also found: `train.en` has 240,192 lines, `train.zh` has 240,213 — 21-line count mismatch.

**Fix**: `train_dataset/clean_dataset.py` — generates `*_clean.en/zh` files with 233,842 clean pairs.

---

## What We Learned

- Small BPE vocab is the primary cause of collapse for multilingual diffusion LMs
- Pretrained init is critical for embedding geometry
- 30k steps is too few (~1.6 epochs over 240k examples)
- Dataset had silent alignment issues that would have confused training
- `num_samples=50` gives BLEU estimates too noisy to compare runs
