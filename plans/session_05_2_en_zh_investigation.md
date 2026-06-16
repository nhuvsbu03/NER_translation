# Session 05.2 — EN→ZH Investigation

## Status: Investigation complete, ready to run

---

## What We Found

### Session 02 ran in the WRONG direction

Checkpoints from Session 02 exist at `D:\Learning\Research\scr\tmp2\`:
- `ema_0.9999_020000.pt`, `model020000.pt`, `model055000.pt`, `model100000.pt`
- Eval results: **SacreBLEU (char) = 7.04, SacreBLEU (13a) = 0.43** (very poor)

**Root issue:** The eval CSV (`eval_step100000.csv`) was run with `--pair zh-en`, and its reference column contains **English text** — not Chinese. This means the model's decoder was producing English as its output target, not Chinese.

Two possible causes (need user to verify which):
1. **Data files were swapped on Colab:** `data/en-zh/train.en` contained Chinese and `train.zh` contained English → model was accidentally trained as ZH→EN
2. **Inference was run with `--src zh --tgt en`** instead of `--src en --tgt zh` → direction swapped at inference only

Local data files (`data/en_zh/`) are **correctly ordered**: `.en` = English source, `.zh` = Chinese target. Whatever happened on Colab diverged from this.

**Conclusion: EN→ZH with all Session 02 fixes (32k BPE + in_channel=768 + init_pretrained=True) has never been correctly run.**

Other session history:
- Session 01: 10k BPE + no pretrained init → clamping collapse ("马斯马斯...")
- Session 04 (ZH→EN): encoder-side failure (Chinese → BART encoder → meaningless hidden states → random English output, BLEU=0.01)

---

## Code Audit Results

### 1. Inference uses greedy topk — not KNN rounding (expected)

**File:** `SeqDiffuSeq/inference_main.py:122-123`
```python
logits = model.get_logits(sample)
cands = th.topk(logits, k=1, dim=-1).indices.squeeze()
```

The `rounding.py` KNN approach is dead code for our model — it only works for `mode in ['random', 'random_up_proj', 'glove']` (original Diffusion-LM modes, not our BART translation setup). Greedy topk is correct.

### 2. lm_head starts from BART's English embeddings but IS trained

**File:** `SeqDiffuSeq/src/modeling/predictor/transformer_model.py:149-152`
```python
self.lm_head = nn.Linear(self.embedding_dim, out_size)
with th.no_grad():
    self.lm_head.weight = nn.Parameter(self.input_transformers.shared.weight[:out_size])
```

- Starts from BART's embedding matrix (sliced to 32,005 tokens)
- `freeze_embeddings=False` (default in `args_utils.py`) → lm_head IS trained during backprop
- Initial weights for high-index CJK BPE tokens (e.g., 30k–32k) come from BART's rarely-trained English tokens, but they will be updated during training

### 3. CJK characters are multi-byte BPE subwords — NOT single tokens

```
'的': NOT in vocab directly (multi-byte BPE)
'是': NOT in vocab directly
Total vocab size: 32,005
High-index entries (~32000-32004): garbled UTF-8 Chinese BPE merges
```

ByteLevelBPE encodes "的" as 3 UTF-8 bytes → BPE training may merge these into a single token at high indices. Common Chinese bigrams get dedicated BPE tokens, but individual characters may stay as 3-byte pieces.

**Implication for clamping collapse:** Session 01's collapse ("马斯马斯...") was caused by 10k BPE + no pretrained init → embedding centroid dominated by 2-byte Chinese merges. With 32k BPE, each Chinese character/bigram has a more distinct region → collapse should be fixed.

### 4. ZH→EN collapse characterization (Session 04 outputs)

Sample model outputs vs ground truth:
```
MODEL: "It, of."  GT: "Other names mentioned are not listed."
MODEL: "the the the the..." (×128)  GT: full 200-word paragraph
MODEL: ". 16 16"  GT: long English sentence
MODEL: "It the been,"  GT: long English sentence
```

**Pattern:** Not single-token clamping collapse — random short English fragments. This is encoder-side failure (Chinese input → BART encoder → meaningless hidden states → decoder outputs random English tokens by chance). Completely different failure mode from EN→ZH.

---

## Diagnosing the Direction Confusion

Before re-running, verify what happened in Session 02 by checking the Colab notebook's data prep cell:
- Does it copy `train_clean.en` → `data/en-zh/train.en` (correct) or `train_clean.zh` → `data/en-zh/train.en` (swapped)?
- Does the training cell use `--src en --tgt zh` or `--src zh --tgt en`?

If data was swapped: fix the data prep cell, clear `data/en-zh/`, re-run.
If args were correct: the collapse at inference was likely the lm_head English bias (see Code Audit above).

---

## Proposed Experiment: Run EN→ZH on Google Colab

### What to run (Session 02 config, finally executed)

The existing Colab notebook from Sessions 01–02 already has all the fixes. Update the training args cell:

| Arg | Value | Why |
|-----|-------|-----|
| `--in_channel` | 768 | Match BART-base d_model |
| `--out_channel` | 768 | Match BART-base d_model |
| `--num_channels` | 3072 | 4× in_channel |
| `--num_heads` | 12 | Match BART-base |
| `--vocab_size` | 32005 | 32k BPE + 5 special tokens |
| `--init_pretrained` | True | Load BART weights |
| `--freeze_embeddings` | False | Allow lm_head to train |
| `--sequence_len` | 128 | Fix 23% truncation (was 64 in Session 01) |
| `--sequence_len_src` | 128 | Match |
| `--lr_anneal_steps` | 100000 | Session 02 target |
| `--batch_size` | 64 | Colab GPU (Colab A100 has 40GB but free tier is T4) |
| `--lr` | 1e-4 | Conservative |
| `--save_interval` | 10000 | 10 checkpoints over 100k steps |

Data: use `train_clean.en/zh` (233,842 pairs after cleaning).

### Diagnostic checkpoints

| Step | Loss expected | Action |
|------|--------------|--------|
| 2,500 | < 8.0 | If > 8.0 → data path or tokenizer broken |
| 10,000 | < 1.0 | Run quick inference (5 samples) — check if output is NOT "的的的的..." |
| 25,000 | < 0.5 | Assess sample quality |
| 100,000 | ~0.07 | Full inference on test set |

### Success / failure criteria

**Pass (collapse fixed):** At step 10k, outputs contain multiple distinct Chinese characters
**Fail (collapse persists):** At step 10k, outputs are still clamped to one repeating token

---

## If Collapse Persists at Step 10k

Check in this order:

1. **Tokenizer sanity check:** `tokenizer.encode("北京").ids` — should be 3-6 tokens, NOT 1 token repeated
2. **Embedding geometry check:** At step 0, print the 5 nearest vocab tokens to the Chinese embedding centroid — if they're all the same token, collapse will happen
3. **lm_head initialization:** Print `model.lm_head.weight[25000:25005].norm(dim=-1)` — if all near zero, the high-index CJK embeddings are uninitialized
4. **Consider:** Random-init lm_head for the 32k BPE vocab (don't copy from BART) — removes English embedding bias for CJK output tokens

---

## Git Branch

All code changes for this investigation: **`investigate/en-zh`** (already created from main)

After running:
- If EN→ZH works at 100k steps → merge investigate/en-zh to main, document in session_05_2 results
- If EN→ZH still fails → add specific fix attempt as a new commit on investigate/en-zh

---

## Files Changed / Created

| File | Status | Notes |
|------|--------|-------|
| `plans/session_05_2_en_zh_investigation.md` | New | This document |
| Colab notebook | Update in-session | Update training args cell only |
| `scripts/data_en_zh.sh` | May need to create | If re-running data prep on vast.ai later |
