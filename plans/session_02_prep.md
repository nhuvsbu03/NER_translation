# Session 02 — Prep (Upcoming)

## Goal

Verify that the clamping collapse is fixed. Outputs should not repeat a single token.
Target: BLEU (13a) > 15 on full test set (29,253 pairs).

---

## Changes from Session 01

### Data
| Change | Before | After | Why |
|--------|--------|-------|-----|
| Dataset | `train.en/zh` (240,192 pairs) | `train_clean.en/zh` (233,842 pairs) | Remove 6,349 bad pairs (empty, wrong language, extreme length mismatch) |
| Inference eval | 50 samples | Full test set (29,253) | Reliable BLEU estimate |

### Tokenizer
| Change | Before | After | Why |
|--------|--------|-------|-----|
| BPE vocab size | 10,000 | **32,000** | Primary fix for clamping collapse — each Chinese char gets its own token |
| model `vocab_size` | 10,005 | **32,005** | Matches 32k BPE + 5 special tokens |

### Training
| Change | Before | After | Why |
|--------|--------|-------|-----|
| `init_pretrained` | False | **True** | Load BART-base transformer weights → better embedding geometry from step 0 |
| `lr_anneal_steps` | 30,000 | **100,000** | Full LR schedule across 100k steps — prevents premature decay |
| `sequence_len` | 64 | 64 | Note: 23% of ZH sentences >64 chars — consider 128 in session 03 |

### Inference
| Change | Before | After | Why |
|--------|--------|-------|-----|
| `diffusion_steps` | 1,000 | **200** | Faster; note: DDIM removed from codebase so still DDPM |
| `use_ddim` | False | True | Flag present but may be no-op — verify from log |
| `num_samples` | 50 | **Full test set** | Dynamic: reads `test.en` line count at runtime |

### Infrastructure
| Change | Before | After |
|--------|--------|-------|
| Platform | Google Colab | **vast.ai** (RTX 3090 or 4090) |
| Data source | Google Drive | Local Mac → rsync |

---

## Notebook Cells Changed

| Cell ID | What changed |
|---------|-------------|
| `6d94af1e` | Drive mount → `print("Running on vast.ai")` |
| `aa158b95` | All paths → `/root/...`, `VENV_PYTHON = "python3"` |
| `82672549` | Skip venv creation |
| `d984b02e` | Skip venv pip upgrade |
| `96c7c840` | Skip torch install, pip install NLP deps only |
| `a37eff9b` | Added `import shutil` (was missing, caused NameError in data copy cell) |
| `e75b12a0` | Check for `*_clean.*` files |
| `99ba7f75` | Copy `*_clean.*` → original names in repo |
| `49ebe56d` | `DST_DATA_DIR` → `data/en-zh`, `CKPT_DIR` → `ckpts/en-zh` (was `zh_vi`, mismatched training args) |
| `c5d1ff36` | BPE vocab 32k; train on local `/content/tok_tmp/` SSD (not Drive) directly in notebook kernel; `min_frequency=2` — was 15+ min on Drive, now ~1–2 min |
| `1d186a49` | Added `BartModel` weights download with `safe_serialization=False` to force `pytorch_model.bin`; cache check changed to `pytorch_model.bin` |
| `23c64bfb` | `init_pretrained=True`, `vocab_size=32005`, `lr_anneal_steps=100000` |
| `da5c31a0` | **New cell** — Phase 5 Inference: auto-finds latest EMA checkpoint + time schedule, `diffusion_steps=200`, `num_samples=-1` (full test set) |

---

## Milestones to Check

| Step | Expected loss | Action |
|------|--------------|--------|
| 2,500 | < 8.0 | If still >8, data path or tokenizer is broken |
| 25,000 | < 1.0 | Run inference — outputs should NOT be "马斯马斯..." |
| 50,000 | < 0.3 | Check sample quality — partial coherence expected |
| 100,000 | ~0.07 | Full inference on test set, compute BLEU |

---

## Results (fill in after run)

| Metric | Value |
|--------|-------|
| Steps completed | |
| Final training loss | |
| SacreBLEU (13a) | |
| SacreBLEU (char) | |
| Eval samples | |
| Collapse observed? | |
| Sample output (good) | |
| Sample output (bad) | |

---

## Issues Encountered

### Bugs fixed before training started

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `NameError: shutil not defined` | `import shutil` missing from imports cell | Added to cell `a37eff9b` |
| `FileNotFoundError: ./data/en_zh/` | Tokenizer called with `en_zh` (underscore) but data dir is `en-zh` (hyphen); `DST_DATA_DIR` was `zh_vi` | Unified all paths to `en-zh` in `DST_DATA_DIR`, `CKPT_DIR`, and tokenizer call |
| Tokenizer took 15+ min | Tokenizer training read 467k lines directly from Google Drive (high latency) | Train on local `/content/tok_tmp/` SSD; run in notebook kernel not subprocess; `min_frequency=2` |
| `pytorch_model.bin not found` (Flax weights present) | BART download cell only saved config + tokenizer, not weights; newer transformers saves `model.safetensors` by default | Added `BartModel.from_pretrained(...).save_pretrained(..., safe_serialization=False)` |
| `AttributeError: no attribute embedding_dim` | `embedding_dim` only set in `if not init_pretrained` block | Fixed in `transformer_model.py`: set `embedding_dim = in_channels` in `else` branch |
| `mat1 (1024x1024) and mat2 (1536x768)` shape mismatch | `embedding_dim` was wrongly set to `config.d_model=768`; projection built as `Linear(1536,768)` but input is `in_channels*2=1024` | Fixed condition to `config.d_model != embedding_dim` and `embedding_dim = in_channels = 512` |
| `rsync --delete` wiped `data/en-zh/` on Drive | Push script synced `SeqDiffuSeq/` with `--delete`, deleting Colab-generated files not present locally | Added `--exclude='data/'` and `--exclude='ckpts/'` to `push_gdrive.sh` |

### Clamping collapse at step 3,000 — root cause & fix

Session 02 training collapsed again (loss ~0.07 by step 3,000, same as Session 01). Root cause: `in_channel=512` but BART-base has `d_model=768`, so the custom `BartModel` (which defaults `embedding_dim=512`) couldn't load the pretrained 768-dim embeddings — `ignore_mismatched_sizes=True` silently skipped them. Embeddings remained randomly initialized → same centroid collapse as Session 01.

**Fix**: change `in_channel` to match BART-base exactly.

| Arg | Before | After |
|-----|--------|-------|
| `--in_channel` | 512 | **768** |
| `--out_channel` | 512 | **768** |
| `--num_channels` | 2048 | **3072** |
| `--num_heads` | 8 | **12** |

### SeqDiffuSeq source code patched

File: `src/modeling/predictor/transformer_model.py` — four bugs fixed, all in `init_pretrained=True` path which was never tested in the original repo:

| Bug | Fix |
|-----|-----|
| `AttributeError: no attribute embedding_dim` | Added `else` branch: `self.embedding_dim = in_channels` |
| Projection condition wrong (`in_channels != embedding_dim`) | Changed to `config.d_model != embedding_dim` |
| `input_up_proj_dec` set to Identity when `d_model == embedding_dim` | `input_up_proj_dec` always built as `Linear(embedding_dim*2, d_model)` — self-conditioning always doubles the decoder input |
| `input_up_proj` used in else branch instead of `input_up_proj_dec`/`input_up_proj_enc` | Renamed to match what forward() calls |
| `from_pretrained` ignored `embedding_dim` — created 512-dim embedding even with `in_channel=768` | Pass `embedding_dim=self.embedding_dim` to `from_pretrained()` so pretrained 768-dim weights actually load |

---

## What to Try in Session 03

_Fill in after reviewing results._

Ideas from brainstorm (see `brainstorm.md`):
- Increase `sequence_len` from 64 → 128 (currently 23% of ZH sentences truncated)
- MBR decoding (`mbr_sample=5`) for better output selection
- `generate_by_q=True` for more stable diffusion trajectory
- `clip_denoised=True` to keep embeddings in [-1, 1] during sampling
