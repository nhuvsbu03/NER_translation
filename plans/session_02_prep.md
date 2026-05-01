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
| `e75b12a0` | Check for `*_clean.*` files |
| `99ba7f75` | Copy `*_clean.*` → original names in repo |
| `c5d1ff36` | BPE vocab `10000` → `32000`, always delete old tokenizer |
| `23c64bfb` | `init_pretrained=True`, `vocab_size=32005`, `lr_anneal_steps=100000` |
| `607671b2` | Inference on full test set (dynamic count), `diffusion_steps=200` |

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

_Fill in after run._

---

## What to Try in Session 03

_Fill in after reviewing results._

Ideas from brainstorm (see `brainstorm.md`):
- Increase `sequence_len` from 64 → 128 (currently 23% of ZH sentences truncated)
- MBR decoding (`mbr_sample=5`) for better output selection
- `generate_by_q=True` for more stable diffusion trajectory
- `clip_denoised=True` to keep embeddings in [-1, 1] during sampling
