# Session 08 — Pitfalls & Results

## Overview
First clean run with `init_pretrained=False` (paper's original config) on RTX 5090 (instance 41028305, ~Hungary).
Training ran from scratch (step 0) → stopped at step 800k after BLEU regression detected.
Instance: 31.46.164.34:18444, RTX 5090 32GB, batch=32, lr=3.75e-5.

**Root fix applied**: `init_pretrained=False, in_channel=128, out_channel=128` — eliminates the near-zero embedding attractor that caused collapse in sessions 05–07.

---

## Training Config

| Parameter | Value |
|-----------|-------|
| `init_pretrained` | `False` ✓ |
| `in_channel` | `128` ✓ |
| `out_channel` | `128` ✓ |
| `batch_size` | `32` |
| `lr` | `3.75e-5` |
| `lr_anneal_steps` | `1000000` |
| `warmup` | `10000` |
| `save_interval` | `50000` |
| `diffusion_steps` | `2000` |
| `noise_schedule` | `sqrt` |
| `vocab_size` | `32005` |
| `sequence_len` | `128` |
| Instance | RTX 5090, 32GB VRAM |

---

## Inference Results

All inference: 3003 sentences (newstest2014), 2000 DDIM steps, no_clamp, seed=42.

| Step | BLEU (13a) | BLEU (char) | Notes |
|------|------------|-------------|-------|
| 500k | 0.07 | 9.75 | Diverse Russian word salad, no collapse |
| 750k | 0.04 | 6.65 | Regression — function-word dominance |
| 800k | TBD | TBD | Pending |

CSV files: `SeqDiffuSeq/results/en-ru/inference_out/step500000.csv`, `step750000_3003.csv`

---

## Key Observations

### No Collapse — Fix Confirmed
- `grad_norm` stayed at ~0.87–1.01 throughout (collapse threshold = 0.005)
- `model_rounding_nll` ~0.864 — healthy, model is predicting real tokens
- No empty strings, no Ğ-filled outputs (sessions 05–07 failure modes absent)
- Outputs are diverse — every sentence is unique, no repetition across examples

### BLEU Regression 500k → 750k
Outputs at 750k shifted from short garbled Russian to long function-word repetition:

```
500k: "Нанутри так быть что что как сравнении быть страна к продвием..."
750k: " в и так о такчет в на еще может, того мы не что что в не не. что ву же..."
```

750k outputs are dominated by `в, что, не, и, как` (the 5 most common Russian function words) with little content. Average output length increased significantly.

### Root Cause of Regression
Two likely contributors:

1. **LR decay**: With `lr_anneal_steps=1M` and linear decay, at step 750k the LR is 25% of original (0.9375e-5). At step 800k it's 20%. Too low for `batch=32` to make meaningful gradient steps — model optimizes for token frequency rather than translation.

2. **Insufficient data throughput**: `batch=32` × 800k steps = 25.6M samples. The SeqDiffuSeq paper uses batch=64 on faster hardware for longer. At batch=32, 800k steps is only ~40% of the paper's training budget.

---

## Pitfall 10: BLEU Regression Caused by LR Decay + Small Batch

### What happened
- At step 500k (midpoint of LR schedule), BLEU (char) = 9.75
- At step 750k (75% of schedule, LR at 25%), BLEU (char) = 6.65
- Model drifted toward high-frequency function words — a safe but low-quality attractor

### Why
With a linear LR decay schedule that reaches 0 at `lr_anneal_steps`:
- Late-stage LR is extremely small
- Small gradients can't overcome the statistical bias toward frequent tokens
- Model degrades into outputting the most probable tokens regardless of input

### Prevention for next run
Option A — Larger batch (recommended):
- Use A100 with `batch=128`: 4× more signal per step, same GPU hours
- Scale LR: `lr = 1.5e-4` (sqrt(128/32) × 3.75e-5 ≈ 7.5e-5, but paper uses 1e-4 baseline → 1.5e-4 at batch=128)

Option B — Cosine schedule instead of linear:
- Cosine decay is less aggressive at the tail — LR at 75% of schedule is ~50% of max, not 25%
- Change `--noise_schedule` if applicable, or patch `_anneal_lr()` in `trainer.py`

Option C — Checkpoint early:
- Best checkpoint is likely around 400k–600k steps at this batch size
- Run inference every 50k steps and stop when BLEU peaks

---

## Checkpoint Status

| Checkpoint | Location | Quality |
|------------|----------|---------|
| `ema_0.9999_500000.pt` | Remote (41028305) | Best so far (BLEU char 9.75) |
| `ema_0.9999_750000.pt` | Remote (41028305) | Regressed (BLEU char 6.65) |
| `ema_0.9999_800000.pt` | Remote (41028305) | TBD |
| `model800000.pt` | Remote (41028305) | Last saved — training stopped here |

---

## Plan for Session 09

**Goal**: Reproduce or improve on the 500k result with a better training setup.

**Recommended config changes**:
1. Use A100 with `batch=128, lr=1.5e-4` — paper-equivalent throughput
2. Monitor BLEU every 50k steps from step 300k onward
3. Stop when BLEU plateaus rather than running to a fixed step count
4. Consider `lr_anneal_steps=500000` with full A100 batch — equivalent compute to ~4M samples per 50k steps
