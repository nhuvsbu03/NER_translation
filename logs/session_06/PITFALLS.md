# Session 06 — Pitfalls & Root Cause Analysis

## Overview
Resumed EN→RU training from `model110000.pt` (clean pre-crash checkpoint).
Instance: 39898796, A100 SXM4 40GB, $1.07/hr.
Ran from step 110k → 280k (stopped Jun 10 after quality collapse confirmed).
Total cost session 06: ~$17 (16 hrs × $1.07).

---

## Pitfall 5: Training Collapses to Empty Strings After Every Restart

### What happened
- Restarted from `model110000.pt` (verified clean, 273 params, 0 NaN).
- Grad norm dropped 100× within the first few hundred steps: `0.103 → 0.001`.
- By step 260k (+150k new steps): 9/10 inference outputs were empty strings.
- Same collapse pattern as Session 05 restart from model150000.pt.
- Collapse happens regardless of which checkpoint we restart from.

### Root Cause (CONFIRMED)
**The diffusion MSE loss converges to a degenerate minimum where the model predicts near-zero embeddings.**

Mechanism:
1. SeqDiffuSeq trains with MSE loss: `L = ||x0_pred - x0_true||²`
2. The mathematical minimum of MSE is to predict the **mean embedding** `E[x0_true]`
3. The mean embedding across the vocabulary is near zero (padding/EOS tokens have small L2 norms and are common)
4. With more training steps, the model increasingly predicts near-zero embeddings
5. The rounding step maps near-zero embeddings → padding/EOS token → empty string output

**Proof:** Comparing DDIM steps:
- 260k EMA with **200 DDIM steps**: 6/10 empty (model partially on zero-attractor)
- 260k EMA with **2000 DDIM steps**: 9/10 empty (more steps = more drift toward zero)

More denoising iterations = more opportunities to refine toward the zero-embedding attractor.

### This is NOT caused by:
- The disk crash (Session 05)
- Bad checkpoints
- LR warmup reset (confirmed: trainer.py uses `self.step + self.resume_step`)
- EMA re-initialization (would only affect first ~50k steps)
- Schedule file mismatch (verified correct pairing at each inference)

### The peak quality window
The model had genuine learning signal between steps **~50k–140k** of the original uninterrupted training (Session 05):
- Step 100k: garbled Russian but real words present
- Step 140k: best quality — most coherent Russian (EMA checkpoint from original run)
- Step 150k+: disk crash + restart → collapse began
- After restart: same collapse trajectory from 110k

After ~150k steps total (original run), the loss landscape becomes flat and the zero-embedding attractor dominates. This appears to be the fundamental training horizon for this setup.

### Quality comparison across all checkpoints
| Checkpoint | Run | Empty/10 | Notes |
|---|---|---|---|
| ema_0.9999_100000.pt | Original | 0/10 | All garbled Russian |
| ema_0.9999_110000.pt | Original | 0/10 | All garbled Russian |
| ema_0.9999_140000.pt | Original | 0/10 | Best quality (partial local copy only) |
| ema_0.9999_160000.pt | Session 06 | 4/10 | Garbled Russian |
| ema_0.9999_210000.pt | Session 06 | 3/10 | Garbled Russian |
| ema_0.9999_260000.pt | Session 06 | 9/10 | Collapsed |

---

## Pitfall 6: ema_0.9999_140000.pt (Best Checkpoint) Lost

### What happened
- `ema_0.9999_140000.pt` was partially downloaded locally (426MB of 785MB) in Session 05.
- During Session 06 training, the training code deleted old EMA files from remote as new ones were saved.
- The remote copy is permanently gone. The local partial copy is unloadable.

### Impact
- Cannot run inference on the best-ever checkpoint.
- Best usable EMA checkpoint now is `ema_0.9999_110000.pt` (full, 822MB, local only).

### Prevention
- Download AND verify each EMA checkpoint IMMEDIATELY after it is saved (don't batch downloads).
- Never rely on the remote copy persisting — the training code deletes old EMA files automatically.
- Script: download + `python3 -c "import torch; d=torch.load(f); print(len(d))"` to verify before considering it saved.

---

## Investigation Needed Before Next Run

### Fix options for the zero-embedding collapse

**Option 1: Embedding clamping during training**
Add a regularization term that penalizes predictions near zero:
```python
# In gaussian_diffusion.py, training_losses():
zero_penalty = lambda_reg * mean_flat(model_out_x_start ** 2, loss_mask)  # penalize near-zero
terms["loss"] = terms["mse"] + decoder_nll + tT_loss - zero_penalty  # push away from zero
```
Or clamp the model's x0 predictions away from zero during training (currently clip_denoised=False).

**Option 2: Enable clip_denoised=True during training**
In `gaussian_diffusion.py` line 580-581, `clip_denoised=True` clamps predictions to [-1, 1].
This would prevent extreme near-zero drift during the diffusion process.

**Option 3: Anchor padding token embedding at zero, others away from zero**
Check whether padding token embedding is at zero in the BART embedding matrix.
If so, initialize with a different scheme that keeps all meaningful tokens away from zero.

**Option 4: Add embedding norm regularization**
After each optimizer step, normalize all output embeddings to have unit L2 norm.
This removes the "predict near zero = low MSE" shortcut.

**Option 5: Reduce lr_anneal_steps**
The LR is too high for too long, allowing the optimizer to find the zero-attractor.
Try stopping LR annealing earlier (e.g., lr_anneal_steps=150000).

**Option 6: Early stopping based on inference quality**
Don't train to convergence. Stop at first sign of quality degradation (grad_norm drops below 0.01).
From our data, this happens around step 150k. Set early stopping at step 130k.

### Recommended approach for Session 07
1. Add grad_norm threshold check: if `grad_norm < 0.005` for 3 consecutive eval intervals, stop training
2. Test Option 2 (clip_denoised during training) — minimal code change
3. Run a 50k-step diagnostic with clip_denoised=True to see if quality holds past 150k

---

## Checkpoint Status After Session 06

| Checkpoint | Location | Quality | Notes |
|---|---|---|---|
| `ema_0.9999_000000.pt` | Local only | Random | 822MB, full |
| `ema_0.9999_010000.pt` | Local only | Random | 822MB, full |
| `ema_0.9999_020000.pt` | Local only | Random | 822MB, full |
| `ema_0.9999_030000.pt` | Local only | Random | 822MB, full |
| `ema_0.9999_040000.pt` | Local only | PARTIAL | 155MB, unusable |
| `ema_0.9999_110000.pt` | Local only | Garbled Russian | 822MB, full — best usable |
| `ema_0.9999_140000.pt` | Local only | PARTIAL | 426MB, unusable — was best ever |
| `ema_0.9999_200000.pt` | Local only | PARTIAL | 474MB, unusable |
| `ema_0.9999_210000.pt` | Local only | Garbled Russian | 785MB, from session 06 |
| `ema_0.9999_250000.pt` | Local only | PARTIAL | 433MB, unusable |
| `ema_0.9999_260000.pt` | Local only | Mostly empty | 785MB, from session 06 |
| `model110000.pt` | Local only | — | 638MB, full — best for resuming |
| `model160000.pt` | Local only | — | 638MB, from session 06 |
| `model210000.pt` | Local only | — | 638MB, from session 06 |
| `model260000.pt` | Local only | — | 638MB, from session 06 |
