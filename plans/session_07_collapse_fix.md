# Plan: Session 07 — Collapse Fix + Better Monitoring

## Context

EN→RU training collapsed twice (Sessions 05 and 06): model predicts near-zero embeddings → rounding yields EOS/PAD → empty string outputs. Grad norm drops 100× (0.103 → 0.001) within ~200 steps. Best usable checkpoint is `ema_0.9999_110000.pt` (step 110k) locally on Windows. Total wasted A100 spend: ~$34 ($17/session × 2 sessions).

Goal: (1) fix the root cause in code so Session 07 doesn't collapse, (2) add early stopping so future runs self-terminate when collapse detected rather than burning GPU money, (3) fix the infer_en_ru.sh bug, (4) protect future checkpoints from accidental loss.

---

## Collapse Root Cause (Two-Component Analysis)

### Why EN→DE (paper) doesn't collapse

Critical config diff:
| | EN→RU (ours) | EN→DE (paper) |
|---|---|---|
| `init_pretrained` | True | **False** |
| `embedding_dim` | 768 (from BART) | **128** (random) |
| Embeddings | BART pretrained | Random N(0,0.02) |

With `init_pretrained=False` (`embedding_dim=128` per transformer_model.py line 80): all token embeddings are random — no token has a privileged near-zero embedding. When model drifts toward zero, rounding hits a random non-special token → garbled output (not empty). The `seq[seq>2]` filter doesn't remove garbled tokens → no empty strings → "looks fine."

With `init_pretrained=True`: BART's pretrained BOS/PAD/EOS embeddings are near-zero (receive minimal gradient during BART pretraining). When MSE attractor pulls model_output → 0, rounding hits token IDs 0/1/2 → filtered by `seq[seq>2]` → empty string. THAT is why our collapse manifests as empty strings while the paper doesn't report collapse.

### Component 1: Zero-embedding attractor (primary cause, ~140-150k steps)

```
training_losses():
    x_start = embed(input_ids) * embed_scale   # scaled BART embeddings
    model_output = model(x_t, t)               # predicted x0

    # MSE pushes model_output → mean(x_start) ≈ 0 (BART mean ≈ 0)
    terms["mse"] = mean_flat((x_start - model_output)**2)

    # decoder_nll uses x_start (GROUND TRUTH), NOT model_output
    # → no gradient pushes model_output AWAY from zero
    decoder_nll = token_discrete_loss(x_start, get_logits, input_ids)

    terms["loss"] = terms["mse"] + decoder_nll + tT_loss
    # MISSING: penalty when model_output maps to wrong tokens
```

Evidence: step 150k sample (original uninterrupted Session 05 run) already shows 4/10 empty. Step 140k: 0/10 empty. So collapse started at ~step 145-150k in the ORIGINAL run, independent of any restart.

### Component 2: Optimizer state never saved (accelerant on restart)

`trainer.py save()` saves `model*.pt` and `ema_*.pt` — NO `opt*.pt`.
`_load_optimizer_state()` looks for `opt{NNNNNN}.pt` — never finds it.

Every resume starts Adam from scratch (m₁=m₂=0). Fresh Adam with zero moments acts like sign-SGD for the first ~100 steps — much more aggressive than tuned Adam. This pushes a model already near the attractor basin into it within 200 steps.

Evidence: log entry 1 = grad_norm 0.103 (resume point), entries 2-4 = 0.013, 0.00255, 0.00122 within ~300 steps of resume. The same checkpoint (model110000.pt) had healthy grad_norm 0.1 in the original run.

**Fix priority**: Fix #1 (optimizer save) is necessary but not sufficient — collapse would still happen at 150k. Fix #2 (rounding loss) is the real cure.

---

## Changes Implemented (all committed to main, d49fbd9)

### Change 1: Save Optimizer State — `SeqDiffuSeq/trainer.py`

In `save()`, after the `for rate, params` loop (before `dist.barrier()`):
```python
if dist.get_rank() == 0:
    opt_filename = f"opt{(self.step+self.resume_step):06d}.pt"
    print('writing optimizer to', bf.join(self.checkpoint_path, opt_filename))
    with bf.BlobFile(bf.join(self.checkpoint_path, opt_filename), "wb") as f:
        torch.save(self.opt.state_dict(), f)
```

### Change 2: Rounding Loss on Model Output — `SeqDiffuSeq/src/modeling/diffusion/gaussian_diffusion.py`

After `model_out_x_start = self.x0_helper(model_output, x_t, t)["pred_xstart"]`:
```python
model_out_discrete_nll = self.token_discrete_loss(
    model_out_x_start, get_logits, input_ids, mask=loss_mask
)
terms["model_rounding_nll"] = model_out_discrete_nll
```

Loss line changed from:
```python
terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)
```
to:
```python
terms["loss"] = terms["mse"] + (decoder_nll + tT_loss) + model_out_discrete_nll
```

### Change 3: Grad-Norm Early Stopping — `scripts/monitor_training.sh`

Collapse detection in main loop: grad_norm < 0.005 for 8/10 readings → kills training + writes `COLLAPSE_FLAG` → stops auto-restart.
Threshold rationale: healthy ~0.1, collapse ~0.001, 0.005 = 5× safety margin. Skip before step 20k.

### Change 4: EMA Auto-cleanup — `scripts/monitor_training.sh`

`cleanup_disk()` now keeps only last 2 EMA checkpoints (~785MB each) and only the latest `opt*.pt` (~350MB).

### Change 5: Fix infer_en_ru.sh — `scripts/infer_en_ru.sh`

Added `--src en --tgt ru`. Without these, defaults to `.src`/`.tgt` suffixes which don't exist → silent empty inference output.

### New file: `analysis/check_embedding_norms.py`

Diagnostic: loads a checkpoint on CPU and checks BART special-token embedding norms. If PAD/EOS/BOS norms << mean → attractor confirmed.
Usage (from `SeqDiffuSeq/`): `python3 ../analysis/check_embedding_norms.py ckpts/en-ru/model110000.pt`

---

## Validation Strategy

Diffusion models need 100k–150k steps before collapse becomes visible. A 10k test is too short. Using `model110000.pt` (already at pre-collapse point) as the starting checkpoint: 50k diagnostic = directly tests whether fix holds at the critical 140-160k range.

### RTX 3090 diagnostic from model110000.pt (~$0.50–$1.00, 3–6 hrs)

| Signal | Fix working | Fix NOT working |
|--------|------------|----------------|
| `grad_norm` at step 115k | > 0.05 | < 0.005 |
| `model_rounding_nll` in log | 1.0–5.0 range | Very low OR very high |
| Sample at step 120k | Garbled Russian (normal) | Empty strings |
| `grad_norm` at step 150k | Stays > 0.05 | Drops to 0.001 |
| Sample at step 160k | Non-empty (improved?) | Collapsed |

If collapse still appears → fall back to `init_pretrained=False` (paper's original config, no attractor).
If fix works → A100 full run FROM SCRATCH (~500k steps, ~$45).

### Why "from scratch" for the final run

The model at step 110k trained 110k steps WITHOUT the rounding loss — the attractor gradient was missing the whole time. Starting from step 0 with the fix gives the model a better embedding space from the beginning. The RTX 3090 diagnostic is only for validating the fix.

---

## Status

- [x] All 5 code changes committed to main (d49fbd9)
- [x] Pushed to GitHub
- [ ] RTX 3090 validation run in progress (instance 40896196)
  - Setup + data prep running on instance
  - model110000.pt uploading via scp
  - Will launch `bash scripts/train_en_ru.sh` once upload + setup complete
- [ ] Monitor grad_norm and model_rounding_nll at steps 115k, 150k, 160k
- [ ] If fix confirmed → plan full A100 run from scratch
