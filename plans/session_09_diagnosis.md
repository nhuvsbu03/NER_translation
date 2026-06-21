# Session 09 — Diagnosing EN→RU BLEU ≈ 0 (CPU probes) — CORRECTED

## TL;DR (verified, after self-correction)

Session 08 (`init_pretrained=False`, dim-128) fixed the *attractor collapse* of S05–07 but
still scored BLEU ≈ 0. After CPU probing **and two rounds of self-correction**, the verified
conclusion is:

> **Rounding, the embedding table, `lm_head`, and the (un)masked padding are all FINE. The
> failure is the diffusion model's x0 prediction itself: it does not generate target
> embeddings that encode the translation. This is a capacity + training-scale problem — the
> Session 08 model is ¼ the width and ¼ the batch of the paper's working recipe.**

Evidence (stated precisely):
- **Direct:** the model's actual hypotheses in `session_8/inference/step800000.csv` are word-
  salad (BLEU 0.04) vs clean Russian references.
- **Control (not a generation):** embedding the *reference* and decoding it back through the
  actual (untied, PAD-biased) `lm_head` returns the reference — so `lm_head` *can* decode a
  correct embedding; rounding is not fundamentally broken. (This is a round-trip of the
  reference, NOT model output — do not mistake it for a hypothesis.)
- **Conclusion:** salad output + a rounding head that decodes correct embeddings fine ⇒ the
  model's predicted x0 is far from the true target embedding ⇒ the bottleneck is the diffusion
  generation/conditioning, not decoding. Corroborated by P1.1 (weak source conditioning) and
  P0.3 (noise dominates).

## ⚠ Corrections to earlier Session-09 claims (kept for honesty/traceability)

1. **"Missing padding loss mask is the root cause" — WRONG.** The upstream repo
   (Yuanhy1997/SeqDiffuSeq) has the *same* behavior: `trainer.py` builds no `loss_mask`,
   `dataloader_utils.py` sets the target mask all-ones, `training_losses` runs `loss_mask=None`.
   The paper's EN→DE works with the identical unmasked padding, so it cannot be the cause.
2. **"`lm_head` round-trip 0.8 %, PAD predicted for 99 % of tokens" — MEASUREMENT BUG.** The
   correct round-trip over the full vocab is **62 %** (PAD wins 38 %, mostly on rare/untrained
   tokens). On *real* content tokens it is effectively 100 % (see sentence control below). My
   earlier 0.8 % came from a scaling error when feeding embeddings through the constructed model.
3. **"Untied `lm_head` is the differentiator" — NOT the cause.** The fork *does* untie
   `lm_head` from the embeddings (upstream ties them), and they have drifted (median cosine
   0.93). But tied round-trip (`E·Eᵀ`) is 58.5 % vs untied 62 % — tying would **not** improve
   decoding. Untying is a minor, free-to-fix divergence, not the reason for BLEU 0.

## What actually differs from the paper (the real cause)

Upstream `train_scripts/iwslt_en_de.sh` (working EN→DE) vs Session 08:

| param | paper EN→DE | Session 08 | effect |
|---|---|---|---|
| `in_channel` → `d_model` | **512** | 128 | transformer 4× narrower |
| `num_channels` → FFN | **2048** | 512 | FFN 4× smaller |
| `batch_size` | **128** | 32 | ~¼ data/step (~26M vs ≥64M samples) |
| `sequence_len` (target) | 64 | 128 | minor (more padding) |
| `init_pretrained` | False | False | same |

The undersized, under-trained model cannot learn the conditional denoising — especially the
high-noise regime where generation starts from pure noise and must rely on the source.

## Probe results (all CPU; what each one really shows)

| Probe | Result | Correct reading |
|---|---|---|
| actual hypotheses (step800000.csv) | word-salad, BLEU 0.04 | the real model output is broken (direct evidence) |
| sentence round-trip (control, NOT a generation) | decode(reference's own embedding) == reference | rounding *can* decode a correct embedding ⇒ rounding not the bottleneck |
| vocab round-trip | 62 % (untied) / 58.5 % (tied); PAD 38 % on rare tokens | decoding of real content is fine; PAD bias harmless for content |
| **P1.1** source ablation (fixed noise) | Jaccard true-vs-shuffled 0.31, true-vs-blank 0.08 | model conditions on source only **weakly** |
| **P0.3** same source, diff noise | Jaccard 0.04 | output dominated by noise, not source |
| **P0.1** token freq/entropy | length→cap, entropy 10.3→8.0, comma ~11 % | x0 predictions drift to a generic prior |
| **P0.2** loss-by-noise (logs) | loss_q0 0.37 ≪ loss_q3 1.6 | high-noise denoising (where generation starts) is weak |
| **P1.4** weights | embeddings healthy; cross-attn present (1.17×) | no collapse, source-wiring intact |
| **P1.2** reconstruct-from-t0 | noisy / unreliable | **discount** — this probe had its own scaling issues; superseded by the clean controls above |

Net: every component except the **diffusion x0 prediction** checks out. The x0 predictions are
weakly conditioned and drift to a generic Russian prior → word-salad → BLEU 0.

## Session 10 plan — reproduce the paper recipe, validate cheaply first

1. **Match capacity** in `scripts/train_en_ru.sh` (keep `init_pretrained False`):
   `--in_channel 512 --out_channel 512 --num_channels 2048` (⇒ d_model=512 / FFN=2048;
   encoder up/down projections active).
2. **Match scale**: `--batch_size 128`, paper-scale steps (≥64M samples), paper LR.
3. `sequence_len`: start at 64 to copy the paper; if RU length needs 128, keep 128.
4. **Optional, free**: re-tie `lm_head` to the embedding and/or drop its bias (matches upstream)
   and add a pad loss mask — robustness/efficiency, **not** the fix.
5. Train from scratch (capacity change invalidates S08 checkpoints).

### Cheap CPU re-validation before the full GPU spend
Reuse the probes on the new checkpoint: the **sentence round-trip** must stay exact, and
**P1.1 source-ablation Jaccard must drop** (stronger conditioning) — the direct sign the
bigger model is actually using the source. Only then commit to the full run.

## How to reproduce these checks
- Sentence round-trip / vocab round-trip / tied-vs-untied: load
  `session_8/checkpoints/ema_0.9999_800000.pt`, compute `argmax(E_true @ lm.T + bias)` and
  `decode`; compare untied (`lm`,`bias`) vs tied (`E`,no bias).
- Upstream parity: upstream `trainer.py`/`dataloader_utils.py` lack `loss_mask`;
  `iwslt_en_de.sh` uses in_channel=512 / num_channels=2048 / batch=128.
- Probes: `analysis/probe_output_tokens.py`, `probe_checkpoint_weights.py`,
  `probe_cpu_infer.py` (P1.1). Run with `PYTHONUTF8=1 TRANSFORMERS_OFFLINE=1`.
