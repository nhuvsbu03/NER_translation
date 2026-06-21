# Session 08 — EN→RU with paper config (`init_pretrained=False`, dim 128)

## Context

Sessions 05–07 (EN→RU, `init_pretrained=True`, dim 768) all collapsed via BART's
near-zero-norm embedding attractor: MSE drove the model output → ~0 → rounding hit
EOS/PAD (empty strings, S05/06) or the Ğ cluster (S07). BLEU ≈ 0 every time. The S07
PITFALLS concluded that patching individual attractor tokens is whack-a-mole and the real
fix is the paper's original configuration — `init_pretrained=False` with random N(0,0.02)
embeddings, where every token has ~equal norm so no token is a privileged attractor.

**Session 08 executed that fix** and trained EN→RU to step 800k.
Artifacts: `D:\Learning\Research\scr\session_8\` (`checkpoints/`, `inference/`,
`logs/train.log`, `training_args.json`, `alpha_cumprod_step_98000.npy`).

---

## Configuration (`session_8/training_args.json`)

| Setting | Value | Note |
|---|---|---|
| `init_pretrained` | **False** | the fix — random equal-norm embeddings, no attractor |
| `in_channel` / `out_channel` | **128** | ⚠ also sets `config.d_model = 128` (see Root Cause) |
| `num_channels` | 512 | → encoder/decoder FFN dim |
| `embedding_dim` | 128 (hardcoded) | diffusion embedding |
| `num_heads` | 8 | |
| encoder/decoder layers | 6 / 6 | |
| `batch_size` | **32** | low — see Under-training |
| `lr` | 3.75e-5 | |
| `lr_anneal_steps` | **1,000,000** | Pitfall 8 fixed (was both schedule + stop) |
| `diffusion_steps` | 2000 | |
| `sequence_len` / `_src` | 128 / 128 | |
| resume | `model500000.pt` | ran 500k → ~818k |

Data exposure at 818k: **samples ≈ 2.62e7 (~26M)**.

---

## Training progress (`logs/train.log`, steps ~505k–818k)

| Metric | Value | Reading |
|---|---|---|
| `grad_norm` | **0.7 – 1.4** (stable) | healthy — NO death-spiral (S05–07 dropped to 0.001) |
| `loss` | ~1.1 (flat) | stable; much higher than S07's 0.036 because dim-128 random embeddings make MSE meaningful |
| `mse` | ~0.18 | |
| `model_rounding_nll` | ~0.85 | |
| `loss_q0 / q1 / q2 / q3` | 0.37 / 0.9 / 1.3 / 1.6 | denoises low-noise well, **high-noise poorly** (q3 = where sampling starts at t=2000) |
| `model_rounding_nll q0→q3` | 0.22 → 1.3 | same pattern |

No COLLAPSE_FLAG, no empty strings during training-time checks.

---

## Inference results (full newstest2014, 3,003 pairs; sacrebleu)

| Checkpoint | BLEU (13a) | chrF | Latin-dominant outputs |
|---|---|---|---|
| ema 500k | 0.071 | 14.6 | 12% |
| ema 750k | 0.044 | 17.0 | 2% |
| ema 800k (`step800000.csv`) | **0.041** | 16.8 | 3% |

- BLEU is **flat/declining** 500k→800k → more steps at this config will not help.
- Output is **fluent-looking Russian word-salad** — correct alphabet, real morphemes, but
  semantically unrelated to the source (behaves like an *unconditional* Russian LM).
- A minority of early outputs were English "the the the" mode-collapse; that fraction
  shrinks with training (12%→3%) as the target-side LM strengthens.
- chrF ≈ 17 is incidental overlap of common Russian function words/characters.
- `clamp` vs `no_clamp`: no meaningful difference (consistent with S07).

**Sample (ema 800k, source / hypothesis / reference):**
```
SRC: In Kineshma and environs two men have committed suicide
HYP: ери-п, Иоуус-А-о- «иб-- и К- иктра,дМра» такМШшВ-к,придЦен И-гиусРент.арЛёз...
REF: В Кинешме и районе двое мужчин покончили жизнь самоубийством
```

---

## What changed vs Sessions 05–07

| | S05/06 | S07 | **S08** |
|---|---|---|---|
| `init_pretrained` | True | True | **False** |
| dim | 768 | 768 | **128** |
| Collapse mode | empty strings | Ğ-filled strings | **none (word-salad)** |
| `grad_norm` | → 0.001 | → low | **stable 0.7–1.4** |
| BLEU | ~0 | 0.01 | **0.04** |

**Net:** the embedding-attractor collapse is genuinely fixed (the S07 diagnosis was correct).
A *new* failure mode remains: the model trains stably but does not learn to condition on the
source.

---

## Root cause (hypothesis — to be confirmed in Session 09)

In the `init_pretrained=False` path,
[`transformer_model.py:71-84`](D:\Learning\Research\scr\NER_translation\SeqDiffuSeq\src\modeling\predictor\transformer_model.py):
```python
self.config.d_model         = in_channels      # 128  ← becomes the WHOLE transformer width
self.config.encoder_ffn_dim = model_channels   # 512  (= --num_channels)
self.embedding_dim          = 128              # HARDCODED diffusion embedding
```
The paper's "Embedding Dimension = 128" is only the *diffusion embedding*; its transformer
backbone is **d_model=512, FFN=2048**. Setting `in_channel=128` collapsed `d_model` to 128
and FFN to 512 (~4× narrower). Because the encoder up/down projections are only built when
`d_model != embedding_dim` ([line 129](D:\Learning\Research\scr\NER_translation\SeqDiffuSeq\src\modeling\predictor\transformer_model.py)),
with both = 128 they were **never created** — the architecture is structurally different
from the paper, not merely smaller.

Confirmation in `logs/train.log`: the model dump shows `Embedding(32005, 128)` and
`fc1: Linear(128, 512)` — the whole backbone is 128-wide / FFN 512.

Secondary issue: **under-training** — `batch_size=32` × 818k = ~26M samples vs the paper's
~128M (1M × 128), ~5× short. (Not the primary cause since BLEU is flat, not still climbing.)

---

## Status / next

- [x] Attractor collapse fixed (S07 hypothesis validated).
- [x] BLEU still ≈ 0; root cause hypothesized (transformer width collapsed to 128).
- [ ] **Session 09** (`session_09_diagnosis.md`): CPU-only probes to *prove* whether the
  cause is (a) the decoder ignoring the source / architecture, (b) high-noise capacity /
  under-training, (c) a rounding/tokenizer bug, or (d) an inference-setting bug — before
  spending GPU money on a wider retrain.

## Cost
Training was done on rented GPU (per project workflow); checkpoints + inference pulled to
`D:\Learning\Research\scr\session_8\`. Exact $ not recorded here.
