# Session 07 — Pitfalls & Root Cause Analysis

## Overview
Applied `model_rounding_nll` collapse fix and retrained EN→RU from step 260k on Hungary RTX 5090 (instance 41028305).
Training ran from step 260k → 500k. Stopped at 500k because `lr_anneal_steps=500000` sets both LR schedule AND training termination.
Total new steps: ~240k. Instance: 31.46.164.34:18444, RTX 5090.

Inference results: `SeqDiffuSeq/results/en-ru/inference_out/step500000.csv`
BLEU: **0.01 (13a) / 0.65 (char)** — effectively zero.

---

## Pitfall 7: model_rounding_nll Fix Prevents Empty-String Collapse But Model Finds a New Attractor (Ğ Token)

### What happened
- Session 06 root cause: MSE attractor drives model output → near-zero embedding → rounding to EOS/PAD → empty strings.
- Fix applied (Session 07): added `model_rounding_nll` loss term — penalizes when predicted x0 maps to wrong tokens, pushes model away from EOS/PAD embeddings.
- Training appeared healthy at step 500k:
  - `loss=0.036`, `mse=0.023`, `model_rounding_nll=0.013`, `grad_norm=0.272`
  - No empty strings during training-time inference checks
  - No COLLAPSE_FLAG triggered
- Inference (2000 DDIM steps, 500 sentences): hypothesis is ~90% Ğ characters (U+011E) with sparse Russian words mixed in.
- BLEU 0.01 — model has not learned to translate.
- Tried `--clamp clamp` instead of `--clamp no_clamp`: **identical output**, same Ğ pattern.

### Root Cause
**The model traded one near-zero-norm attractor (EOS/PAD, token IDs 0–2) for another (Ğ = U+011E, BART byte-level BPE encoding of ASCII byte 0x1E).**

Mechanism:
1. `model_rounding_nll` correctly penalized EOS/PAD predictions — those tokens no longer receive the model's output
2. BART pretrained embeddings have multiple near-zero-norm tokens, not just EOS/PAD
3. Ğ (U+011E) is the GPT-2 byte-level BPE encoding of ASCII control character 0x1E (Record Separator). Its pretrained embedding is near-zero in BART.
4. MSE attractor still drives model toward zero; now Ğ is the nearest token with a low-norm embedding
5. The fix patched one specific attractor but the underlying cause (BART's pretrained near-zero embedding cluster) remains

### Why training metrics were misleading
`model_rounding_nll=0.013` at step 500k looks like the model is correctly predicting tokens. But:
- During training, diffusion step t is sampled uniformly from [0, T=2000]
- At low-noise steps (small t), x_t ≈ x_0, so the model's x0 prediction is trivially accurate
- The rounding NLL is dominated by these easy low-t samples, masking failure at high t
- Inference always starts at t=2000 (pure noise) — the model fails there even though training NLL looks good

### Why clamp vs no_clamp makes no difference
With `--clamp clamp`, each DDIM step projects x_0 back to the nearest token embedding. But since the model's generated embeddings are already closest to Ğ, clamping just snaps to Ğ and stays there. The trajectory never escapes the Ğ attractor regardless of clamping.

### Inference output quality
| Setting | Hypothesis example | BLEU (13a) |
|---|---|---|
| no_clamp, 500 sents | `"Ğ.ĞĞ внестиĞĞĞĞĞ..."` | 0.01 |
| clamp, 50 sents | identical | ~0.01 |

Full CSV: `SeqDiffuSeq/results/en-ru/inference_out/step500000.csv` (500 rows, columns: source_en, hypothesis_ru, reference_ru)

---

## Pitfall 8: lr_anneal_steps Is Both LR Schedule Length AND Training Stop Condition

### What happened
- `train_en_ru.sh` sets `--lr_anneal_steps 500000`
- Expected: train to 1M steps (continuing from 260k checkpoint)
- Actual: training stopped at exactly step 500k

### Root Cause
In `trainer.py`, the training loop terminates when `self.step + self.resume_step >= self.lr_anneal_steps`. The LR reaching zero IS the termination condition. There is no separate `max_steps` parameter.

With resume at step 260k and `lr_anneal_steps=500000`, the model trained for 240k new steps (260k → 500k), then exited.

### Prevention
To train beyond 500k total steps, set `--lr_anneal_steps` to the desired total step count before launching. For 1M total steps from a 260k checkpoint, use `--lr_anneal_steps 1000000`.

---

## Pitfall 9: init_pretrained=True With a Custom Tokenizer Causes Vocab-Position Mismatch in lm_head

### What happened
Embedding analysis of the trained model revealed:
- `lm_head.weight [32005, 768]`: mean norm 12.94, but `<pad>` norm = 36.22
- `<pad>` bias = **+7.21** (enormously above all others, which are negative)
- `<pad>` and Ğ are nearly anti-parallel: `cosine(W[1], W[223]) = -0.91`
- Before training: 1 token with cosine > 0.99 to Ğ (Ğ itself)
- After 500k steps: **325 tokens** with cosine > 0.99 to Ğ — a completely degenerate cluster

The 325-token cluster includes all extended Latin characters (À, Á, Â, Ý, ą, Ğ, ğ, etc.).

### Root Cause: Index-Position Copy, Not Token-Identity Copy

`build_embeddings()` in `transformer_model.py` lines 147–152:

```python
pretrained_vocab_size = self.input_transformers.shared.weight.shape[0]
out_size = self.vocab_size if self.vocab_size < pretrained_vocab_size else pretrained_vocab_size
self.lm_head = nn.Linear(self.embedding_dim, out_size)
with th.no_grad():
    self.lm_head.weight = nn.Parameter(self.input_transformers.shared.weight[:out_size])
```

This assigns BART's embedding at position N to `lm_head.weight[N]` — regardless of whether BART's token N matches our custom token N.

Example mismatch at ID 223:

| | BART vocab | Custom EN-RU BPE |
|---|---|---|
| Token 223 | `Ġunder` (common English subword) | `Ğ` (U+011E, rare extended Latin) |
| BART embedding norm | ~1.3 (low — "Ġunder" is uncommon in BART) | wrongly inherits ~1.3 |

Extended Latin chars (Ğ, À, Á, Ý, ą…) cluster at token IDs ~129–230 in our bilingual BPE. BART's embeddings at those same IDs correspond to uncommon English subwords with **small norms** (~1.3–1.6 vs. BART mean 5.08). Our tokenizer's completely different tokens inherit those small norms.

### Three-Layer Collapse Chain

```
Layer 1 (init): init_pretrained=True copies BART embedding by index position
                → tokens 129–230 (extended Latin in our BPE) inherit small-norm BART embeddings
                → mismatched — BART had completely different tokens at those positions

Layer 2 (training): Small-norm tokens receive similar gradient signals (equally
                    "forgotten" by BART, equally confused by mismatch)
                    → all 325 tokens converge to same direction after 500k steps
                    → cosine > 0.99 cluster; Ğ is the representative

Layer 3 (loss fix): model_rounding_nll penalizes <pad> predictions (bias = +7.21)
                    → x0_pred pushed away from <pad> direction
                    → lands in Ğ cluster (cosine(<pad>, Ğ) = -0.91 means anti-pad = Ğ)
                    → inference outputs Ğ at every position
```

### Why This Was Hard to Detect

Training logs looked healthy at step 500k:
- `loss=0.036`, `mse=0.023`, `model_rounding_nll=0.013`, `grad_norm=0.272`
- No empty strings; no COLLAPSE_FLAG
- `model_rounding_nll` was low because at small noise levels (t close to 0), x_t ≈ x_0, so predictions are trivially accurate — this masked complete failure at t=2000 (inference start)

Clamping (`--clamp clamp`) made no difference: clamping each DDIM step to the nearest token embedding just snaps to Ğ faster, since the cluster is the nearest thing in embedding space.

### Why Patching Is Whack-a-Mole

| Session | Attractor | Symptom | Patch |
|---|---|---|---|
| 05–06 | `<pad>`/EOS (zero-region, bias dominates) | Empty strings | `model_rounding_nll` |
| 07 | Ğ cluster (anti-pad direction, 325 tokens) | Ğ-filled strings | ??? |

Any further patch (penalize Ğ) will send x0_pred somewhere else in a structurally broken embedding space. The cluster occupies a large solid angle; with 325 tokens converged to the same direction, any unlucky initialization point gets pulled in.

### Fix

Use `init_pretrained=False` with `in_channel=128, out_channel=128` (the paper's original config):
- All embeddings initialized as N(0, 0.02) → equal norm ~0.23 for every token
- No token has a privileged direction, no inherited mismatch, no attractor
- 128 dims: the paper achieves BLEU ~30+ on EN→DE at this dimension

**Code change in `train_en_ru.sh`** (already applied for Session 08):
```bash
--init_pretrained    False \
--in_channel         128 \
--out_channel        128 \
```

### Quick Sanity Check Before Any Future Run With a Custom Tokenizer

If ever using `init_pretrained=True` with a non-BART tokenizer, check alignment first:
```python
from transformers import BartTokenizer
import json
bart = BartTokenizer.from_pretrained("facebook/bart-base")
with open("data/en-ru/vocab.json") as f:
    custom = {v: k for k, v in json.load(f).items()}
mismatches = [(i, bart.convert_ids_to_tokens(i), custom[i])
              for i in range(min(len(custom), bart.vocab_size))
              if bart.convert_ids_to_tokens(i) != custom[i]]
print(f"{len(mismatches)} mismatches out of {len(custom)} tokens")
# If this is >> 5, init_pretrained=True is wrong for this tokenizer.
```

---

## Root Cause Summary: Why init_pretrained=True Fails Long-Term

| Config | Embedding init | Attractor risk | Observed collapse |
|---|---|---|---|
| Paper (EN→DE) | `init_pretrained=False`, dim=128 | None — random embeddings, no zero-norm cluster | No collapse reported |
| Our runs (EN→RU) | `init_pretrained=True`, dim=768 | High — BART pretrained embeddings have near-zero-norm cluster (EOS, PAD, control chars) | Session 05: empty strings; Session 07: Ğ-filled strings |

**Patching individual tokens (EOS/PAD → Ğ) is whack-a-mole. The root fix is to use random embeddings (`init_pretrained=False`) so no token has a privileged near-zero embedding.**

---

## Recommended Fix for Session 08

Switch to `init_pretrained=False` with `embedding_dim=128` — the paper's original configuration.

**Why this fixes it:**
- Random N(0, 0.02) embeddings: all tokens have roughly equal L2 norm (~0.02 × √128 ≈ 0.23)
- No token is near zero → MSE attractor doesn't map to any specific token → no collapse
- This is exactly how the paper achieves BLEU ~30+ on EN→DE

**Cost estimate:**
- RTX 3090 validation from `model110000.pt` → +50k steps (covers the 140k–160k danger zone): ~3–6 hrs, ~$0.50–1.00
- A100 full run from scratch to 500k steps: ~9 hrs, ~$10

**Code change:** In `train_en_ru.sh`, change:
```bash
# Current (broken):
--init_pretrained True \

# Fixed:
--init_pretrained False \
--in_channel 128 \
--out_channel 128 \
```
Also update `gaussian_diffusion.py` embedding_dim reference if hardcoded.
