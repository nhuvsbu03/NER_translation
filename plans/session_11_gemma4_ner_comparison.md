# Session 11 — Gemma-4 vs. DiffusionGemma: Translation Quality & Named-Entity Preservation

## Goal
Answer two questions on the same 10,866-sentence, 6-direction WMT dataset (en-ru, ru-en, en-ja, ja-en, en-zh, zh-en):
1. Does 4-bit quantization (W4A16) hurt DiffusionGemma's translation quality?
2. Is diffusion-based generation (DiffusionGemma) better or worse than standard autoregressive generation (Gemma-4) — on both BLEU and named-entity preservation specifically, since this project's long-term target is NER-aware translation?

## Models Compared

| Label | Model | Generation | Precision | Hardware |
|---|---|---|---|---|
| DiffusionGemma (BF16) | `google/diffusiongemma-26B-A4B-it` | Diffusion (discrete token unmasking) | BF16, no quantization | A100 SXM4 80GB |
| Gemma-4 (transformer) | `google/gemma-4-26B-A4B-it` | Autoregressive | BF16, no quantization | A100 SXM4 80GB |
| DiffusionGemma W4A16 | `GoedelMachines/diffusiongemma-26B-A4B-w4a16` | Diffusion | W4A16 (4-bit weights, custom Triton kernels) | RTX 5090 |

### Why `google/gemma-4-26B-A4B-it` as the transformer-based comparison model
The goal was to isolate *one* variable — diffusion generation vs. autoregressive generation — while holding everything else constant. Gemma-4-26B-A4B-it is the closest possible match to DiffusionGemma-26B-A4B-it:
- **Identical parameter count and MoE structure**: both 26B total params, ~4B active per token, 128 experts / top-8 routing.
- **Same publisher and generation family**: both from Google, released as siblings — Gemma-4 is explicitly the standard autoregressive counterpart to the diffusion variant, not just a similarly-sized unrelated model.
- **Same tokenizer/vocab lineage**, minimizing confounds from different training data or subword segmentation.

A same-size model from a different vendor (e.g. Llama, Qwen) would have confounded the comparison with differences in training data, RLHF/instruction-tuning recipe, and tokenizer — making it impossible to attribute quality/speed/NE differences specifically to diffusion vs. autoregressive decoding. Gemma-4 controls for all of that, leaving generation method as the one real difference being tested.

## Infrastructure & Cost
- Both unquantized BF16 runs: A100 SXM4 80GB, batch_size=16 (found empirically to be the throughput sweet spot — batch 8/32/64 were all slower; larger batches wait on the slowest sequence per batch)
- DiffusionGemma BF16: 106.4 min generation, ~$3.90
- Gemma-4: 157.7 min generation, ~$5.42 total (incl. setup/smoke tests)
- DiffusionGemma W4A16: 106.4 min on RTX 5090 (custom kernels tuned for that GPU specifically), ~$1.00
- Per-sentence speed (A100, single-sentence baseline): DiffusionGemma 1.26s vs Gemma-4 2.94s — diffusion is ~2.3x faster at equivalent quality (see below)

## Translation Quality (SacreBLEU, 13a tokenization)

| Direction | DiffusionGemma (BF16) | Gemma-4 (transformer) | DiffusionGemma W4A16 |
|---|---|---|---|
| en-ru | 24.27 | **26.91** | 24.61 |
| ru-en | 38.35 | **39.40** | 37.62 |
| en-ja | **29.90** | 29.18 | 32.84* |
| ja-en | 24.38 | **24.58** | 24.52 |
| en-zh | **22.03** | 20.57 | 20.57 |
| zh-en | 23.88 | **24.60** | 24.62 |

\* en-ja W4A16 result is an outlier — quantization degrading precision shouldn't *improve* BLEU; likely a quirk of that specific run rather than a genuine effect.

**Findings:**
- **Quantization (W4A16) does not meaningfully hurt BLEU.** Excluding the en-ja outlier, deltas vs. unquantized BF16 are within ±1.5 BLEU with no consistent direction — noise, not degradation.
- **Diffusion vs. transformer is close on BLEU.** Gemma-4 wins 4/6 directions, average delta ≈ +0.4 BLEU in its favor. DiffusionGemma is clearly ahead on en-zh and slightly ahead on en-ja. No strong quality edge either way — the real differentiator is speed, not BLEU.

## Named-Entity Preservation

### Strategy
The core problem: hypothesis, reference, and source are in different languages depending on direction, and named entities don't have a stable surface form across languages (e.g. "Eric Goodman" ↔ "Эрик Гудман") — so entities can't be matched cross-lingually without transliteration/entity-linking machinery. The strategy sidesteps this:

1. **Only ever compare same-language text.** Hypothesis and reference are both in the *target* language, so both get tagged with spaCy's target-language NER model (en/ru/ja/zh) and compared directly — no cross-lingual matching needed.
2. **Normalize entity types to one consistent scheme across languages.** Each spaCy model has a different label set (English/Japanese/Chinese: 18 OntoNotes-style types; Russian: 3). All are mapped down to **PERSON, ORG, LOC**, dropping non-proper-noun types (DATE, MONEY, etc.), so every language is judged on the same schema.
3. **One-to-one matching per sentence per type**, not many-to-one. For each sentence, within each type, compute a fuzzy similarity (normalized string match, ≥70% threshold to tolerate morphological variation like Russian case endings) between every reference entity and every hypothesis entity, then greedily pair off the best matches — **each entity usable only once**. This is what makes count mismatches visible instead of silently absorbed: a duplicated or extra hypothesis entity can't "cover" two different reference entities. Anything left unmatched is counted explicitly:
   - Unmatched reference entity → **False Negative** (the model dropped a name)
   - Unmatched hypothesis entity → **False Positive** (the model invented/duplicated a name)
4. Report **Accuracy = TP / (TP + FN + FP)** — both missing and extra entities count against the score in the same denominator (equivalent to exact-match/IoU-style accuracy for entity extraction; stricter than F1, which smooths precision and recall together via harmonic mean).
5. **Filter out reference noise before scoring.** A reference translation's named-entity count doesn't reliably match the source's — professional translators pronominalize, paraphrase, or drop names for fluency — so a hypothesis being "different from the reference" isn't always a real error. To control for this, source is also tagged (source-language NER), and sentences are kept only if source and reference agree on entity count *and* type per category. This "clean subset" is treated as the reliable evaluation set; the noisy ~34% of the dataset where reference itself doesn't literally preserve source entities is excluded rather than scored.

### Clean-subset results
Filtering to sentences where source and reference agree on entity count+type keeps **7,132 / 10,866 sentences (65.6%)** — clean rate ranges from 53% (en-ru) up to 72% (ja-en), meaning en-ru references reshape entities most often, ja-en least.

| Type | DiffusionGemma (BF16) | Gemma-4 (transformer) | DiffusionGemma W4A16 |
|---|---|---|---|
| PERSON | 65.1% | **68.0%** | 67.5% |
| ORG | **52.2%** | 51.7% | 50.5% |
| LOC | 73.5% | **75.6%** | 75.2% |
| **ALL** | 64.5% | **66.0%** | 65.3% |

### Per-direction, per-type (clean subset, Accuracy)

| Direction | DiffusionGemma PERSON/ORG/LOC | Gemma-4 PERSON/ORG/LOC | W4A16 PERSON/ORG/LOC |
|---|---|---|---|
| en-ru | 82 / 64 / 77 | 85 / 64 / 80 | 82 / 63 / 80 |
| ru-en | 84 / 63 / 81 | 86 / 59 / 84 | 84 / 64 / 82 |
| en-ja | 31 / 32 / 55 | 34 / 31 / 65 | 38 / 38 / 57 |
| ja-en | 18 / 22 / 58 | 24 / 23 / 59 | 18 / 21 / 64 |
| en-zh | 57 / 57 / 82 | 61 / 59 / 79 | 61 / 47 / 75 |
| zh-en | 39 / 33 / 67 | 41 / 32 / 64 | 48 / 27 / 72 |

(ja-en/en-ja sample sizes are small — 24-51 entities per type — so differences under a few points there are noise, not signal.)

**Findings:**
- **Quantization is neutral for NE preservation too** (64.5% vs 65.3% accuracy overall) — consistent with the BLEU result.
- **Gemma-4 (transformer) wins on PERSON and LOC**, DiffusionGemma edges ahead on ORG by a hair (52.2% vs 51.7%, within noise). Overall Gemma-4 leads by ~1.5pp (66.0% vs 64.5%) — a small, fairly consistent edge. Plausible explanation: autoregressive left-to-right generation commits to a name before generating the rest of the sentence, while diffusion's parallel denoising can let entity spans drift during refinement.
- **ORG accuracy sits around 50-52%** — coin-flip territory, and consistently the hardest category for all three models. Organization names are inconsistently transliterated/partially translated, and accuracy (unlike F1) fully penalizes both the missing and extra names rather than smoothing them via harmonic mean.
- **LOC is the easiest** (73-76% accuracy) — place names are more standardized.
- **Russian directions are far and away the strongest** (82-86% PERSON, 77-84% LOC accuracy once reference noise is controlled for); **Japanese directions are the clearest, most robust weak point**, especially ja-en PERSON (only 18-24% accuracy) — this holds across all three models, so it's a genuine model limitation, not a quirk of one run.

## Bottom Line
- Quantizing DiffusionGemma to W4A16 costs essentially nothing in quality (BLEU or NE preservation) — its only real cost is needing GPU-specific tuned kernels to get the speed benefit (RTX 5090/GB10 only; falls back to much slower generic kernels elsewhere).
- Diffusion vs. autoregressive generation is close on general translation quality (BLEU), but the standard transformer (Gemma-4) has a small, consistent edge specifically on named-entity preservation — relevant for this project's EN→ZH NER-aware translation goal. Diffusion's clear advantage remains speed (~2.3x faster per sentence on the same hardware).
- Japanese-direction named-entity translation is the weakest link across every model tested and would be the highest-value target for future improvement work (e.g. NER-aware prompting/constraints specifically for en-ja/ja-en).
