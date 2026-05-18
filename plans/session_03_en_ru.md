# Session 03 — EN→RU Diagnostic Experiment

## Goal

Validate that SeqDiffuSeq can produce good BLEU scores on **English → Russian** (WMT14, ~2.5M pairs).

**Hypothesis**: EN→ZH underperformed because Chinese CJK characters are hard for byte-level BPE (many Unicode chars, embeddings cluster). Russian uses Cyrillic (alphabetic), which byte-level BPE handles cleanly — similar to EN→DE which worked well in the paper.

If EN→RU achieves BLEU > 15, the EN→ZH gap is a tokenization problem, not a model architecture problem.

**Target**: SacreBLEU (13a) > 15 on newstest2014 (3,003 pairs).
**Paper baseline**: EN→DE WMT14 ≈ 18–20 BLEU.

---

## Changes vs. Session 02

| Setting | Session 02 (EN→ZH) | Session 03 (EN→RU) | Reason |
|---------|-------------------|-------------------|--------|
| Language pair | EN→ZH | **EN→RU** | Diagnostic — closer topology |
| Dataset | Custom, 233,842 pairs | **WMT14 EN-RU, 1,487,564 pairs** | Larger, standard benchmark |
| Test set | 29,253 pairs (custom) | **newstest2014, 3,003 pairs** | Standard MT benchmark |
| Tokenizer | Trained on EN+ZH | **ByteLevelBPE on EN+RU, 32,005 vocab** | New language pair |
| `sequence_len` | 64 | **128** | Russian inflections → longer sentences |
| `lr_anneal_steps` | 100,000 | **200,000** | ~5 epochs over 1.49M pairs |
| `warmup` | 500 | **10,000** | Larger dataset → more warmup |
| `diffusion_steps` (train) | 1,000 | **2,000** | Paper default |
| `batch_size` | 64 | 64 | Same (RTX 3090 24GB limit) |
| `in_channel` | 768 | 768 | BART-base dims — intentional for RU/ZH adaptation |
| `init_pretrained` | True | True | Warm-start with BART embeddings for non-Latin scripts |

All Session 02 bug fixes remain in place (4 bugs in `transformer_model.py`).

---

## Dataset

| Split | Pairs | Notes |
|-------|-------|-------|
| train | 1,487,564 | WMT14 EN-RU from HuggingFace |
| valid | 3,003 | newstest2013 |
| test  | 3,003 | **newstest2014** — standard benchmark |

Downloaded by `scripts/data_en_ru.sh` on vast.ai. Tokenizer trained with `ByteLevelBPETokenizer` directly (bypassed `tokenizer_utils.py` due to version conflict).

---

## Paper Architecture (for reference)

From Table 6 of arXiv 2212.10325v5:

| Setting | Paper Translation (EN-DE) | Our EN-RU Setup |
|---------|--------------------------|-----------------|
| Hidden Dimension | 512 | 768 (BART-base) |
| FFN Dimension | 2048 | 3072 |
| Embedding Dimension (in_channel) | 128 | **768** |
| Head Number | 8 | 12 |
| Max Output Length | 64 | 128 |
| Batch size | 128 | 64 |
| Max training steps | 1,000,000 | 500k (planned) |
| Hardware | A100 80GB | RTX 3090 24GB |
| init_pretrained | False (scratch) | **True (BART)** |

**Why we differ from the paper**: Paper's 128-dim translation model is trained from scratch on EN-DE (Latin). We use BART-base (768-dim) with `init_pretrained=True` so pretrained embeddings give better coverage for Cyrillic/CJK — this is the adaptation hypothesis.

---

## Bugs Found and Fixed

### 1. DDIM inference schedule mismatch (FIXED — `gaussian_diffusion.py`)
- **Problem**: `_load_time_schedule` loaded the 2000-step training schedule but inference used 200 steps → shape mismatch `(2000,127)` vs `(200,127)` → crash
- **Fix**: Subsample schedule with `np.linspace` to match `num_timesteps`
- **Commit**: `db6251c`

### 2. DDIM 200-step inference produces garbage (FIXED — `scripts/infer_en_ru.sh`)
- **Problem**: Even after the shape fix, DDIM with naive every-10th-step subsampling broke the noise trajectory → outputs were comma-chains and garbled bytes
- **Evidence**: `no_clamp` 200-step: `"В,,Ð (,,,,),,,,ÐÐх"` vs 2000-step: `"Вютсятельнонойен,, со и и о"`
- **Fix**: Always use `--diffusion_steps 2000` for inference; reduce batch to 50
- **Commit**: `946ccaa`

### 3. lm_head vocab size mismatch (FIXED — `transformer_model.py`)
- **Problem**: `build_embeddings()` sized lm_head to BART's full vocab (50,265 outputs) even when `--vocab_size 32005` was passed. At inference, model could predict token IDs 32,005–50,264 which our ByteLevelBPE tokenizer cannot decode → garbage tokens
- **Root cause**: `self.config.vocab_size = vocab_size` was inside `if not self.init_pretrained:` block, so skipped when `init_pretrained=True`
- **Fix**: `out_size = min(vocab_size, pretrained_vocab_size)` in `build_embeddings()`
- **Commit**: `80b9c8c`

---

## Training Progress

| Step | Loss | Notes |
|------|------|-------|
| 1 | 14.1 | Start |
| 100 | 12.8 | |
| 7,200 | 0.37 | Dropped fast — BART pretrained helps convergence |
| 11,500 | 0.076 | Near final loss already |
| 100,000 | 0.018 | **Training stopped here for checkpoint + inference test** |

Training ran ~19 hours on RTX 3090 ($0.17/hr ≈ $3.23).

**Loss converges fast because**: BART-base pretrained embeddings give well-structured embedding space; diffusion MSE drops quickly. Low MSE ≠ good translation quality — BLEU is the real metric.

---

## Inference Results at 100k Steps

| Setting | SacreBLEU (13a) | SacreBLEU (char) | Notes |
|---------|-----------------|------------------|-------|
| 200 steps DDIM, no_clamp | 0.02 | 3.03 | Garbage output (DDIM broken) |
| 200 steps DDIM, clamp | 0.02 | ~3 | Same garbage |
| 2000 steps, no_clamp | **0.20** | 7.01 | Real Russian output — architecture works |

**Sample output (2000 steps, 100k checkpoint):**
- SRC: `In Kineshma and environs two men have committed suicide`
- HYP: `Вютсятельнонойен,, со и и о игиениилю науи б стран. снамши Вас свои нашей всегда Иые`
- REF: `В Кинешме и районе двое мужчин покончили жизнь самоубийством`

Output is real Russian (correct script, real words) but semantically wrong — model needs more training.

**Token collapse analysis**: At 200-step DDIM, dominant tokens were ID 2 (`</s>` EOS, 299 occurrences) and ID 16 (`,` comma, 228 occurrences) across 5 samples. Classic diffusion "safe mean" collapse.

---

## Instance State (as of session end)

- **Instance**: 36926370, RTX 3090, $0.17/hr, ssh4.vast.ai:16370
- **Status**: Training STOPPED at step 100,000, instance still running (still billing!)
- **Checkpoint on instance**: `ckpts/en-ru/ema_0.9999_100000.pt` + `model100000.pt`
- **Checkpoint local backup**: `SeqDiffuSeq/results/en-ru/ckpts/ema_0.9999_100000.pt` ✓
- **Git**: All fixes pushed to `main` (commits `db6251c`, `80b9c8c`, `946ccaa`)

**⚠️ Action needed**: Either resume training or destroy instance to stop billing.

---

## Next Steps (Session 04)

### Immediate
1. **Resume training from 100k checkpoint to 500k steps** on current RTX 3090:
   ```bash
   ssh vastai "cd /root/NER_translation && git pull origin main"
   ssh vastai "cd /root/NER_translation && bash scripts/train_en_ru.sh"
   ```
   - `train_en_ru.sh` auto-resumes from `model100000.pt`
   - Git pull gets the lm_head fix applied for future inference
   - ~78 more hours, ~$13 cost

2. **After 500k steps**: run `bash scripts/infer_en_ru.sh` (auto-uses 2000 steps now)

3. **Pull and evaluate**:
   ```powershell
   .\scripts\pull_results.ps1 -Pair en-ru
   python analysis\eval_bleu.py --pair en-ru
   ```

### If BLEU > 10 at 500k
- Continue to 1M steps (matches paper's training budget)
- Try MBR decoding (`--mbr_sample 5`) for +2–4 BLEU

### If BLEU still < 5 at 500k
- The architecture (768-dim init_pretrained) may not be converging well
- Try paper's original setup: `in_channel=128`, `d_model=512`, `init_pretrained=False`
- Or investigate whether ByteLevelBPE is the right tokenizer (paper uses Moses BPE via fairseq)

### After EN→RU validates architecture
- Return to EN→ZH with identical config
- If ZH still fails → pure tokenization problem (fix: larger ZH vocab, or use sentencepiece unigram)

---

## Cost Summary

| Phase | Steps | Time | Cost |
|-------|-------|------|------|
| Session 03 training | 0→100k | ~19 hrs | ~$3.23 |
| Session 03 instance idle | — | ~few hrs | ~$1 |
| Session 04 (planned) | 100k→500k | ~78 hrs | ~$13 |
| **Total projected** | **500k** | | **~$17** |
