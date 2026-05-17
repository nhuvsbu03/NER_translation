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
| Dataset | Custom, 233,842 pairs | **WMT14 EN-RU, ~2.5M pairs** | Larger, standard benchmark |
| Test set | 29,253 pairs (custom) | **newstest2014, 3,003 pairs** | Standard MT benchmark |
| Tokenizer | Trained on EN+ZH | **Trained on EN+RU** | New language pair |
| `sequence_len` | 64 | **128** | Russian inflections → longer sentences |
| `sequence_len_src` | 128 | **128** | Same |
| `lr_anneal_steps` | 100,000 | **200,000** | ~5 epochs over 2.5M pairs |
| `warmup` | 500 | **10,000** | Larger dataset → more warmup |
| `diffusion_steps` | 1,000 (notebook) | **2,000** | Paper default |
| `batch_size` | 64 | 64 | Same |
| `in_channel` | 768 | 768 | Same (BART-base dims) |
| `init_pretrained` | True | True | Same |
| BLEU metric | SacreBLEU char (ZH) + 13a | **SacreBLEU 13a** (primary) | Standard for non-CJK |

All Session 02 bug fixes remain in place (4 bugs in `transformer_model.py`).

---

## Dataset

| Split | Pairs | Notes |
|-------|-------|-------|
| train | ~2,495,081 | WMT14 EN-RU from HuggingFace |
| valid | 3,003 | newstest2013 |
| test  | 3,003 | **newstest2014** — standard benchmark |

Downloaded by `scripts/data_en_ru.sh` directly on vast.ai from HuggingFace.

---

## Tokenizer

- Type: Byte-level BPE (32k vocab + 5 special tokens = 32,005)
- Trained on: `data/en-ru/train.en` + `data/en-ru/train.ru`
- Sanity check: `"Москва"` should tokenize to ≤ 3 tokens (Cyrillic is BPE-friendly)
- Files: `SeqDiffuSeq/data/en-ru/vocab.json`, `merges.txt`

---

## Training Milestones

| Step | Expected loss | Action |
|------|--------------|--------|
| 2,500 | < 8.0 | If > 8: check data paths, tokenizer |
| 25,000 | < 1.0 | Run one inference sample — output should be recognizable Russian |
| 100,000 | < 0.3 | Partial coherence, improving quality |
| 200,000 | ~0.07 | Full inference on newstest2014, compute BLEU |

---

## How to Run

```bash
# On vast.ai (from /root/NER_translation):
bash scripts/vastai_setup.sh      # once per instance
bash scripts/data_en_ru.sh        # once for EN-RU
bash scripts/train_en_ru.sh       # launches in tmux

# Monitor:
tail -f SeqDiffuSeq/ckpts/en-ru/log/train.log

# After training:
bash scripts/infer_en_ru.sh

# From Windows:
.\scripts\pull_results.ps1 -Pair en-ru
python analysis\eval_bleu.py --pair en-ru
```

---

## Results (fill in after run)

| Metric | Value |
|--------|-------|
| Steps completed | |
| Final training loss | |
| SacreBLEU (13a) | |
| SacreBLEU (char) | |
| Eval samples | 3,003 |
| Collapse observed? | |
| Sample output (good) | |
| Sample output (bad) | |

---

## Issues Encountered

_(fill in during/after run)_

---

## What to Try in Session 04

_(fill in after reviewing results)_

Ideas:
- If EN→RU BLEU > 15: return to EN→ZH with same config; consider sequence_len=128 for ZH
- If EN→RU BLEU < 10: model architecture issue — try `generate_by_q=True`, `clip_denoised=True`
- MBR decoding (`mbr_sample=5`) for +2–4 BLEU
- Compare against Helsinki-NLP/opus-mt-en-ru as oracle baseline
