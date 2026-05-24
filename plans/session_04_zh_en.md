# Session 04 — ZH→EN Experiment

## Goal

Train SeqDiffuSeq on **Chinese → English** (WMT17, ~2.1M pairs) with staged evaluation at 50k, 100k, and 200k steps.

**Hypothesis**: BART-base is English-pretrained, so having English as the *target* (decoder side) should give better BLEU than EN→ZH where the decoder had to produce CJK. This tests whether direction matters independently of language pair difficulty.

**Target**: SacreBLEU (13a) > 5 at 50k, improving across milestones.

---

## Changes vs. Session 03 (EN→RU)

| Setting | Session 03 (EN→RU) | Session 04 (ZH→EN) | Reason |
|---------|-------------------|-------------------|--------|
| Language pair | EN→RU | **ZH→EN** | Test direction hypothesis |
| Dataset | WMT14 EN-RU, 1.49M | **WMT17 ZH-EN, ~2.1M** | Standard ZH-EN benchmark |
| Test set | newstest2014, 3,003 | **newstest2017, 2,001** | Standard ZH-EN test set |
| Tokenizer | ByteLevelBPE on EN+RU | **ByteLevelBPE on ZH+EN** | New language pair |
| `--src / --tgt` | en / ru | **zh / en** | Direction flip |
| `lr_anneal_steps` | 200,000 | 200,000 | Same scale |
| `warmup` | 10,000 | 10,000 | Same |
| Eval strategy | Single run at end | **50k → 100k → 200k staged** | Track improvement |
| GPU | RTX 3090 ($0.17/hr) | **RTX 4090 ($0.60/hr)** | ~2x faster, similar total cost |

All Session 02/03 bug fixes remain (4 bugs in `transformer_model.py`, lm_head fix, DDIM fix).

---

## Dataset

| Split | Pairs | Notes |
|-------|-------|-------|
| train | ~2,141,491 | WMT17 ZH-EN from HuggingFace |
| valid | 1,999 | newstest2016 |
| test  | 2,001 | **newstest2017** — standard benchmark |

Downloaded by `scripts/data_zh_en.sh` on vast.ai.

---

## Infrastructure

- **Instance**: 37608290, RTX 4090, $0.60/hr
- **SSH alias**: `vastai`

---

## Training Progress

| Step | Loss | Notes |
|------|------|-------|
| — | — | — |

---

## Inference Results

| Step | SacreBLEU (13a) | SacreBLEU (char) | Notes |
|------|-----------------|------------------|-------|
| 50k  | TBD | TBD | |
| 100k | TBD | TBD | |
| 200k | TBD | TBD | |

**Paper baseline (EN→DE, WMT14)**: ~18–20 BLEU. Our target > 5 at 50k to confirm learning.

---

## Bugs Found and Fixed

_(none yet — to be filled during session)_

---

## Cost Summary

| Phase | Steps | Time (est.) | Cost (est.) |
|-------|-------|-------------|-------------|
| Training | 0→200k | ~19 hrs (RTX 4090) | ~$11.4 |
| Inference ×3 | — | ~1.5 hrs total | ~$0.9 |
| **Total projected** | | | **~$12.3** |
