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

- **Instance**: 37613174, RTX 3090, $0.17/hr (switched from 4090 after dataset was reduced)
- **Dataset**: OPUS-100 zh-en (~1M train pairs, 2k valid, 2k test) — switched from WMT17 (15M pairs)
- **SSH alias**: `vastai` → `ssh2.vast.ai:13174` (destroyed after session)

---

## Training Progress

| Step | Loss | Notes |
|------|------|-------|
| ~100 | 13.6 | Training start, healthy loss |
| 50k  | 0.0146 | Still decreasing |
| 100k | 0.0139 | Stable, low loss |

Training was interrupted once (~71k) due to vast.ai disruption, restarted from 70k checkpoint.

---

## Inference Results

| Step | SacreBLEU (13a) | SacreBLEU (char) | Notes |
|------|-----------------|------------------|-------|
| 50k  | — | — | Failed: OOM (11 MPI workers × 3.7GB RAM each) |
| 100k | **0.01** | **0.01** | **COLLAPSE** — garbage output (see analysis below) |
| 200k | — | — | Cancelled after 100k failure |

---

## Root Cause Analysis — FAILURE

**Problem**: BART-base is English-only pretrained. It cannot encode Chinese source text.

- BART encoder receives Chinese tokens via our custom 32k ByteLevelBPE
- Encoder weights were pretrained for English — produce meaningless hidden states for Chinese input
- Diffusion decoder collapses to common English tokens: `"the the the the..."`, `"It, of."`, `". 16 16"`
- BLEU 0.01 = model learned nothing meaningful

**Why EN→RU (Session 03) worked**: BART encodes English source fluently → meaningful hidden states → decoder learns Russian.

**Why ZH→EN failed**: Same as EN→ZH but different bottleneck:
- EN→ZH: encoder OK, decoder can't produce CJK embeddings (10k BPE too small)
- ZH→EN: encoder can't represent Chinese, decoder collapses

**Conclusion**: The hypothesis was wrong. The bottleneck is the ENCODER language, not the DECODER language. BART only works well when the **source** is English.

---

## Saved Artifacts

| File | Size | Location |
|------|------|----------|
| `ema_0.9999_100000.pt` | 784MB | `SeqDiffuSeq/ckpts/zh-en/` (local) |
| `tokenizer_vocab.json` | 0.6MB | `SeqDiffuSeq/ckpts/zh-en/` (local) |
| `tokenizer_merges.txt` | 0.4MB | `SeqDiffuSeq/ckpts/zh-en/` (local) |
| Inference output | — | `SeqDiffuSeq/results/zh-en/inference_out_step100000/` |

`model100000.pt` — incomplete (instance destroyed mid-download, delete it).

---

## Cost Summary

| Phase | Notes | Cost (est.) |
|-------|-------|-------------|
| RTX 4090 (instance 37608290) | ~1 hr — destroyed after switching to smaller dataset | ~$0.60 |
| RTX 3090 (instance 37613174) | ~38 hrs total (training + inference) | ~$6.5 |
| **Total actual** | | **~$7.1** |
