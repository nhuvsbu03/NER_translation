# Session 05 — EN→RU + EN→JA + EN→KR (250k Steps × batch=256, Parallel)

## Context

Sessions 01–04 diagnosed that BART-base only works when English is the source.
Session 03 (EN→RU) confirmed the architecture is sound on alphabetic scripts.
Session 05 trains three pairs to 250k steps (= paper's 1M steps at batch=64, via batch scaling) in parallel:
- **EN→RU** (WMT14, ~2.5M pairs) — already proven, Cyrillic script
- **EN→JA** (WMT20, ~3.6M pairs) — Japanese, mixed CJK+kana, tests CJK decoder
- **EN→KR** (OPUS-100 en-ko, ~1M pairs) — Korean Hangul, alphabetic-like script

All three use English source → BART encoder works. Together they test Cyrillic, CJK, and
Hangul scripts. EN→DE (WMT14) can be added later as the paper's fourth benchmark if needed.

---

## Infrastructure: vast.ai (NOT Google Colab)

**Don't buy Colab Pro.** Training takes many hours per pair — Colab sessions disconnect after 24 hrs and don't guarantee A100.

Calibrated from Session 04: RTX 3090 at batch=64 FP32 → ~0.85 steps/sec.

**FP16 is supported** (`--use_fp16 True` flag exists in codebase) — use it on A100 for significant speedup.

### Scaling rule: steps scale inversely with batch size

To match the same total training data: **steps = target_samples / batch_size**
- Paper's full training: 1M steps × 64 = **64M samples**
- To match at batch=256: 64M / 256 = **250k steps**

### Recommended config: A100 40GB, batch=256, 250k steps, FP16

This is equivalent to the paper's full 1M-step run but faster.

**Why A100 over A40:** Diffusion model training is compute-bound (self-conditioning runs decoder
twice per step). A100 FP16 (312 TFLOPS) is ~2–3× faster than A40 (149 TFLOPS). Once inference
cost is included (~1–2 hrs per pair on A100 vs 3–5 hrs on A40), A40 costs the same but takes 3×
longer — A100 is the clear choice.

### Staged evaluation strategy

| Stage | Steps | Training/inst | Inference/inst | Total/inst | 3× total |
|-------|-------|--------------|----------------|------------|----------|
| Stage 1 | 100k | ~9 hrs | ~2 hrs | ~11 hrs | **~$60** |
| Stage 2 | +150k (→250k) | ~14 hrs | — | ~14 hrs | **~$75 more** |
| Full run | 250k | ~23 hrs | ~2 hrs | ~25 hrs | **~$135** |

**Strategy:**
1. Train to 100k steps (~$60 total) — verify all 3 pairs are non-collapsed, get preliminary BLEU
2. If EN→JA collapses: stop that instance, save money
3. Continue surviving pairs to 250k steps (~$75 more per surviving pair)

LR: `--lr 2e-4` (sqrt(256/64) × 1e-4 = linear scaling × 0.5 for stability)
`--lr_anneal_steps 250000`, `--warmup 5000`

---

## Files to Create

### EN→JA

#### `scripts/data_en_ja.sh`
Mirrors `scripts/data_en_ru.sh` pattern:
```bash
# Dataset: WMT20 ja-en (~3.6M train pairs)
# HuggingFace: load_dataset("wmt20", "ja-en")
#   → split["translation"]["en"] / ["ja"]
# Save to: SeqDiffuSeq/data/en-ja/{train,valid,test}.{en,ja}
#   train: wmt20 train split
#   valid: newstest2019 or wmt20 validation
#   test:  newstest2020
# Train 32k ByteLevelBPE on en+ja combined corpus
# Sanity check: tokenize "東京" (≤ 4 tokens = OK)
```

#### `scripts/train_en_ja.sh`
```bash
--src en --tgt ja
--dataset en-ja
--train_txt_path ./data/en-ja/train
--val_txt_path   ./data/en-ja/valid
--checkpoint_path ckpts/en-ja
--sequence_len 128 --sequence_len_src 128
--lr_anneal_steps 250000
--warmup 5000
--batch_size 256           # 4× batch → 4× fewer steps → same 64M samples as paper's 1M steps
--lr 2e-4                  # sqrt(256/64) × 1e-4
--use_fp16 True
--save_interval 25000      # 10 checkpoints across 250k steps
--vocab_size 32005
--in_channel 768 --out_channel 768 --num_channels 3072 --num_heads 12
--init_pretrained True
```

#### `scripts/infer_en_ja.sh`
Mirrors `scripts/infer_en_ru.sh` with en-ja paths.

---

### EN→KR

#### `scripts/data_en_kr.sh`
```bash
# Dataset: OPUS-100 en-ko (~1M train, 2k valid, 2k test pairs)
# HuggingFace: load_dataset("opus100", "en-ko")
#   → split["translation"]["en"] / ["ko"]
# Save to: SeqDiffuSeq/data/en-kr/{train,valid,test}.{en,kr}
# Train 32k ByteLevelBPE on en+ko combined corpus
# Sanity check: tokenize "서울" (≤ 4 tokens = OK)
```

#### `scripts/train_en_kr.sh`
```bash
--src en --tgt kr
--dataset en-kr
--train_txt_path ./data/en-kr/train
--val_txt_path   ./data/en-kr/valid
--checkpoint_path ckpts/en-kr
--sequence_len 128 --sequence_len_src 128
--lr_anneal_steps 250000
--warmup 5000
--batch_size 256
--lr 2e-4
--use_fp16 True
--save_interval 25000
--vocab_size 32005
--in_channel 768 --out_channel 768 --num_channels 3072 --num_heads 12
--init_pretrained True
```

#### `scripts/infer_en_kr.sh`
Mirrors `scripts/infer_en_ru.sh` with en-kr paths.

---

### EN→RU (update existing script)

`scripts/train_en_ru.sh` needs these changes:
- `--lr_anneal_steps 200000` → `250000`
- `--batch_size 64` → `256`
- Add `--lr 2e-4`
- Add `--use_fp16 True`
- `--save_interval` → `25000`

`scripts/data_en_ru.sh` — no changes needed (WMT14 already implemented).

---

## Staged Evaluation Workflow

```bash
# ── Setup: 3 separate vast.ai instances ────────────────────────────────────
.\scripts\push_vastai.ps1 -Pair en-ru   # on instance A
.\scripts\push_vastai.ps1 -Pair en-ja   # on instance B
.\scripts\push_vastai.ps1 -Pair en-kr   # on instance C

# ── On each instance: setup + data + train ─────────────────────────────────
bash scripts/vastai_setup.sh
bash scripts/data_en_ru.sh && bash scripts/train_en_ru.sh   # instance A
bash scripts/data_en_ja.sh  && bash scripts/train_en_ja.sh  # instance B
bash scripts/data_en_kr.sh  && bash scripts/train_en_kr.sh  # instance C

# ── Check at 100k steps: tmux attach -t train ──────────────────────────────
# If EN→JA loss is diverging or output is garbage → kill that instance

# ── After 250k steps ────────────────────────────────────────────────────────
bash scripts/infer_en_ru.sh   # instance A
bash scripts/infer_en_ja.sh   # instance B
bash scripts/infer_en_kr.sh   # instance C

# ── Pull + evaluate locally ─────────────────────────────────────────────────
.\scripts\pull_results.ps1 -Pair en-ru
.\scripts\pull_results.ps1 -Pair en-ja
.\scripts\pull_results.ps1 -Pair en-kr

python analysis\eval_bleu.py --pair en-ru
python analysis\eval_bleu.py --pair en-ja
python analysis\eval_bleu.py --pair en-kr
```

---

## Open Question: Paper Baselines

Current 3 pairs (RU, JA, KR) test multi-script capability but may not satisfy MT reviewers.
Standard MT papers almost always include **EN→DE WMT14** (the benchmark every prior work reports).

For a paper on diffusion-based NER-aware EN→ZH translation, consider adding:
- **EN→DE WMT14** (~4.5M pairs) — compare to SeqDiffuSeq paper's own reported numbers
- **EN→ZH WMT17/18** — directly test CJK decoder with Session 02 fixes; needed before NER phase

Decision deferred to Session 05+ review. If EN→JA succeeds (CJK decoder works), EN→ZH can follow.

---

## Risk Flag: EN→JA and CJK Decoder

Japanese output contains kanji (Chinese characters). This may hit the same decoder
collapse seen in EN→ZH Sessions 01–02. The 32k BPE + `init_pretrained=True` fixes
may be enough. If EN→JA collapses but EN→KR works, it confirms the CJK decoder
problem persists and Session 06 would need to revisit EN→ZH specifically.

---

## Verification

- `data_en_*.sh`: `wc -l data/en-*/train.en` → EN→RU ~2.5M, EN→JA ~3.6M, EN→KR ~1M
- `train_*.sh`: loss < 8.0 by step 2,500 on all 3 instances; check at step 100k before continuing
- After 250k steps: `tmux attach -t train` shows step ≥ 250,000
- Expected BLEU: EN→RU > 15, EN→KR > 10 (Hangul), EN→JA uncertain (CJK decoder)
- `eval_bleu.py` prints SacreBLEU (13a) for each pair
- FP16 stability check: if loss is NaN/Inf at step ~100, remove `--use_fp16 True` and rerun
