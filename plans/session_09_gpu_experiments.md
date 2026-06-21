# Session 09 — GPU experiments to find the TRUE cause of EN→RU BLEU ≈ 0

## Context

Session 08 trained healthy (no collapse, grad_norm 0.7–1.4, loss ~1.1) but scored BLEU ≈ 0
(fluent Russian word-salad). CPU probing (see `session_09_diagnosis.md`) ruled OUT embedding
collapse, dead cross-attention, tokenizer/direction errors, and broken rounding (a correct
target embedding decodes back to the exact reference). It showed the model conditions on the
source only **weakly** (P1.1 Jaccard 0.31; P0.3 noise-dominated). The leading hypothesis is the
**config gap vs the paper's working recipe**, but this is **not proven** — CPU analysis cannot
settle it. We stay in Session 09 and run controlled **GPU experiments** on a rented
**RTX 5090 (32 GB)** to identify the true cause.

### Session 08 vs the paper's working EN→DE recipe (`train_scripts/iwslt_en_de.sh`, upstream)

| param | paper (works) | Session 08 (broken) |
|---|---|---|
| `in_channel`/`out_channel` → d_model | **512** | 128 |
| `num_channels` → FFN | **2048** | 512 |
| `batch_size` | **128** | 32 |
| `sequence_len` (target) | **64** | 128 |
| `schedule_update_stride` (adaptive noise schedule) | **20000** | 2000 |
| `lr` | **1e-4** | 3.75e-5 |
| `warmup` / `lr_anneal_steps` | 10000 / 1e6 | 10000 / 5e5→1e6 |

seq_len=64 is fine for RU: only 3.3% of newstest2014 targets exceed 64 tokens (p95=60).

## Experiment design — 2 runs to isolate the cause

**Run A — faithful paper recipe on EN-RU** (copy `iwslt_en_de.sh`, dataset = en-ru), from scratch:
```
--in_channel 512 --out_channel 512 --num_channels 2048 --num_heads 8
--encoder_layers 6 --decoder_layers 6 --batch_size 128
--sequence_len 64 --sequence_len_src 128 --lr 1e-4 --warmup 10000
--lr_anneal_steps 1000000 --schedule_update_stride 20000 --loss_update_granu 20
--dropout 0.3 --diffusion_steps 2000 --noise_schedule sqrt
--init_pretrained False --predict_xstart True --save_interval 10000 --vocab_size 32005
```

**Run B — capacity-only change from the S08 baseline** (change ONLY width; keep every other
S08 setting), from scratch:
```
--in_channel 512 --out_channel 512 --num_channels 2048      # changed
--batch_size 32 --sequence_len 128 --sequence_len_src 128   # kept from S08
--lr 3.75e-5 --schedule_update_stride 2000 --dropout 0.3 --init_pretrained False
```

### Decision matrix (how we name the cause)
| Run A (paper) | Run B (capacity-only) | Conclusion |
|---|---|---|
| works | works | **Capacity is the cause** — width 128→512 alone fixes it |
| works | fails | Capacity necessary but **not sufficient** — batch/seq_len/`schedule_update_stride` also matter; next isolate `schedule_update_stride` (S08 set it 10× too low) |
| fails | fails | **Not the config** — cause is in the fork/data pipeline (tokenizer, data, a bug); resume code-level investigation |

"Works" = by ~50–100k steps: inference is source-tracking Russian, BLEU clearly rising
(≫ S08's 0.04 — e.g. >3 at 50k, >8 at 100k), and the P1.1 CPU probe shows
Jaccard(true,shuffled) dropping well below S08's 0.31 (toward <0.15).

**Caveat for Run B:** at batch 32 it sees 4× less data per step than Run A — judge it by whether
source-conditioning *emerges at all* (output tracks source, BLEU rising, P1.1 Jaccard falling),
not by matching A's absolute BLEU. If ambiguous at 100k, extend to ~200–400k steps (matched
sample budget) before calling it a fail.

## Execution (reuse existing infra)

1. **Provision**: rent RTX 5090; `scripts/start_vastai.sh` → `scripts/push_vastai.ps1` →
   `ssh vastai "cd /root/NER_translation && bash scripts/vastai_setup.sh"`.
2. **Data** (once): `bash scripts/data_en_ru.sh` (downloads WMT14 en-ru, trains 32k BPE).
3. **Run scripts**: clone `scripts/train_en_ru.sh` → `train_en_ru_paperA.sh` and
   `train_en_ru_capB.sh` with the args above, each writing to a distinct
   `ckpts/en-ru-A/` / `ckpts/en-ru-B/`. Keep the tmux + `scripts/monitor_training.sh`
   collapse monitor + checkpoint watcher.
4. **Order**: A first (decisive "does the paper config work"); then B (isolate capacity).
   Parallel only if a 2nd instance is available.
5. **VRAM**: 32 GB should fit Run A at batch 128 (smaller than S08's failed 768-dim attempts).
   If OOM at step ~100, set `--microbatch 64` (grad-accum) to keep effective batch 128 —
   do NOT silently drop batch size (that adds a variable).

## Evaluation protocol (25k / 50k / 100k per run)
- `bash scripts/infer_en_ru.sh` with `--sequence_len` matching the run (A: 64, B: 128).
- `scripts/pull_results.ps1 -Pair en-ru`; `python analysis/eval_bleu.py --pair en-ru`.
- CPU validation on the pulled checkpoint: `analysis/probe_cpu_infer.py` (P1.1 source ablation)
  and `analysis/probe_checkpoint_weights.py` (embeddings/cross-attn/lm_head sanity).
  Run with `PYTHONUTF8=1 TRANSFORMERS_OFFLINE=1`.
- `monitor_training.sh` auto-kills on grad-norm collapse → no wasted spend.

## Timing (single RTX 5090, batch 128, paper model)
- Calibrate steps/sec from the first ~200 log lines. Expect ~**1.5–2.5 steps/s**.
- → **100k steps ≈ 11–18 h**; usable early readout at **50k ≈ 6–9 h**.
- Two runs sequentially to 100k ≈ **~1–1.5 days**. Kill a run early if its 25k/50k eval is
  clearly salad + collapse-flagged.

## Deliverable
Append results (BLEU@25k/50k/100k, P1.1 Jaccard, sample outputs) for Run A and Run B to
`session_09_diagnosis.md`, fill the decision matrix → a named, **proven** cause. Only after a
run demonstrates the cause do we open a Session 10 "fix / scale-up" plan.
