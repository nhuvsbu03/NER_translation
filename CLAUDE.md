# SeqDiffuSeq — NER Translation Project

## Project Goal

Train **SeqDiffuSeq** (arXiv 2212.10325), a diffusion-based seq2seq model, for machine translation.
Long-term target: **EN→ZH NER-aware translation** (named entities preserved in `[entity:TYPE]` format).
Current experiment: **ZH→EN** (WMT17, ~2.1M pairs, Session 04) — testing whether BART's English pretraining helps when English is the target.

## Workflow

```
1. Local (Windows)
   └── edit code → git commit → git push

2. vast.ai instance (SSH in)
   ├── scripts/push_vastai.ps1 -Pair zh-en  # git pull + push tokenizer files
   ├── bash scripts/vastai_setup.sh          # install deps + download BART (once per instance)
   ├── bash scripts/data_zh_en.sh            # download WMT17 ZH-EN + train tokenizer (once)
   ├── bash scripts/train_zh_en.sh           # launch training in tmux → returns immediately
   └── bash scripts/infer_zh_en.sh [STEP]   # run inference (optionally at specific step)

3. Local (pull results + analyze)
   ├── .\scripts\pull_results.ps1       # rsync results back from vast.ai
   └── python analysis/eval_bleu.py     # compute BLEU, save CSV + summary
```

### Starting a new vast.ai instance

```powershell
# 1. Start instance and update SSH config
bash scripts/start_vastai.sh

# 2. Install git + clone repo + push tokenizer
.\scripts\push_vastai.ps1

# 3. On vast.ai: install deps and download BART weights (takes ~5 min)
ssh vastai "cd /root/NER_translation && bash scripts/vastai_setup.sh"
```

### Monitoring training

```bash
# Attach to tmux session (detach with Ctrl+B then D)
tmux attach -t train

# Or tail the log file
tail -f /root/NER_translation/SeqDiffuSeq/ckpts/en-ru/log/train.log
```

## Key File Map

```
NER_translation/
├── CLAUDE.md                          ← this file
├── SeqDiffuSeq/                       ← model source (SeqDiffuSeq paper code + our fixes)
│   ├── main.py                        ← training entry point
│   ├── inference_main.py              ← inference entry point
│   ├── tokenizer_utils.py             ← BPE tokenizer (train_bytelevel / read_byte_level)
│   ├── args_utils.py                  ← all CLI args and defaults
│   ├── trainer.py                     ← training loop, EMA, checkpointing
│   ├── modeling_bart.py               ← custom BART encoder-decoder
│   ├── src/modeling/predictor/
│   │   └── transformer_model.py       ← TransformerNetModel_encoder_decoder (4 bugs fixed here)
│   ├── src/modeling/diffusion/
│   │   ├── gaussian_diffusion.py      ← noise schedule, forward/reverse process
│   │   └── rounding.py                ← continuous embedding → discrete token
│   ├── pretrained/bart-base/          ← BART-base weights (downloaded by vastai_setup.sh)
│   ├── data/en-ru/                    ← EN-RU data + tokenizer (created by data_en_ru.sh)
│   ├── data/en-zh/                    ← EN-ZH data + tokenizer (Session 01/02)
│   └── ckpts/                         ← checkpoints (excluded from git)
├── scripts/
│   ├── start_vastai.sh / .ps1         ← start instance, update SSH config
│   ├── push_vastai.sh / .ps1          ← git pull on instance + push tokenizer files
│   ├── vastai_setup.sh                ← install deps + BART weights (run once per instance)
│   ├── data_en_ru.sh                  ← download WMT14 EN-RU + train tokenizer
│   ├── train_en_ru.sh                 ← launch training in tmux
│   ├── infer_en_ru.sh                 ← run inference
│   └── pull_results.ps1               ← pull results from vast.ai to local
├── analysis/
│   └── eval_bleu.py                   ← compute SacreBLEU from inference output
├── train_dataset/                     ← local data (excluded from git, large files)
│   ├── train_clean.en / zh            ← 233,842 cleaned EN-ZH pairs (Session 01/02)
│   └── wmt14_en_ru/                   ← downloaded on vast.ai by data_en_ru.sh
└── plans/                             ← experiment logs
    ├── overview.md
    ├── session_01_baseline.md         ← EN→ZH collapse root cause
    ├── session_02_prep.md             ← fixes applied, bugs patched
    └── session_03_en_ru.md            ← current experiment
```

## Session History

### Session 01 — EN→ZH Baseline (FAILED)
- **Result**: Clamping collapse — output = "马斯马斯马斯..."
- **Root cause**: 10k BPE vocab too small for CJK → embeddings cluster at centroid → rounding always hits the same token
- **Training loss**: ~0.07 (looked fine, but outputs were garbage)

### Session 02 — EN→ZH Fixes Applied
- **Fixes**:
  1. BPE vocab 10k → **32k** (each CJK character gets its own token region)
  2. `init_pretrained=True` (load BART-base weights for structured embedding geometry)
  3. `in_channel` 512 → **768** (must match BART-base `d_model`; was loading silently wrong)
  4. `lr_anneal_steps` 30k → **100k**
  5. 4 bugs fixed in `transformer_model.py` (see below)
- **Status**: Training not completed / results pending

### Session 03 — EN→RU Diagnostic
- **Goal**: Validate SeqDiffuSeq works on a language closer to the paper's EN→DE
- **Dataset**: WMT14 EN-RU (~2.5M pairs, newstest2014 = 3,003 test pairs)
- **Result**: Architecture works — real Russian output confirmed at 100k steps

### Session 04 — ZH→EN (FAILED)
- **Result**: Collapsed, BLEU 0.01 — `"the the the..."`
- **Root cause**: BART encoder is English-pretrained; cannot represent Chinese source
- **Key insight**: BART only works when **source is English**

### Session 05 — EN→RU Production + CJK Diagnostic (CURRENT)
- **Plan 5_1**: EN→RU on A100 SXM4 40GB, batch=128, 500k steps (~$45 est.)
  - Instance: 39898796, launched 2026-06-07, loss healthy at step ~100
- **Plan 5_2**: 25k-step diagnostic for EN→ZH + EN→JA (scripts not yet created)
- **See**: `plans/session_05_en_ru_ja_kr.md`

## Known Bugs Already Fixed (DO NOT REINVESTIGATE)

All 4 bugs are in `SeqDiffuSeq/src/modeling/predictor/transformer_model.py`, in the `init_pretrained=True` path (never tested in original repo):

| Bug | Fix |
|-----|-----|
| `AttributeError: no attribute embedding_dim` | Added `else` branch: `self.embedding_dim = in_channels` |
| Projection condition wrong (`in_channels != embedding_dim`) | Changed to `config.d_model != embedding_dim` |
| `input_up_proj_dec` set to Identity when dims matched | Always build as `Linear(embedding_dim*2, d_model)` — self-conditioning always doubles decoder input |
| `from_pretrained` ignored `embedding_dim`, loaded 512-dim even with `in_channel=768` | Pass `embedding_dim=self.embedding_dim` to `from_pretrained()` |

## Infrastructure

- **GPU**: vast.ai A100 SXM4 40GB (~$1.07/hr) for production; RTX 3090 (~$0.17/hr) for cheap tests
- **SSH alias**: `vastai` (configured by `start_vastai.sh`)
- **Repo on instance**: `/root/NER_translation/` (git clone)
- **Data on instance**: `/root/NER_translation/SeqDiffuSeq/data/` (created by data scripts)
- **Checkpoints**: `/root/NER_translation/SeqDiffuSeq/ckpts/` (NOT in git)

## A100 Datacenter Setup — Known Issues (DO NOT REINVESTIGATE)

Datacenter A100 nodes differ from consumer RTX 3090/4090 in three ways.
All fixes are already baked into `scripts/vastai_setup.sh`.

| Error | Root cause | Fix (already in vastai_setup.sh) |
|-------|-----------|----------------------------------|
| `ImportError: libucc.so.1: undefined symbol` on `import torch` | HPC-X `/opt/hpcx/ucc/` conflicts with PyTorch bundled libs | `pip install --upgrade torch` |
| `MPI_Init_thread on a NULL communicator` at training start | `mpi4py` pip wheel links against system OpenMPI; datacenter uses HPC-X MPI at `/usr/local/mpi/` | Rebuild from source: `MPICC=/usr/local/mpi/bin/mpicc pip install --no-binary mpi4py mpi4py` |
| `tokenizers` build fails (no Rust compiler) | Old `transformers==4.18.0` pinned `tokenizers<0.13` which has no Python 3.12 wheel; A100 instances run Python 3.12 (vs Python 3.10 on RTX) | Updated to `transformers>=4.35,<4.45` + `tokenizers>=0.19` |

## A100 Training Config Constraints (DO NOT CHANGE WITHOUT TESTING)

| Setting | Value | Reason |
|---------|-------|--------|
| `--batch_size` | **128** | batch=256 OOMs on A100 40GB (backward pass needs ~37GB) |
| `--use_fp16` | **omit / False** | `TransformerNetModel_encoder_decoder.convert_to_fp16()` not implemented |
| TF32 speedup | **automatic** | PyTorch enables TF32 on Ampere by default → ~4× faster than RTX 3090 FP32, no flags needed |
| `--lr` | **1.5e-4** | sqrt(128/64) × 1e-4 scaling for batch=128 |
| `--lr_anneal_steps` | **500000** | 500k × 128 = 64M samples = paper's 1M × 64 |
