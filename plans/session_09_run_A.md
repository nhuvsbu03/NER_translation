# Session 09 — Run A: Faithful Paper Recipe

## Research Question

**Does the SeqDiffuSeq paper's IWSLT config work on EN→RU?**

Session 08 trained without collapse but produced BLEU ≈ 0 (fluent Russian word-salad,
no source conditioning). CPU probing showed every component works — embeddings healthy,
rounding correct, cross-attention present — except the diffusion x0 prediction itself is
weak and source-agnostic (P1.1 Jaccard 0.31).

The leading hypothesis: Session 08 used a model that is **4× too narrow** (d_model=128 vs
512) and **4× too small a batch** (32 vs 128) compared to the paper's working recipe.
Run A tests this directly: copy the paper's `iwslt_en_de.sh` config exactly onto EN→RU
and see if source-conditioning emerges.

If Run A works → the cause of BLEU=0 was the config gap, and we have a path to production.
If Run A fails → the problem is deeper (in our fork's code/data pipeline), needs code investigation.

---

## Exact Experiment Setup

### Hardware
- **GPU**: NVIDIA GeForce RTX 5090 (32 GB VRAM)
- **Instance**: vast.ai 41901200
- **SSH**: `ssh -p 27316 root@47.184.161.176`
- **Cost**: ~$0.50–0.80/hr (RTX 5090 consumer tier)

### Dataset
- **Pair**: EN→RU
- **Train**: WMT14 EN-RU, **1,486,965 pairs** (at `/root/NER_translation/SeqDiffuSeq/data/en-ru/train.{en,ru}`)
- **Valid**: 3,000 pairs (newstest2013)
- **Test**: 3,003 pairs (newstest2014, standard benchmark)
- **Tokenizer**: 32k ByteLevelBPE (vocab.json + merges.txt in data/en-ru/)

### Model Architecture
Matches `train_scripts/iwslt_en_de.sh` from the upstream SeqDiffuSeq repo exactly:

| Parameter | Value | Note |
|-----------|-------|------|
| `--in_channel` | **512** | d_model of the diffusion transformer (was 128 in S08) |
| `--out_channel` | **512** | same |
| `--num_channels` | **2048** | FFN width (was 512 in S08) |
| `--num_heads` | **8** | attention heads |
| `--encoder_layers` | **6** | BART encoder |
| `--decoder_layers` | **6** | BART decoder |
| `--vocab_size` | 32005 | 32k BPE + 5 special tokens |
| `--init_pretrained` | **False** | random init, no BART weights (matches paper) |
| `--dropout` | 0.3 | |
| **Total params** | ~55M | confirmed in log |

### Diffusion Config
| Parameter | Value |
|-----------|-------|
| `--diffusion_steps` | 2000 |
| `--noise_schedule` | sqrt |
| `--predict_xstart` | True |
| `--schedule_update_stride` | **20000** | adaptive noise schedule update frequency (was 2000 in S08 — 10× wrong) |
| `--loss_update_granu` | 20 |
| `--schedule_sampler` | uniform |

### Training Config
| Parameter | Value | Note |
|-----------|-------|------|
| `--batch_size` | **128** | matches paper single-GPU recipe (was 32 in S08) |
| `--lr` | **1e-4** | paper LR (was 3.75e-5 in S08) |
| `--warmup` | 10000 | |
| `--lr_anneal_steps` | 1000000 | 1M steps cosine decay |
| `--sequence_len` | **64** | target length (was 128 in S08; p95 of RU newstest=60 so 64 is fine) |
| `--sequence_len_src` | 128 | source length |
| `--seed` | 42 | paper uses 10708 — minor difference, not restarted |
| `--save_interval` | 25000 | paper uses 10000 — only affects checkpoint frequency |

### Infrastructure
- **Launch script**: `scripts/train_en_ru_exp.sh A`
- **Checkpoint dir**: `SeqDiffuSeq/ckpts/en-ru-A/`
- **Log**: `ckpts/en-ru-A/log/train.log`
- **Auto BLEU eval**: `watch_checkpoints.sh` runs 10-sentence sample + 200-sentence BLEU at each checkpoint
- **Collapse guard**: `monitor_training.sh` kills training if grad_norm < 0.005 for 8/10 readings (after step 20k)

### Differences from Paper's iwslt_en_de.sh
| Param | Paper | Run A | Impact |
|-------|-------|-------|--------|
| seed | 10708 | 42 | negligible |
| save_interval | 10000 | 25000 | checkpoint freq only |
| dataset | IWSLT14 EN-DE (160k pairs) | WMT14 EN-RU (1.49M pairs) | intentional — testing if paper config generalises |
| config_name | `facebook/bart-base` (HF) | local `pretrained/bart-base` | same weights |

---

## What "Works" Means

Run A is considered **successful** if by step 50–100k:

1. **Quantitative**: BLEU (sacrebleu 13a, newstest2014) clearly rising:
   - > 3 at step 50k
   - > 8 at step 100k
   *(S08 baseline: 0.04 at step 800k — any positive trend is a signal)*

2. **Qualitative**: 10-sentence samples contain recognisable Russian words that correspond to
   the English source (not the same words regardless of input)

3. **Source conditioning probe P1.1**: Jaccard(true_source, shuffled_source) drops below 0.15
   *(S08 had 0.31 — meaning the model barely distinguished true from shuffled source)*

4. **No collapse**: grad_norm stays above 0.01 throughout
   *(collapse would mean the model converged to predicting zero embeddings)*

---

## Sanity Checks at Key Steps

### Step 2,500 (check within ~1.5 hrs of launch)
- `loss` should be dropping from ~20.9 toward < 10
- `grad_norm` should be in range 0.05–2.0
- If loss is stuck at ~20.9 or explodes: something is wrong with data loading

### Step 10,000 (first mini-sample from watch window)
- Sample output should contain Cyrillic characters (not Latin/empty/repeated)
- Loss should be around 3–8

### Step 25,000 (first checkpoint + BLEU eval)
- **First hard number**: 200-sentence BLEU from `bleu_step_025000.txt`
- Expected if working: > 1.0 (even 1 is far above S08's 0.04)
- If 0.0–0.1: still may be too early, check qualitative output

### Step 50,000 (~6–9 hrs in)
- Decision point: is BLEU rising and output coherent?
- If yes: continue to 100k for confirmation
- If no: move to code investigation (Run A fails → cause not config)

---

## Decision Outcome

| Run A result | Run B result | Conclusion | Next step |
|---|---|---|---|
| Works (BLEU rising, source-tracking) | Works | Capacity (128→512) was the sole cause | Session 10: production EN→RU run (WMT recipe, effective batch 1024 via grad-accum) |
| Works | Fails | Capacity + batch/seq/stride all matter | Session 10: isolate which of the other params matters |
| Fails | Fails | Not the config — code/data bug | Resume code investigation |

---

## Live Run Log

| Time | Step | Loss | grad_norm | Notes |
|------|------|------|-----------|-------|
| 2026-06-21 04:54 UTC | 0 | 20.9 | 1.48 | Training started, healthy init |
| 2026-06-21 04:54 UTC | 1 | 20.9 | 1.46 | Confirmed running |
| | 2500 | | | |
| | 25000 | | | First BLEU eval |
| | 50000 | | | Decision point |
| | 100000 | | | Final diagnostic |

---

## Pull Results Command (from Windows)

```powershell
# Pull checkpoint samples and BLEU files
scp -r "vastai:/root/NER_translation/SeqDiffuSeq/ckpts/en-ru-A/samples" SeqDiffuSeq\results\en-ru-A\

# Or tail the live log
ssh vastai "tail -f /root/NER_translation/SeqDiffuSeq/ckpts/en-ru-A/log/train.log"

# Check BLEU at a specific step
ssh vastai "cat /root/NER_translation/SeqDiffuSeq/ckpts/en-ru-A/samples/bleu_step_025000.txt"
```
