# Session 10 — DiffusionGemma Zero-Shot Translation

## Goal
Evaluate Google's DiffusionGemma (released June 10, 2026) on the same WMT14 EN→RU test set (3003 sentences, newstest2014) used in Sessions 07–09, for a direct BLEU comparison against SeqDiffuSeq.

## Model
- **ID**: `google/diffusiongemma-26B-A4B-it`
- **Type**: Discrete token-masking diffusion LLM (not continuous Gaussian like SeqDiffuSeq)
- **Params**: 26B total, 4B active (Mixture of Experts)
- **License**: Apache 2.0
- **VRAM at NF4 4-bit**: ~13 GB quantized, but ~23.5 GB peak during loading
- **Prerequisite**: Accept Gemma 4 license at huggingface.co/google/diffusiongemma-26B-A4B-it

| | SeqDiffuSeq | DiffusionGemma |
|---|---|---|
| Diffusion type | Continuous Gaussian on BART embeddings | Discrete token masking → unmasking |
| Base model | BART-base 140M | Gemma 4 MoE 26B (3.8B active) |
| Training | Fine-tuned on MT pairs | Pretrained on 35+ languages |
| Translation | Task-specific training | Zero-shot instruction prompt |

## Code Location
`D:\Learning\Research\scr\NER_translation\Gemmadiffusion\`

```
Gemmadiffusion/
├── scripts/
│   ├── vastai_setup.sh       # install env + download all 5 test sets
│   ├── infer_en_ru.sh        # WMT14 EN→RU (smoke: pass 20 as $1)
│   ├── infer_en_zh.sh        # WMT17 EN→ZH
│   ├── infer_zh_en.sh        # WMT17 ZH→EN
│   ├── infer_en_ja.sh        # WMT20 EN→JA
│   ├── infer_ja_en.sh        # WMT20 JA→EN
│   ├── push_vastai.ps1       # rsync code → vastai
│   └── pull_results.ps1      # rsync results/ ← vastai
├── analysis/
│   └── infer_diffgemma.py    # inference + SacreBLEU (13a)
└── notebooks/
    └── diffgemma_translation.ipynb  # Kaggle 2×T4 version
```

## Infrastructure
- **vast.ai instance**: `vastai-4090` (38.117.87.44:49874) — RTX 4090 24GB
- **Kaggle**: 2×T4 (32GB total) — notebook running in parallel
- HF token stored at `~/.hf_token` on each instance (not in code)

## Inference Script Key Config
```python
# infer_diffgemma.py — current working config
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto",
    max_memory={0: "20GiB", "cpu": "64GiB"},
    token=hf_token,
)
# inputs go to "cuda:0" explicitly (not model.device which returns "meta")
```

Shell env required before Python:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## RTX 4090 — OOM Root Cause Analysis

| Attempt | Config | Result |
|---------|--------|--------|
| 1 | `device_map="auto"` only | ValueError: modules dispatched to CPU (no cpu_offload flag) |
| 2 | `device_map="auto"` + `max_memory={"cuda:0": "22GiB"}` | ValueError: "cuda:0" key not recognized (must be int) |
| 3 | `device_map={"": 0}` (force all GPU) | OOM: 22.71GB in use, 817MB free, needed 968MB |
| 4 | `device_map={"": 0}` + `double_quant=False` | OOM: 22.66GB in use, 475MB free (double_quant=False uses MORE memory!) |
| 5 | `device_map="auto"` + `max_memory={0:"20GiB","cpu":"64GiB"}` + `llm_int8_enable_fp32_cpu_offload=True` | **Model loaded** ✓ but inference fails: `Tensor.item() on meta` |
| 6 | Same as 5 + `inputs.to("cuda:0")` | **Model loaded** ✓ but inference fails: `cuda:0 not on expected device meta` |

**Root cause of inference failures**: DiffusionGemma's block-diffusion generation code creates internal tensors on `model.device` (which returns `meta` with dispatch). These meta tensors conflict with the cuda:0 inputs during generation. The model assumes single-device execution; CPU offload via accelerate dispatch is fundamentally incompatible.

**The 4090 is 9 MiB short** — the model needs 23.524 GiB peak during loading; the 4090 has 23.52 GiB.

## Next Steps (Session 11)

- [ ] **Option A**: Check Kaggle 2×T4 results — with `device_map="auto"` across two CUDA GPUs (no CPU layers), the meta tensor issue should not occur. 32GB > 23.5GB loading peak.
- [ ] **Option B**: Rent A100 40GB on vast.ai. Loads clean with `device_map={"": 0}`, no dispatch needed.
- [ ] Once smoke test (20 lines) passes: run full EN→RU (3003 lines) → get BLEU
- [ ] Run all 5 pairs: EN↔RU, EN↔ZH, EN↔JA
- [ ] Compare DiffusionGemma BLEU vs SeqDiffuSeq Session 07–09 results

## Key Insight for Next Session
`double_quant=True` uses LESS memory than `double_quant=False` (counterintuitive).
The `expandable_segments:True` env var works (reduces reserved-but-unallocated from 737MB to 10MB) but doesn't free enough for the 968MB needed.
