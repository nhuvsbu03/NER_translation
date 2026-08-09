"""Zero-shot translation with DiffusionGemma. Writes hypotheses + SacreBLEU report."""
import argparse, gc, os, sacrebleu, torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForMultimodalLM


def load_model(model_id, hf_token):
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    # DiffusionGemma 26B has MoE expert weights stored as batched nn.Parameter tensors
    # (shape [128, 1408, 2816]) which bitsandbytes cannot quantize directly.
    # Total BF16 footprint is ~47 GB; the A6000 has 47.4 GB, leaving no room for activations.
    # Strategy: use device_map="auto" with a 40 GB GPU budget so ~7 GB overflows to CPU.
    # Inference is slightly slower due to CPU↔GPU transfers on those layers, but it works.
    print(f"Loading {model_id} in BF16 with device_map=auto (40 GB GPU cap)…")
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "40GiB", "cpu": "400GiB"},
        token=hf_token,
    )
    model.eval()
    print(f"GPU memory after loading: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")
    return model, processor


def _infer_device(model):
    """Return the first real CUDA device used by the model (fallback: cuda:0)."""
    return next(
        (p.device for p in model.parameters() if p.device.type == "cuda"),
        torch.device("cuda:0"),
    )


def translate_one(model, processor, text, src_lang, tgt_lang, max_new_tokens=150):
    prompt = (
        f"Translate the following {src_lang} sentence to {tgt_lang}. "
        f"Output only the {tgt_lang} translation, nothing else.\n{text}"
    )
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(_infer_device(model))
    with torch.no_grad():
        # generate() returns DiffusionGemmaGenerationOutput; take .sequences
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

    full = processor.decode(out.sequences[0], skip_special_tokens=True)
    # Strip the echoed prompt section ("model\nthought\n" or "model\n")
    for marker in ("model\nthought\n", "model\n"):
        if marker in full:
            full = full.split(marker, 1)[1]
            break
    return full.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_file",  required=True, help="Source text file (one sentence per line)")
    ap.add_argument("--ref_file",  required=True, help="Reference translation file")
    ap.add_argument("--out_dir",   required=True, help="Output directory for hyps and report")
    ap.add_argument("--src_lang",  default="English")
    ap.add_argument("--tgt_lang",  default="Russian")
    ap.add_argument("--model_id",  default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--max_lines", type=int, default=None,
                    help="Limit lines for smoke test (e.g. --max_lines 5)")
    ap.add_argument("--max_new_tokens", type=int, default=150,
                    help="Max output tokens per sentence (reduce to lower peak VRAM)")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN not set. Set it with: export HF_TOKEN=hf_...")

    model, processor = load_model(args.model_id, hf_token)
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GB used")

    src = Path(args.src_file).read_text().splitlines()
    ref = Path(args.ref_file).read_text().splitlines()
    if args.max_lines:
        src, ref = src[:args.max_lines], ref[:args.max_lines]
    print(f"Translating {len(src)} lines ({args.src_lang} → {args.tgt_lang})…")

    hyps = [
        translate_one(model, processor, line, args.src_lang, args.tgt_lang,
                      max_new_tokens=args.max_new_tokens)
        for line in tqdm(src, desc="Translating")
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"hyps.{args.tgt_lang.lower()[:2]}").write_text("\n".join(hyps))

    bleu = sacrebleu.corpus_bleu(hyps, [ref], tokenize="13a")
    report = (
        f"SacreBLEU (13a): {bleu.score:.2f}\n"
        f"Model: {args.model_id}\n"
        f"Lines: {len(hyps)}\n"
        f"src_lang: {args.src_lang}\n"
        f"tgt_lang: {args.tgt_lang}\n"
    )
    (out_dir / "bleu_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
