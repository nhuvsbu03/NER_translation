"""Zero-shot translation with DiffusionGemma W4A16 quantized model.

Uses GoedelMachines/diffusiongemma-26B-A4B-w4a16 — a Triton-kernel W4A16
quantization that properly compresses the batched MoE expert tensors.
GPU footprint: ~11.5 GB vs ~40 GB for the BF16 version.

Writes hypotheses + SacreBLEU report identical to infer_diffgemma.py.
"""
import argparse, gc, os, sacrebleu, torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID_W4A16 = "GoedelMachines/diffusiongemma-26B-A4B-w4a16"


def load_model(model_id, hf_token):
    # W4A16 model ships with custom Triton kernels via trust_remote_code=True.
    # No device_map needed — the whole model fits in ~11.5 GB.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, trust_remote_code=True
    )
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Loading {model_id} (W4A16, ~11.5 GB)…")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
        device_map="cuda",
    )
    model.eval()
    print(f"GPU memory after loading: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")
    return model, tokenizer


def translate_one(model, tokenizer, text, src_lang, tgt_lang, max_new_tokens=150):
    prompt = (
        f"Translate the following {src_lang} sentence to {tgt_lang}. "
        f"Output only the {tgt_lang} translation, nothing else.\n{text}"
    )
    msgs = [{"role": "user", "content": prompt}]

    # Apply chat template — fall back to plain tokenization if unavailable
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens)

    # DiffusionGemmaGenerationOutput has .sequences; plain tensor also works
    seqs = out.sequences if hasattr(out, "sequences") else out
    full = tokenizer.decode(seqs[0], skip_special_tokens=True)

    # Strip echoed prompt section ("model\nthought\n" or "model\n")
    for marker in ("model\nthought\n", "model\n"):
        if marker in full:
            full = full.split(marker, 1)[1]
            break
    return full.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_file",  required=True)
    ap.add_argument("--ref_file",  required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--src_lang",  default="English")
    ap.add_argument("--tgt_lang",  default="Russian")
    ap.add_argument("--model_id",  default=MODEL_ID_W4A16)
    ap.add_argument("--max_lines", type=int, default=None,
                    help="Limit lines for smoke test (e.g. --max_lines 5)")
    ap.add_argument("--max_new_tokens", type=int, default=150)
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN not set.")

    model, tokenizer = load_model(args.model_id, hf_token)
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GB used")

    src = Path(args.src_file).read_text().splitlines()
    ref = Path(args.ref_file).read_text().splitlines()
    if args.max_lines:
        src, ref = src[:args.max_lines], ref[:args.max_lines]
    print(f"Translating {len(src)} lines ({args.src_lang} → {args.tgt_lang})…")

    hyps = [
        translate_one(model, tokenizer, line, args.src_lang, args.tgt_lang,
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
