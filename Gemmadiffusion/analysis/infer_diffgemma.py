"""Zero-shot translation with DiffusionGemma. Writes hypotheses + SacreBLEU report."""
import argparse, os, sacrebleu, torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig


def load_model(model_id, hf_token):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,  # allow overflow layers on CPU
    )
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    # max_memory lets accelerate offload the ~3GB overflow to CPU RAM;
    # the NF4 model is ~13GB so GPU gets nearly everything after loading.
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        max_memory={0: "20GiB", "cpu": "64GiB"},
        token=hf_token,
    )
    return model, processor


def translate_one(model, processor, text, src_lang, tgt_lang, max_new_tokens=200):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": f"Translate the following {src_lang} text to {tgt_lang}:\n{text}"}
    ]}]
    inputs = processor.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")  # device_map dispatch: always send inputs to first GPU
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(
        out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_file",  required=True, help="Source text file (one sentence per line)")
    ap.add_argument("--ref_file",  required=True, help="Reference translation file")
    ap.add_argument("--out_dir",   required=True, help="Output directory for hyps and report")
    ap.add_argument("--src_lang",  default="English")
    ap.add_argument("--tgt_lang",  default="Russian")
    ap.add_argument("--model_id",  default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--max_lines", type=int, default=None,
                    help="Limit lines for smoke test (e.g. --max_lines 20)")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN not set. Set it with: export HF_TOKEN=hf_...")

    print(f"Loading {args.model_id} with 4-bit quantization...")
    model, processor = load_model(args.model_id, hf_token)
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated(0)/1e9:.1f}GB used")

    src = Path(args.src_file).read_text().splitlines()
    ref = Path(args.ref_file).read_text().splitlines()
    if args.max_lines:
        src, ref = src[:args.max_lines], ref[:args.max_lines]
    print(f"Translating {len(src)} lines ({args.src_lang} → {args.tgt_lang})...")

    hyps = [
        translate_one(model, processor, line, args.src_lang, args.tgt_lang)
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
