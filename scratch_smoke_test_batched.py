import os, time, csv, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()
MODEL_ID = "GoedelMachines/diffusiongemma-26B-A4B-w4a16"

LANG_NAMES = {"en": "English", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}

with open("/root/diffusiongemma_wmt_results.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = [next(reader) for _ in range(3)]

direction = rows[0]["direction"]
src_code, tgt_code = direction.split("-")
src_lang, tgt_lang = LANG_NAMES[src_code], LANG_NAMES[tgt_code]

print(f"Direction: {direction} ({src_lang} -> {tgt_lang})")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # required for batched causal generation to align correctly

print("Loading model (device_map=cuda, tier=turbo, single GPU)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
    device_map="cuda",
    tier="turbo",
)
model.eval()
print(f"  model loaded in {time.time()-t0:.1f}s")
print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")


def build_prompt(text, src_lang, tgt_lang):
    return (
        f"Translate the following {src_lang} sentence to {tgt_lang}. "
        f"Output only the {tgt_lang} translation, nothing else.\n{text}"
    )


def strip_marker(full):
    for marker in ("model\nthought\n", "model\n"):
        if marker in full:
            return full.split(marker, 1)[1].strip()
    return full.strip()


def translate_batch(texts, src_lang, tgt_lang, max_new_tokens=150):
    msgs_list = [[{"role": "user", "content": build_prompt(t, src_lang, tgt_lang)}] for t in texts]
    formatted = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in msgs_list
    ]
    enc = tokenizer(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
    seqs = out.sequences if hasattr(out, "sequences") else out
    return [strip_marker(tokenizer.decode(seqs[i], skip_special_tokens=True)) for i in range(seqs.shape[0])]


srcs = [r["source"] for r in rows]

print(f"\nRunning BATCHED smoke test ({len(srcs)} sentences in ONE generate() call)...")
t0 = time.time()
hyps = translate_batch(srcs, src_lang, tgt_lang)
dt = time.time() - t0

for i, (s, h) in enumerate(zip(srcs, hyps)):
    print(f"\n[{i}] SRC: {s}")
    print(f"[{i}] HYP: {h}")

print(f"\n=== Batch of {len(srcs)}: {dt:.2f}s total, {dt/len(srcs):.2f}s/sentence equivalent ===")

print("\nRunning a SECOND batch call (steady-state, no JIT warmup)...")
t0 = time.time()
hyps2 = translate_batch(srcs, src_lang, tgt_lang)
dt2 = time.time() - t0
print(f"=== Second batch call: {dt2:.2f}s total, {dt2/len(srcs):.2f}s/sentence equivalent ===")
