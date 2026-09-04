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
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
print(f"  tokenizer loaded in {time.time()-t0:.1f}s")

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
load_time = time.time() - t0
print(f"  model loaded in {load_time:.1f}s")
print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")


def translate(text, src_lang, tgt_lang, max_new_tokens=150):
    prompt = (
        f"Translate the following {src_lang} sentence to {tgt_lang}. "
        f"Output only the {tgt_lang} translation, nothing else.\n{text}"
    )
    msgs = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = enc["input_ids"].to("cuda")
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens)
    seqs = out.sequences if hasattr(out, "sequences") else out
    full = tokenizer.decode(seqs[0], skip_special_tokens=True)
    for marker in ("model\nthought\n", "model\n"):
        if marker in full:
            full = full.split(marker, 1)[1]
            break
    return full.strip()


print("\nRunning 3-sentence timed smoke test (tier=turbo)...")
times = []
for i, row in enumerate(rows):
    t0 = time.time()
    hyp = translate(row["source"], src_lang, tgt_lang)
    dt = time.time() - t0
    times.append(dt)
    print(f"\n[{i}] SRC: {row['source']}")
    print(f"[{i}] HYP: {hyp}")
    print(f"[{i}] time: {dt:.2f}s")

print(f"\n=== First call (incl. Triton JIT warmup): {times[0]:.2f}s ===")
if len(times) > 1:
    steady = sum(times[1:]) / len(times[1:])
    print(f"=== Steady-state avg (calls 2-{len(times)}): {steady:.2f}s ===")
