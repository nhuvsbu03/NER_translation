import os, time, csv, torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

HF_TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()
MODEL_ID = "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic"
LANG_NAMES = {"en": "English", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}
N_SENTENCES = 5

with open("/root/diffusiongemma_wmt_results.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = [next(reader) for _ in range(N_SENTENCES)]

direction = rows[0]["direction"]
src_code, tgt_code = direction.split("-")
src_lang, tgt_lang = LANG_NAMES[src_code], LANG_NAMES[tgt_code]
print(f"Direction: {direction} ({src_lang} -> {tgt_lang}), {len(rows)} test sentences")

print("\nLoading processor...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
print(f"  processor loaded in {time.time()-t0:.1f}s")

print("Loading model (FP8-dynamic, RedHatAI, device_map=auto)...")
t0 = time.time()
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)
model.eval()
load_time = time.time() - t0
print(f"  model loaded in {load_time:.1f}s")
print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")


def translate(text, src_lang, tgt_lang, max_new_tokens=150):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": f"Translate the following {src_lang} sentence to {tgt_lang}. "
                                  f"Output only the {tgt_lang} translation, nothing else.\n{text}"}
    ]}]
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    seqs = out.sequences if hasattr(out, "sequences") else out
    full = processor.decode(seqs[0][input_len:], skip_special_tokens=True)
    for marker in ("thought\n", "model\n"):
        if full.startswith(marker):
            full = full[len(marker):]
            break
    return full.strip()


srcs = [r["source"] for r in rows]

print(f"\nRunning {len(srcs)}-sentence timed test (FP8-dynamic on A100 -- expect no native accel)...")
times = []
for i, s in enumerate(srcs):
    t0 = time.time()
    hyp = translate(s, src_lang, tgt_lang)
    dt = time.time() - t0
    times.append(dt)
    print(f"[{i}] {dt:.2f}s | SRC: {s[:60]}")
    print(f"[{i}] {' '*7} HYP: {hyp[:60]}")

print(f"\n=== First call (incl. any warmup): {times[0]:.2f}s ===")
if len(times) > 1:
    steady = sum(times[1:]) / len(times[1:])
    print(f"=== Steady-state avg (calls 2-{len(times)}): {steady:.2f}s ===")
