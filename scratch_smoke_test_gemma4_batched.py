import os, time, csv, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = None
tok_path = os.path.expanduser("~/.hf_token")
if os.path.exists(tok_path):
    HF_TOKEN = open(tok_path).read().strip()
elif os.environ.get("HF_TOKEN"):
    HF_TOKEN = os.environ["HF_TOKEN"]

MODEL_ID = "google/gemma-4-26B-A4B-it"
LANG_NAMES = {"en": "English", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}
N_SENTENCES = int(os.environ.get("N_SENTENCES", 32))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))

with open("/root/diffusiongemma_wmt_results.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = [next(reader) for _ in range(N_SENTENCES)]

direction = rows[0]["direction"]
src_code, tgt_code = direction.split("-")
src_lang, tgt_lang = LANG_NAMES[src_code], LANG_NAMES[tgt_code]
print(f"Direction: {direction} ({src_lang} -> {tgt_lang}), {len(rows)} test sentences, batch_size={BATCH_SIZE}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (plain BF16, device_map=cuda)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, token=HF_TOKEN, trust_remote_code=True, dtype="auto", device_map="cuda",
)
model.eval()
print(f"  model loaded in {time.time()-t0:.1f}s")
print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")


def build_prompt(text):
    msgs = [{"role": "user", "content":
             f"Translate the following {src_lang} sentence to {tgt_lang}. "
             f"Output only the {tgt_lang} translation, nothing else.\n{text}"}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


srcs = [r["source"] for r in rows]
prompts = [build_prompt(s) for s in srcs]

all_hyps = []
batch_times = []
for b in range(0, len(prompts), BATCH_SIZE):
    batch = prompts[b:b + BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    dt = time.time() - t0
    batch_times.append(dt)
    seqs = out.sequences if hasattr(out, "sequences") else out
    for i in range(len(batch)):
        hyp = tokenizer.decode(seqs[i][input_len:], skip_special_tokens=True).strip()
        all_hyps.append(hyp)
    per_sentence = dt / len(batch)
    print(f"batch {b // BATCH_SIZE}: {len(batch)} sentences in {dt:.2f}s ({per_sentence:.2f}s/sentence)")

for i, (s, h) in enumerate(zip(srcs[:5], all_hyps[:5])):
    print(f"[{i}] SRC: {s[:60]}")
    print(f"[{i}] HYP: {h[:60]}")

total_time = sum(batch_times)
print(f"\n=== Total: {total_time:.2f}s for {len(prompts)} sentences ===")
print(f"=== Avg per sentence (batched): {total_time/len(prompts):.3f}s ===")
if len(batch_times) > 1:
    steady = sum(batch_times[1:]) / (len(prompts) - BATCH_SIZE)
    print(f"=== Steady-state avg per sentence (excl. first batch): {steady:.3f}s ===")
