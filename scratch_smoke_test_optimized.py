import os, time, csv, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()
MODEL_ID = "GoedelMachines/diffusiongemma-26B-A4B-w4a16"
LANG_NAMES = {"en": "English", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}

with open("/root/diffusiongemma_wmt_results.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = [next(reader) for _ in range(9)]

direction = rows[0]["direction"]
src_code, tgt_code = direction.split("-")
src_lang, tgt_lang = LANG_NAMES[src_code], LANG_NAMES[tgt_code]
print(f"Direction: {direction} ({src_lang} -> {tgt_lang}), {len(rows)} test sentences")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading model (tier=turbo)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, token=HF_TOKEN, trust_remote_code=True, device_map="cuda", tier="turbo",
)
model.eval()
print(f"  model loaded in {time.time()-t0:.1f}s, GPU mem: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

# --- Test 1: enable MOE_NS on every expert module ---
print("\n=== Test 1: enabling MOE_NS (nibble-preshuffled down-GEMM) ===")
try:
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextExperts
    import fused_moe_w4_v2
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, DiffusionGemmaTextExperts):
            fused_moe_w4_v2.enable_ns_down(mod)
            n += 1
    print(f"  MOE_NS enabled on {n} expert modules")
except Exception as e:
    print(f"  FAILED to enable MOE_NS: {e!r}")


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


def translate_one(text, src_lang, tgt_lang, max_new_tokens):
    msgs = [{"role": "user", "content": build_prompt(text, src_lang, tgt_lang)}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = enc["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens)
    seqs = out.sequences if hasattr(out, "sequences") else out
    return strip_marker(tokenizer.decode(seqs[0], skip_special_tokens=True))


def translate_batch(texts, src_lang, tgt_lang, max_new_tokens):
    msgs_list = [[{"role": "user", "content": build_prompt(t, src_lang, tgt_lang)}] for t in texts]
    formatted = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in msgs_list
    ]
    enc = tokenizer(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
    seqs = out.sequences if hasattr(out, "sequences") else out
    return [strip_marker(tokenizer.decode(seqs[i], skip_special_tokens=True)) for i in range(seqs.shape[0])]


srcs = [r["source"] for r in rows]

# Warmup call (JIT compile the NS kernel path) — excluded from timing
print("\nWarmup call (excluded from timing, compiles NS kernels)...")
_ = translate_one(srcs[0], src_lang, tgt_lang, 150)

# --- Test 1 continued: sequential, max_new_tokens=150, with NS enabled ---
print("\n=== Sequential, max_new_tokens=150, MOE_NS enabled ===")
times_150 = []
for s in srcs:
    t0 = time.time()
    hyp = translate_one(s, src_lang, tgt_lang, 150)
    dt = time.time() - t0
    times_150.append(dt)
    print(f"  {dt:.2f}s  | {hyp[:60]}")
avg_150 = sum(times_150) / len(times_150)
print(f"Avg (max_new_tokens=150): {avg_150:.2f}s/sentence")

# --- Test 2: sequential, max_new_tokens=80 ---
print("\n=== Sequential, max_new_tokens=80, MOE_NS enabled ===")
times_80 = []
for s in srcs:
    t0 = time.time()
    hyp = translate_one(s, src_lang, tgt_lang, 80)
    dt = time.time() - t0
    times_80.append(dt)
    print(f"  {dt:.2f}s  | {hyp[:60]}")
avg_80 = sum(times_80) / len(times_80)
print(f"Avg (max_new_tokens=80): {avg_80:.2f}s/sentence")

# --- Test 3: batched, sorted by length, groups of 3, max_new_tokens=80 ---
print("\n=== Batched (sorted by length, groups of 3), max_new_tokens=80 ===")
sorted_srcs = sorted(srcs, key=len)
groups = [sorted_srcs[i:i+3] for i in range(0, len(sorted_srcs), 3)]
total_batch_time = 0.0
for g in groups:
    t0 = time.time()
    hyps = translate_batch(g, src_lang, tgt_lang, 80)
    dt = time.time() - t0
    total_batch_time += dt
    print(f"  batch of {len(g)} (lens {[len(x) for x in g]}): {dt:.2f}s total, {dt/len(g):.2f}s/sentence")
    for s, h in zip(g, hyps):
        print(f"    {h[:60]}")
avg_batch = total_batch_time / len(sorted_srcs)
print(f"Avg (sorted batches of 3, max_new_tokens=80): {avg_batch:.2f}s/sentence-equivalent")

print("\n=== SUMMARY ===")
print(f"Sequential, mnt=150, NS on:        {avg_150:.2f}s/sentence")
print(f"Sequential, mnt=80,  NS on:        {avg_80:.2f}s/sentence")
print(f"Sorted-batch(3), mnt=80, NS on:    {avg_batch:.2f}s/sentence-equivalent")
