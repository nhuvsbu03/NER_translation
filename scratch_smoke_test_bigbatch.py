import os, time, csv, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()
MODEL_ID = "GoedelMachines/diffusiongemma-26B-A4B-w4a16"
LANG_NAMES = {"en": "English", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}
N_SENTENCES = 48
MAX_NEW_TOKENS = 150

with open("/root/diffusiongemma_wmt_results.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = [next(reader) for _ in range(N_SENTENCES)]

direction = rows[0]["direction"]
src_code, tgt_code = direction.split("-")
src_lang, tgt_lang = LANG_NAMES[src_code], LANG_NAMES[tgt_code]
print(f"Direction: {direction} ({src_lang} -> {tgt_lang}), {len(rows)} test sentences")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading model (vanilla, single 80GB GPU, no split needed, tier=turbo)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, token=HF_TOKEN, trust_remote_code=True, device_map="cuda", tier="turbo",
)
model.eval()
print(f"  model loaded in {time.time()-t0:.1f}s, GPU mem: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextExperts
import fused_moe_w4_v2
n = 0
for name, mod in model.named_modules():
    if isinstance(mod, DiffusionGemmaTextExperts):
        fused_moe_w4_v2.enable_ns_down(mod)
        n += 1
print(f"MOE_NS enabled on {n} expert modules")


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
sorted_srcs = sorted(srcs, key=len)

# Warmup (excluded from timing)
print("\nWarmup call...")
_ = translate_one(sorted_srcs[0], src_lang, tgt_lang, MAX_NEW_TOKENS)
torch.cuda.reset_peak_memory_stats()

# Sequential baseline on a subset (for reference, matches earlier measured ~8.36s number)
print("\n=== Sequential baseline (8 sentences) ===")
seq_times = []
for s in sorted_srcs[:8]:
    t0 = time.time()
    _ = translate_one(s, src_lang, tgt_lang, MAX_NEW_TOKENS)
    dt = time.time() - t0
    seq_times.append(dt)
seq_avg = sum(seq_times) / len(seq_times)
print(f"Sequential avg: {seq_avg:.2f}s/sentence")

results = {}
for group_size in (8, 16, 24):
    groups = [sorted_srcs[i:i+group_size] for i in range(0, len(sorted_srcs), group_size)]
    total_time = 0.0
    peak_mem = 0.0
    n_done = 0
    for g in groups:
        if len(g) < group_size:
            continue  # skip partial trailing group for clean comparison
        t0 = time.time()
        _ = translate_batch(g, src_lang, tgt_lang, MAX_NEW_TOKENS)
        dt = time.time() - t0
        total_time += dt
        n_done += len(g)
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(0) / 1e9)
    avg = total_time / n_done if n_done else float("nan")
    results[group_size] = (avg, peak_mem, n_done)
    print(f"\nGroup size {group_size}: {n_done} sentences processed, "
          f"{avg:.2f}s/sentence-equivalent, peak GPU mem {peak_mem:.1f}GB")

print("\n=== SUMMARY (W4A16, tier=turbo, MOE_NS on, single 80GB A100, mnt=150) ===")
print(f"  sequential:      {seq_avg:.2f}s/sentence")
for gs, (avg, mem, n) in results.items():
    speedup = seq_avg / avg if avg else float("nan")
    print(f"  batch={gs}:       {avg:.2f}s/sentence-equivalent ({speedup:.2f}x vs sequential), peak {mem:.1f}GB")
print(f"\n  (reference: plain BF16 steady-state was ~1.05s/sentence)")
