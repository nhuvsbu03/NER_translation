import os, time, csv, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()
MODEL_ID = "GoedelMachines/diffusiongemma-26B-A4B-w4a16"
LANG_NAMES = {"en": "English", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}
N_SENTENCES = 18
MAX_NEW_TOKENS = 80

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

print("Loading model (tier=turbo)...")
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
_ = translate_batch(sorted_srcs[:2], src_lang, tgt_lang, MAX_NEW_TOKENS)
torch.cuda.reset_peak_memory_stats()

results = {}
for group_size in (3, 6, 9):
    groups = [sorted_srcs[i:i+group_size] for i in range(0, len(sorted_srcs), group_size)]
    total_time = 0.0
    peak_mem = 0.0
    for g in groups:
        t0 = time.time()
        _ = translate_batch(g, src_lang, tgt_lang, MAX_NEW_TOKENS)
        dt = time.time() - t0
        total_time += dt
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(0) / 1e9)
    avg = total_time / len(sorted_srcs)
    results[group_size] = (avg, peak_mem)
    print(f"\nGroup size {group_size}: {len(groups)} batches, "
          f"{avg:.2f}s/sentence-equivalent, peak GPU mem {peak_mem:.1f}GB")

print("\n=== SUMMARY (sorted-length batching, MOE_NS on, max_new_tokens=80) ===")
for gs, (avg, mem) in results.items():
    print(f"  group_size={gs}: {avg:.2f}s/sentence-equivalent, peak {mem:.1f}GB")
