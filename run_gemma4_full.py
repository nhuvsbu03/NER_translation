import os, csv, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu import corpus_bleu

HF_TOKEN = None
tok_path = os.path.expanduser("~/.hf_token")
if os.path.exists(tok_path):
    HF_TOKEN = open(tok_path).read().strip()
elif os.environ.get("HF_TOKEN"):
    HF_TOKEN = os.environ["HF_TOKEN"]

MODEL_ID = "google/gemma-4-26B-A4B-it"
LANG_NAMES = {"en": "English", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}
IN_CSV = "/root/diffusiongemma_wmt_results.csv"
OUT_CSV = "/root/diffusiongemma_wmt_results_gemma4.csv"
SUMMARY_PATH = "/root/bleu_summary_gemma4.txt"
MAX_NEW_TOKENS = 150
BATCH_SIZE = 16

with open(IN_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

directions = {}
for r in rows:
    directions.setdefault(r["direction"], []).append(r)

for d, rs in directions.items():
    print(f"{d}: {len(rs)} lines")

done = set()
write_header = not os.path.exists(OUT_CSV)
if not write_header:
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done.add((r["direction"], r["idx"]))
    print(f"Resuming: {len(done)} already done")

out_f = open(OUT_CSV, "a", newline="", encoding="utf-8")
writer = csv.DictWriter(out_f, fieldnames=["direction", "idx", "source", "hypothesis", "reference"])
if write_header:
    writer.writeheader()
    out_f.flush()

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


def build_prompt(text, src_lang, tgt_lang):
    msgs = [{"role": "user", "content":
             f"Translate the following {src_lang} sentence to {tgt_lang}. "
             f"Output only the {tgt_lang} translation, nothing else.\n{text}"}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def translate_batch(batch_rows, src_lang, tgt_lang):
    prompts = [build_prompt(r["source"], src_lang, tgt_lang) for r in batch_rows]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    seqs = out.sequences if hasattr(out, "sequences") else out
    hyps = []
    for i in range(len(batch_rows)):
        hyp = tokenizer.decode(seqs[i][input_len:], skip_special_tokens=True).strip()
        hyps.append(hyp)
    return hyps


summary_lines = []
overall_start = time.time()

for direction, rs in directions.items():
    src_code, tgt_code = direction.split("-")
    src_lang, tgt_lang = LANG_NAMES[src_code], LANG_NAMES[tgt_code]

    todo = [r for r in rs if (direction, r["idx"]) not in done]
    if not todo:
        print(f"{direction}: already fully done, skipping")
    else:
        print(f"\n=== {direction} ({src_lang} -> {tgt_lang}): {len(todo)} remaining of {len(rs)} ===")
        dir_start = time.time()
        n_done = 0
        for b in range(0, len(todo), BATCH_SIZE):
            batch_rows = todo[b:b + BATCH_SIZE]
            t0 = time.time()
            hyps = translate_batch(batch_rows, src_lang, tgt_lang)
            dt = time.time() - t0
            for r, hyp in zip(batch_rows, hyps):
                writer.writerow({
                    "direction": direction, "idx": r["idx"], "source": r["source"],
                    "hypothesis": hyp, "reference": r["reference"],
                })
            out_f.flush()
            n_done += len(batch_rows)
            elapsed = time.time() - dir_start
            print(f"  [{n_done}/{len(todo)}] batch of {len(batch_rows)} in {dt:.2f}s "
                  f"({dt/len(batch_rows):.2f}s/sentence)  (elapsed {elapsed/60:.1f}min)")

    # Compute BLEU for this direction from the output CSV (covers resumed + fresh rows)
    hyps_all, refs_all = [], []
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["direction"] == direction:
                hyps_all.append(r["hypothesis"])
                refs_all.append(r["reference"])
    if hyps_all:
        bleu = corpus_bleu(hyps_all, [refs_all], tokenize="13a")
        line = f"{direction}: BLEU {bleu.score:.2f} ({len(hyps_all)} lines)"
        print(line)
        summary_lines.append(line)

total_time = time.time() - overall_start
print(f"\n=== Total generation time: {total_time/60:.1f} min ===")

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write(f"Model: {MODEL_ID}\n")
    f.write(f"Total generation time: {total_time/60:.1f} min\n\n")
    for line in summary_lines:
        f.write(line + "\n")
    f.write(f"\nResults CSV: {OUT_CSV}\n")
    f.write(f"Summary: {SUMMARY_PATH}\n")

out_f.close()
print("\n=== Summary ===")
for line in summary_lines:
    print(line)
