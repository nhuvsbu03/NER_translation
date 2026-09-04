import time, csv, requests
from pathlib import Path
import sacrebleu

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
LANG_NAMES = {"en": "English", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}
MAX_NEW_TOKENS = 150

IN_CSV = "/root/diffusiongemma_wmt_results_input.csv"
OUT_CSV = "/root/diffusiongemma_wmt_results_nvfp4.csv"
SUMMARY_PATH = "/root/bleu_summary_nvfp4.txt"

by_direction = {}
with open(IN_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        by_direction.setdefault(row["direction"], []).append(row)

lines = {}
for direction, rows in sorted(by_direction.items()):
    rows = sorted(rows, key=lambda r: int(r["idx"]))
    src_code, tgt_code = direction.split("-")
    src_lang, tgt_lang = LANG_NAMES[src_code], LANG_NAMES[tgt_code]
    lines[direction] = (
        [r["source"] for r in rows], [r["reference"] for r in rows], src_lang, tgt_lang,
    )
    print(f"{direction} ({src_lang} -> {tgt_lang}): {len(rows)} lines")


def translate(text, src_lang, tgt_lang, max_new_tokens=MAX_NEW_TOKENS):
    prompt = (
        f"Translate the following {src_lang} sentence to {tgt_lang}. "
        f"Output only the {tgt_lang} translation, nothing else.\n{text}"
    )
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
    }
    resp = requests.post(API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


done = {}
if Path(OUT_CSV).exists():
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.setdefault(row["direction"], {})[int(row["idx"])] = row["hypothesis"]
    n_done = sum(len(v) for v in done.values())
    print(f"Resuming: found {n_done} already-translated lines in {OUT_CSV}")

file_mode = "a" if Path(OUT_CSV).exists() else "w"
write_header = not Path(OUT_CSV).exists()

run_start = time.time()
summary = []

with open(OUT_CSV, file_mode, newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow(["direction", "idx", "source", "hypothesis", "reference"])

    for direction, (src_lines, ref_lines, src_lang, tgt_lang) in lines.items():
        already = done.get(direction, {})
        hyps = [None] * len(src_lines)
        for idx, hyp in already.items():
            hyps[idx] = hyp

        todo = [i for i in range(len(src_lines)) if i not in already]
        print(f"\n=== {direction} ({src_lang} -> {tgt_lang}): {len(src_lines)} lines "
              f"({len(already)} done, {len(todo)} remaining) ===")

        for n, idx in enumerate(todo):
            t0 = time.time()
            hyp = translate(src_lines[idx], src_lang, tgt_lang)
            dt = time.time() - t0
            hyps[idx] = hyp
            writer.writerow([direction, idx, src_lines[idx], hyp, ref_lines[idx]])
            out_f.flush()
            if n % 25 == 0:
                elapsed = time.time() - run_start
                print(f"  [{n+1}/{len(todo)}] {dt:.2f}s  (elapsed {elapsed/60:.1f}min)")

        bleu = sacrebleu.corpus_bleu(hyps, [ref_lines], tokenize="13a")
        print(f"{direction}: SacreBLEU (13a) = {bleu.score:.2f}  ({len(hyps)} lines)")
        summary.append((direction, bleu.score, len(hyps)))

total_elapsed = time.time() - run_start
print(f"\n=== Total generation time: {total_elapsed/60:.1f} min ===")

print("\n=== Summary ===")
with open(SUMMARY_PATH, "w") as f:
    f.write(f"Model: {MODEL_ID}\n")
    f.write(f"Total generation time: {total_elapsed/60:.1f} min\n\n")
    for direction, score, n in summary:
        line = f"{direction}: BLEU {score:.2f} ({n} lines)"
        print(line)
        f.write(line + "\n")

print(f"\nResults CSV: {OUT_CSV}")
print(f"Summary: {SUMMARY_PATH}")
