"""
Compute SacreBLEU scores from SeqDiffuSeq inference output files.

Usage:
    python analysis/eval_bleu.py --pair en-ru
    python analysis/eval_bleu.py --pair en-zh --results_dir SeqDiffuSeq/results
"""
import argparse
import csv
import glob
import json
import os
import re


def find_output_file(results_dir: str, pair: str) -> str:
    pattern = os.path.join(results_dir, pair, "inference_out", "ema_*.samples_*.txt")
    files = [f for f in glob.glob(pattern) if "raw-output-ids" not in f]
    if not files:
        raise FileNotFoundError(
            f"No inference output found at: {pattern}\n"
            f"Run inference first, then pull results with: .\\scripts\\pull_results.ps1 -Pair {pair}"
        )
    return sorted(files)[-1]


def find_source_file(results_dir: str, pair: str) -> str:
    src_lang = pair.split("-")[0]
    candidates = [
        os.path.join(results_dir, pair, "test." + src_lang),
        os.path.join(os.path.dirname(results_dir), "SeqDiffuSeq", "data", pair, "test." + src_lang),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="en-ru", help="Language pair (e.g. en-ru, en-zh)")
    parser.add_argument("--results_dir", default="SeqDiffuSeq/results",
                        help="Local results directory (default: SeqDiffuSeq/results)")
    args = parser.parse_args()

    try:
        import sacrebleu
    except ImportError:
        raise SystemExit("sacrebleu not installed. Run: pip install sacrebleu")

    output_file = find_output_file(args.results_dir, args.pair)
    print(f"Output file : {output_file}")

    with open(output_file, "r", encoding="utf-8") as f:
        pairs = [json.loads(line.strip()) for line in f if line.strip()]

    hypotheses = [p[0] for p in pairs]
    references  = [p[1] for p in pairs]

    # Try to load source sentences for the CSV
    src_file = find_source_file(args.results_dir, args.pair)
    if src_file:
        with open(src_file, encoding="utf-8") as f:
            sources = [line.strip() for line in f if line.strip()]
    else:
        sources = [""] * len(hypotheses)

    n = min(len(sources), len(hypotheses), len(references))
    sources, hypotheses, references = sources[:n], hypotheses[:n], references[:n]

    tgt_lang = args.pair.split("-")[1]
    tokenize = "char" if tgt_lang == "zh" else "13a"

    bleu_primary = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize)
    bleu_13a     = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="13a")
    bleu_char    = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="char")

    m = re.search(r"ema_[\d.]+_(\d+)", os.path.basename(output_file))
    step_tag = f"step{m.group(1)}" if m else "eval"

    out_dir = os.path.join(args.results_dir, args.pair, "inference_out")
    csv_path     = os.path.join(out_dir, f"{step_tag}.csv")
    summary_path = os.path.join(out_dir, f"{step_tag}_summary.txt")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        src_lang = args.pair.split("-")[0]
        writer.writerow([f"source_{src_lang}", f"hypothesis_{tgt_lang}", f"reference_{tgt_lang}"])
        for src, hyp, ref in zip(sources, hypotheses, references):
            writer.writerow([src, hyp, ref])

    summary_lines = [
        f"Pair            : {args.pair}",
        f"Output file     : {output_file}",
        f"Num samples     : {n}",
        "",
        f"SacreBLEU (13a) : {bleu_13a.score:.2f}",
        f"SacreBLEU (char): {bleu_char.score:.2f}",
        f"Primary metric  : {bleu_primary.score:.2f}  (tokenize={tokenize})",
    ]
    summary = "\n".join(summary_lines)

    print()
    print(summary)
    print()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"CSV     : {csv_path}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
