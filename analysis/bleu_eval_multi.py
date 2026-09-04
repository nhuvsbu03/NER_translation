"""
SacreBLEU comparison across one or more translation result CSVs, broken down by direction.

Input CSV schema: direction,idx,source,hypothesis,reference
  direction is e.g. "en-ru" (source_lang-target_lang)

Usage:
    python analysis/bleu_eval_multi.py --csv results_a.csv --csv results_b.csv
    python analysis/bleu_eval_multi.py --csv results_a.csv --labels "Model A"
"""
import argparse
import csv

from sacrebleu import corpus_bleu

DIRECTIONS = ["en-ru", "ru-en", "en-ja", "ja-en", "en-zh", "zh-en"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True, dest="csvs",
                     help="Result CSV (direction,idx,source,hypothesis,reference). Repeat for multiple files.")
    ap.add_argument("--labels", action="append", default=None,
                     help="Display label per --csv, in order. Defaults to the filename.")
    args = ap.parse_args()

    labels = args.labels or [c.split("/")[-1] for c in args.csvs]
    if len(labels) != len(args.csvs):
        raise SystemExit("--labels must be given once per --csv if used")

    results = {}
    for label, path in zip(labels, args.csvs):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_dir = {}
        for r in rows:
            by_dir.setdefault(r["direction"], []).append(r)
        results[label] = {}
        for d, rs in by_dir.items():
            hyps = [r["hypothesis"] for r in rs]
            refs = [r["reference"] for r in rs]
            bleu = corpus_bleu(hyps, [refs], tokenize="13a")
            results[label][d] = (bleu.score, len(rs))

    print(f"{'Direction':<10}" + "".join(f"{k[:28]:<30}" for k in labels))
    for d in DIRECTIONS:
        row = f"{d:<10}"
        for label in labels:
            score, n = results[label].get(d, (None, 0))
            row += f"{score:.2f} (n={n})".ljust(30) if score is not None else "N/A".ljust(30)
        print(row)


if __name__ == "__main__":
    main()
