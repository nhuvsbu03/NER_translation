"""
Named-entity translation accuracy across one or more translation result CSVs.

Compares hypothesis vs. reference (both target-language) using spaCy NER,
normalized to a consistent PERSON/ORG/LOC scheme across en/ru/ja/zh, with
one-to-one entity matching per sentence per type so missing entities count
as false negatives and extra/hallucinated entities count as false positives.

Filters to a "clean" subset first: sentences where source and reference agree
on entity count *and* type per category, since reference translations don't
always preserve every source entity 1:1 (pronominalization, paraphrase), which
otherwise pollutes the hypothesis-vs-reference comparison with reference noise.

Input CSV schema: direction,idx,source,hypothesis,reference
  direction is e.g. "en-ru" (source_lang-target_lang)

Usage:
    python analysis/ner_eval.py --csv results_a.csv --csv results_b.csv
    python analysis/ner_eval.py --csv results_a.csv --labels "Model A"
"""
import argparse
import csv
import difflib
import re
import time
from collections import Counter

import spacy

MODEL_MAP = {
    "en": "en_core_web_sm",
    "ru": "ru_core_news_sm",
    "ja": "ja_core_news_sm",
    "zh": "zh_core_web_sm",
}

LABEL_MAP = {
    "PERSON": "PERSON", "PER": "PERSON",
    "GPE": "LOC", "LOC": "LOC",
    "ORG": "ORG",
}

TYPES = ["PERSON", "ORG", "LOC"]
DIRECTIONS = ["en-ru", "ru-en", "en-ja", "ja-en", "en-zh", "zh-en"]


def norm(s):
    return re.sub(r"[^\w\s]", "", s).strip().lower()


def extract_entities(doc):
    ents = []
    for e in doc.ents:
        mapped = LABEL_MAP.get(e.label_)
        if mapped:
            ents.append((norm(e.text), mapped))
    return ents


def type_counts(ents):
    c = Counter(ty for _, ty in ents)
    return tuple(c.get(t, 0) for t in TYPES)


def sim(a, b):
    if not a or not b:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def one_to_one_match(ref_ents, hyp_ents, etype, threshold):
    """Greedy one-to-one bipartite matching for a single entity type.
    Returns (tp, fn, fp): matched pairs, missing ref entities, extra hyp entities."""
    refs = [t for t, ty in ref_ents if ty == etype]
    hyps = [t for t, ty in hyp_ents if ty == etype]
    if not refs and not hyps:
        return 0, 0, 0
    pairs = []
    for i, r in enumerate(refs):
        for j, h in enumerate(hyps):
            s = sim(r, h)
            if s >= threshold:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    used_ref, used_hyp = set(), set()
    tp = 0
    for s, i, j in pairs:
        if i in used_ref or j in used_hyp:
            continue
        used_ref.add(i)
        used_hyp.add(j)
        tp += 1
    fn = len(refs) - len(used_ref)
    fp = len(hyps) - len(used_hyp)
    return tp, fn, fp


def prf(tp, fn, fp):
    prec = tp / (tp + fp) * 100 if (tp + fp) else None
    rec = tp / (tp + fn) * 100 if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    acc = tp / (tp + fn + fp) * 100 if (tp + fn + fp) else None
    return prec, rec, f1, acc


def load_nlps():
    print("Loading spaCy models...")
    nlps = {}
    for lang, model in MODEL_MAP.items():
        nlps[lang] = spacy.load(model, disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
    return nlps


def find_clean_keys(base_csv, nlps, threshold):
    """Sentences where source and reference agree on entity count+type per category."""
    print("\nTagging source + reference to find clean (matching count/type) sentences...")
    with open(base_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_dir = {}
    for r in rows:
        by_dir.setdefault(r["direction"], []).append(r)

    clean_keys = set()
    for direction, rs in by_dir.items():
        src_code, tgt_code = direction.split("-")
        src_nlp, tgt_nlp = nlps[src_code], nlps[tgt_code]
        src_docs = list(src_nlp.pipe([r["source"] for r in rs], batch_size=256))
        ref_docs = list(tgt_nlp.pipe([r["reference"] for r in rs], batch_size=256))
        n_clean = 0
        for r, sdoc, rdoc in zip(rs, src_docs, ref_docs):
            if type_counts(extract_entities(sdoc)) == type_counts(extract_entities(rdoc)):
                clean_keys.add((direction, r["idx"]))
                n_clean += 1
        print(f"  {direction}: {n_clean}/{len(rs)} clean ({n_clean / len(rs) * 100:.1f}%)")

    print(f"\nTotal clean sentences: {len(clean_keys)}/{len(rows)} ({len(clean_keys) / len(rows) * 100:.1f}%)")
    return clean_keys


def evaluate_file(path, clean_keys, nlps, threshold):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r["direction"], r["idx"]) in clean_keys]

    by_dir = {}
    for r in rows:
        by_dir.setdefault(r["direction"], []).append(r)

    overall = {t: {"tp": 0, "fn": 0, "fp": 0} for t in TYPES}
    per_direction = {}

    for direction, rs in by_dir.items():
        tgt_code = direction.split("-")[1]
        nlp = nlps[tgt_code]
        hyp_docs = list(nlp.pipe([r["hypothesis"] for r in rs], batch_size=256))
        ref_docs = list(nlp.pipe([r["reference"] for r in rs], batch_size=256))
        per_direction[direction] = {t: {"tp": 0, "fn": 0, "fp": 0} for t in TYPES}
        for hdoc, rdoc in zip(hyp_docs, ref_docs):
            hyp_ents = extract_entities(hdoc)
            ref_ents = extract_entities(rdoc)
            for etype in TYPES:
                tp, fn, fp = one_to_one_match(ref_ents, hyp_ents, etype, threshold)
                for d in (overall[etype], per_direction[direction][etype]):
                    d["tp"] += tp
                    d["fn"] += fn
                    d["fp"] += fp

    return overall, per_direction, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True, dest="csvs",
                     help="Result CSV (direction,idx,source,hypothesis,reference). Repeat for multiple files.")
    ap.add_argument("--labels", action="append", default=None,
                     help="Display label per --csv, in order. Defaults to the filename.")
    ap.add_argument("--threshold", type=float, default=0.7,
                     help="Fuzzy string match threshold for entity matching (default 0.7)")
    args = ap.parse_args()

    labels = args.labels or [c.split("/")[-1] for c in args.csvs]
    if len(labels) != len(args.csvs):
        raise SystemExit("--labels must be given once per --csv if used")

    nlps = load_nlps()
    clean_keys = find_clean_keys(args.csvs[0], nlps, args.threshold)

    results, dir_results = {}, {}
    for label, path in zip(labels, args.csvs):
        t0 = time.time()
        overall, per_direction, n = evaluate_file(path, clean_keys, nlps, args.threshold)
        results[label] = overall
        dir_results[label] = per_direction
        print(f"{label} done in {time.time() - t0:.1f}s (n={n})")

    print(f"\n\n=== Accuracy on CLEAN subset (source/reference entity count+type match), n={len(clean_keys)} ===")
    for label in labels:
        print(f"\n{label}:")
        tot_tp = tot_fn = tot_fp = 0
        for etype in TYPES:
            d = results[label][etype]
            tp, fn, fp = d["tp"], d["fn"], d["fp"]
            tot_tp += tp; tot_fn += fn; tot_fp += fp
            prec, rec, f1, acc = prf(tp, fn, fp)
            if prec is not None:
                print(f"  {etype:<8} TP={tp:<5} FN={fn:<5} FP={fp:<5} "
                      f"Precision={prec:.1f}% Recall={rec:.1f}% F1={f1:.1f}% Accuracy={acc:.1f}%")
            else:
                print(f"  {etype:<8} no entities")
        prec, rec, f1, acc = prf(tot_tp, tot_fn, tot_fp)
        print(f"  {'ALL':<8} TP={tot_tp:<5} FN={tot_fn:<5} FP={tot_fp:<5} "
              f"Precision={prec:.1f}% Recall={rec:.1f}% F1={f1:.1f}% Accuracy={acc:.1f}%")

    print("\n\n=== Per-direction, per-type Accuracy (CLEAN subset) ===")
    for label in labels:
        print(f"\n{label}:")
        print(f"{'Direction':<10}" + "".join(f"{t:<20}" for t in TYPES))
        for d in DIRECTIONS:
            row = f"{d:<10}"
            for etype in TYPES:
                dd = dir_results[label].get(d, {}).get(etype, {"tp": 0, "fn": 0, "fp": 0})
                _, _, _, acc = prf(dd["tp"], dd["fn"], dd["fp"])
                n = dd["tp"] + dd["fn"]
                cell = f"{acc:.0f}% (n={n})" if acc is not None else "n/a"
                row += cell.ljust(20)
            print(row)


if __name__ == "__main__":
    main()
