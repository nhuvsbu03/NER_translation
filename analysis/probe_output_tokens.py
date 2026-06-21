"""Session 09 / P0.1 + P0.3 — analyze Session 08 inference WITHOUT running the model.

P0.1  Output token-ID frequency + per-position entropy from the *.raw-output-ids* dumps.
      -> Is the output a few high-frequency tokens (mode collapse) or diverse?
P0.3  Same-source cross-run divergence: identical source, different num_samples runs.
      -> If the output changes a lot when only batch size changed, the output is driven
         by the noise draw, not the source (source-insensitivity signal).

Usage (from NER_translation/):
    python analysis/probe_output_tokens.py
"""
import json, glob, os, math, collections, re

INFER = os.path.join(os.path.dirname(__file__), "..", "..", "session_8", "inference")
INFER = os.path.normpath(INFER)
SPECIAL = {0, 1, 2}  # <s>, <pad>, </s>  (filtered by seq>2 before decode)


def load_raw(path):
    """Return list of hypothesis id-lists (first element of each [hyp, gt] line)."""
    hyps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hyp = json.loads(line)[0]
            hyps.append(hyp)
    return hyps


def p01_frequency():
    print("=" * 70)
    print("P0.1  Output token-ID frequency / entropy (content tokens only, excl 0/1/2)")
    print("=" * 70)
    for step in ("500000", "750000", "800000"):
        path = os.path.join(
            INFER,
            f"ema_0.9999_{step}.pt.samples_3003.steps-2000.clamp-no_clamp.raw-output-ids-normal_42.txt",
        )
        if not os.path.exists(path):
            continue
        hyps = load_raw(path)
        counter = collections.Counter()
        total_content = 0
        for h in hyps:
            for tid in h:
                if tid in SPECIAL:
                    continue
                counter[tid] += 1
                total_content += 1
        n_unique = len(counter)
        # entropy over the content-token distribution (bits)
        ent = -sum((c / total_content) * math.log2(c / total_content) for c in counter.values())
        top = counter.most_common(10)
        top_share = 100 * sum(c for _, c in top) / total_content
        avg_len = total_content / len(hyps)
        print(f"\nstep {step}: {len(hyps)} sentences, {total_content} content tokens, "
              f"avg content len {avg_len:.1f}")
        print(f"  unique content token IDs: {n_unique}   entropy: {ent:.2f} bits "
              f"(uniform over {n_unique} would be {math.log2(n_unique):.2f})")
        print(f"  top-10 IDs account for {top_share:.1f}% of all content tokens:")
        print(f"    {top}")


def p03_divergence():
    print("\n" + "=" * 70)
    print("P0.3  Same-source cross-run divergence (samples_10 vs samples_3003), step 800k")
    print("=" * 70)
    small = os.path.join(INFER, "ema_0.9999_800000.pt.samples_10.steps-2000.clamp-no_clamp.raw-output-ids-normal_42.txt")
    big = os.path.join(INFER, "ema_0.9999_800000.pt.samples_3003.steps-2000.clamp-no_clamp.raw-output-ids-normal_42.txt")
    if not (os.path.exists(small) and os.path.exists(big)):
        print("  (missing files)"); return
    hs = load_raw(small)
    hb = load_raw(big)
    n = min(len(hs), len(hb))

    def content(seq):
        return [t for t in seq if t not in SPECIAL]

    def jaccard(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0

    print(f"  comparing first {n} sentences (identical source, only batch size differs):")
    js = []
    exact = 0
    for i in range(n):
        a, b = content(hs[i]), content(hb[i])
        j = jaccard(a, b)
        js.append(j)
        if a == b:
            exact += 1
    print(f"    exact-match outputs: {exact}/{n}")
    print(f"    mean token-set Jaccard overlap: {sum(js)/len(js):.3f} (1.0 = identical sets)")
    print(f"    per-sentence Jaccard: {[round(x,2) for x in js]}")
    print("  -> low overlap => output driven by noise draw, not the source.")


if __name__ == "__main__":
    p01_frequency()
    p03_divergence()
