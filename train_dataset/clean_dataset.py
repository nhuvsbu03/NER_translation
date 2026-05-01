import os

DATA = os.path.dirname(os.path.abspath(__file__))

def cjk_ratio(text):
    if not text:
        return 0
    return sum(1 for c in text if '一' <= c <= '鿿') / max(len(text), 1)

def zh_char_count(text):
    return len([c for c in text if '一' <= c <= '鿿'])

def en_word_count(text):
    return len(text.split())

def classify(en, zh):
    if not zh:
        return "empty_zh"
    if cjk_ratio(zh) < 0.15 and len(zh) > 5:
        return "bad_lang_zh"
    e_len = max(en_word_count(en), 1)
    z_len = max(zh_char_count(zh), 1)
    if (e_len > 10 and z_len < 3) or (z_len > 10 and e_len < 3):
        return "extreme_ratio"
    return "ok"

for split in ["train", "valid", "test"]:
    with open(f"{DATA}/{split}.en") as f:
        en_lines = f.readlines()
    with open(f"{DATA}/{split}.zh") as f:
        zh_lines = f.readlines()

    n = min(len(en_lines), len(zh_lines))
    clean_en, clean_zh = [], []
    bad_en, bad_zh, bad_reasons = [], [], []

    for i in range(n):
        en = en_lines[i].rstrip('\n')
        zh = zh_lines[i].rstrip('\n')
        reason = classify(en, zh)
        if reason == "ok":
            clean_en.append(en + '\n')
            clean_zh.append(zh + '\n')
        else:
            bad_en.append(en + '\n')
            bad_zh.append(zh + '\n')
            bad_reasons.append(reason + '\n')

    with open(f"{DATA}/{split}_clean.en", 'w') as f:
        f.writelines(clean_en)
    with open(f"{DATA}/{split}_clean.zh", 'w') as f:
        f.writelines(clean_zh)
    with open(f"{DATA}/{split}_bad.en", 'w') as f:
        f.writelines(bad_en)
    with open(f"{DATA}/{split}_bad.zh", 'w') as f:
        f.writelines(bad_zh)
    with open(f"{DATA}/{split}_bad_reasons.txt", 'w') as f:
        f.writelines(bad_reasons)

    from collections import Counter
    reason_counts = Counter(bad_reasons)
    print(f"{split}: {n:,} total  →  clean={len(clean_en):,}  bad={len(bad_en):,}")
    for reason, count in sorted(reason_counts.items()):
        print(f"    {reason.strip()}: {count:,}")
