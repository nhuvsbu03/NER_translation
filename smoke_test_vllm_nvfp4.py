import time, csv, requests

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
LANG_NAMES = {"en": "English", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}

with open("/root/diffusiongemma_wmt_results_input.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    row = next(reader)

direction = row["direction"]
src_code, tgt_code = direction.split("-")
src_lang, tgt_lang = LANG_NAMES[src_code], LANG_NAMES[tgt_code]
text = row["source"]

print(f"Direction: {direction} ({src_lang} -> {tgt_lang})")
print(f"SRC: {text}")

prompt = (
    f"Translate the following {src_lang} sentence to {tgt_lang}. "
    f"Output only the {tgt_lang} translation, nothing else.\n{text}"
)

payload = {
    "model": MODEL_ID,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 150,
}

t0 = time.time()
resp = requests.post(API_URL, json=payload, timeout=120)
dt = time.time() - t0

print(f"\nStatus: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    hyp = data["choices"][0]["message"]["content"]
    print(f"HYP: {hyp}")
else:
    print(f"ERROR: {resp.text[:2000]}")

print(f"\n=== Time for 1 sentence: {dt:.2f}s ===")
