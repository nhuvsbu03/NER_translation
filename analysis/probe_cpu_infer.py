"""Session 09 / P1.1 — source-conditioning ablation on CPU (the decisive behavioral test).

Holds the diffusion noise FIXED and varies only the source:
  (a) true source, (b) shuffled source (sentence i gets sentence j's source),
  (c) blank source (only <s></s>).
If the outputs barely change across (a)/(b)/(c), the decoder is NOT conditioning on the
source — it is an unconditional target LM. If outputs change with the source, conditioning
works and the failure is elsewhere.

Run from SeqDiffuSeq/ :  python ../analysis/probe_cpu_infer.py
"""
import os, sys, json, argparse, numpy as np, torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "SeqDiffuSeq"))
SESS8 = os.path.normpath(os.path.join(HERE, "..", "..", "session_8"))
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.chdir(REPO)                      # create_tokenizer uses relative data/<dataset>/
sys.path.insert(0, REPO)

from model_utils import create_model_and_diffusion
from tokenizer_utils import create_tokenizer

CKPT = os.path.join(SESS8, "checkpoints", "ema_0.9999_800000.pt")
SCHED = os.path.join(SESS8, "alpha_cumprod_step_98000.npy")
SRC_FILE = os.path.join(REPO, "..", "analysis", "test.en")
N = 4
SEQ = 128
DIFF_STEPS = 2000

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=N)
ap.add_argument("--steps", type=int, default=DIFF_STEPS)
args_cli = ap.parse_args()
N, DIFF_STEPS = args_cli.n, args_cli.steps

# ---- load training args, build model -------------------------------------
targs = json.load(open(os.path.join(SESS8, "training_args.json")))
targs["config_name"] = os.path.join(HERE, "_bart_cfg")     # local bart config
targs["diffusion_steps"] = DIFF_STEPS
targs["timestep_respacing"] = ""
targs["resume_checkpoint"] = ""
targs["load_ckpt"] = None

tok = create_tokenizer(return_pretokenized=False, path="data/en-ru/", tokenizer_type="byte-level")
PAD = tok.get_vocab()["<pad>"]

from args_utils import model_and_diffusion_defaults
keys = list(model_and_diffusion_defaults().keys()) + [
    "sequence_len", "resume_checkpoint", "loss_update_granu", "schedule_update_stride"]
mkw = {k: targs[k] for k in keys if k in targs}
model, diffusion = create_model_and_diffusion(pad_tok_id=PAD, **mkw)
diffusion._load_time_schedule(SCHED)
model.load_state_dict(th.load(CKPT, map_location="cpu"))
model.eval()
EMB = model.input_transformers.shared.weight.shape[1]
print(f"loaded model: emb_dim={EMB}, params={sum(p.numel() for p in model.parameters()):,}")

# ---- encode N source sentences -------------------------------------------
src_lines = [l.strip() for l in open(SRC_FILE, encoding="utf-8")][:N]


def encode_src(lines):
    ids = th.ones(len(lines), SEQ).long() * PAD
    mask = th.zeros(len(lines), SEQ).long()
    for i, line in enumerate(lines):
        e = tok.encode(line).ids[:SEQ]
        ids[i, : len(e)] = th.tensor(e)
        mask[i, : len(e)] = 1
    return ids, mask


def make_kwargs(lines):
    ids, mask = encode_src(lines)
    return {"input_ids": ids, "attention_mask": mask,
            "decoder_attention_mask": th.ones(len(lines), SEQ).long()}


def decode(sample):
    logits = model.get_logits(sample)
    cands = th.topk(logits, k=1, dim=-1).indices.squeeze(-1)
    out = []
    for seq in cands:
        seq = seq[seq > 2]
        out.append(tok.decode(seq.tolist(), skip_special_tokens=True))
    return out


def run(lines):
    th.manual_seed(42)
    noise = th.randn(N, SEQ, EMB)          # FIXED noise (same seed for every condition)
    sample = diffusion.p_sample_loop(
        model, (N, SEQ, EMB), noise=noise, clip_denoised=False,
        model_kwargs=make_kwargs(lines), top_p=-1.0, progress=True, tokenizer=tok)
    return decode(sample)

# ---- conditions -----------------------------------------------------------
true_src = src_lines
shuffled = src_lines[1:] + src_lines[:1]    # rotate: each sentence gets a different source
blank = ["" for _ in src_lines]

print("\n>>> running TRUE source"); out_true = run(true_src)
print("\n>>> running SHUFFLED source"); out_shuf = run(shuffled)
print("\n>>> running BLANK source"); out_blank = run(blank)


def content_ids(s):
    return set(tok.encode(s).ids)


def jacc(a, b):
    A, B = content_ids(a), content_ids(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


print("\n" + "=" * 70)
print("P1.1 RESULT — output vs source (fixed noise)")
print("=" * 70)
for i in range(N):
    print(f"\n[{i}] SRC      : {true_src[i][:80]}")
    print(f"    TRUE out : {out_true[i][:90]}")
    print(f"    SHUF out : {out_shuf[i][:90]}")
    print(f"    BLANK out: {out_blank[i][:90]}")
    print(f"    Jaccard(true,shuf)={jacc(out_true[i],out_shuf[i]):.2f}  "
          f"Jaccard(true,blank)={jacc(out_true[i],out_blank[i]):.2f}")
ts = np.mean([jacc(out_true[i], out_shuf[i]) for i in range(N)])
tb = np.mean([jacc(out_true[i], out_blank[i]) for i in range(N)])
print(f"\nMEAN Jaccard(true,shuffled)={ts:.3f}  Mean Jaccard(true,blank)={tb:.3f}")
print("HIGH (~>0.6) => output invariant to source => decoder IGNORES source.")
print("LOW  (~<0.3) => output tracks source => conditioning works.")
