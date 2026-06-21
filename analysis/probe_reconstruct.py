"""Session 09 / P1.2 — low-noise reconstruction sweep (CPU).

Take a ground-truth Russian target, embed -> x0, forward-diffuse to t0, then run the
reverse denoiser from t0 down to 0 (with the TRUE source as conditioning). If the model
reconstructs the target well at low t0 but collapses at t0=2000, the denoiser works locally
and the failure is the global trajectory from pure noise (capacity at high noise / under-
training). If it fails even at low t0, the rounding/decode path is broken.

Run from SeqDiffuSeq/ :  python ../analysis/probe_reconstruct.py
"""
import os, sys, json, numpy as np, torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "SeqDiffuSeq"))
SESS8 = os.path.normpath(os.path.join(HERE, "..", "..", "session_8"))
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.chdir(REPO); sys.path.insert(0, REPO)

from model_utils import create_model_and_diffusion
from tokenizer_utils import create_tokenizer
from args_utils import model_and_diffusion_defaults
import sacrebleu

CKPT = os.path.join(SESS8, "checkpoints", "ema_0.9999_800000.pt")
SCHED = os.path.join(SESS8, "alpha_cumprod_step_98000.npy")
EN = os.path.join(REPO, "..", "analysis", "test.en")
RU = os.path.join(REPO, "..", "analysis", "test.ru")
N, SEQ = 2, 128
T0S = [50, 200, 500, 1000, 2000]

targs = json.load(open(os.path.join(SESS8, "training_args.json")))
targs.update(config_name=os.path.join(HERE, "_bart_cfg"), timestep_respacing="",
             resume_checkpoint="", load_ckpt=None, diffusion_steps=2000)
tok = create_tokenizer(return_pretokenized=False, path="data/en-ru/", tokenizer_type="byte-level")
PAD = tok.get_vocab()["<pad>"]
keys = list(model_and_diffusion_defaults().keys()) + [
    "sequence_len", "resume_checkpoint", "loss_update_granu", "schedule_update_stride"]
model, diffusion = create_model_and_diffusion(pad_tok_id=PAD, **{k: targs[k] for k in keys if k in targs})
diffusion._load_time_schedule(SCHED)
model.load_state_dict(th.load(CKPT, map_location="cpu")); model.eval()
EMB = model.input_transformers.shared.weight.shape[1]

src = [l.strip() for l in open(EN, encoding="utf-8")][:N]
tgt = [l.strip() for l in open(RU, encoding="utf-8")][:N]


def enc_ids(lines, n=SEQ):
    ids = th.ones(len(lines), n).long() * PAD
    mask = th.zeros(len(lines), n).long()
    for i, s in enumerate(lines):
        e = tok.encode(s).ids[:n]
        ids[i, :len(e)] = th.tensor(e); mask[i, :len(e)] = 1
    return ids, mask


def decode(sample):
    cands = th.topk(model.get_logits(sample), k=1, dim=-1).indices.squeeze(-1)
    out = []
    for seq in cands:
        seq = seq[seq > 2]
        out.append(tok.decode(seq.tolist(), skip_special_tokens=True))
    return out


src_ids, src_mask = enc_ids(src)
tgt_ids, _ = enc_ids(tgt)
with th.no_grad():
    x_start = model.get_embeds(tgt_ids)            # ground-truth target embedding (x0)


def reconstruct(t0):
    B = N
    mk = {"input_ids": src_ids, "attention_mask": src_mask,
          "decoder_attention_mask": th.ones(B, SEQ).long(), "self_conditions": None}
    t0v = th.tensor([t0 - 1] * B)
    img = diffusion.q_sample(x_start, t0v)         # forward-diffuse GT target to t0
    mk["self_conditions"] = th.zeros_like(img)
    with th.no_grad():
        enc = model.forward_encoder(decoder_inputs_embeds=img,
                                    timesteps=diffusion._scale_timesteps(th.tensor([10] * B)), **mk)
    mk["encoder_outputs"] = (enc,)
    mk.pop("input_ids"); mk.pop("self_conditions")
    for i in list(range(t0))[::-1]:
        t = th.tensor([i] * B)
        with th.no_grad():
            img = diffusion.p_sample(model, img, t, clip_denoised=False, model_kwargs=mk, top_p=-1.0)["sample"]
    return decode(img)


print("\n" + "=" * 70)
print("P1.2 RESULT — reconstruct GT target from t0 (with true source)")
print("=" * 70)
print(f"\nReference[0]: {tgt[0][:90]}")
for t0 in T0S:
    rec = reconstruct(t0)
    bleus = [sacrebleu.sentence_bleu(rec[i], [tgt[i]]).score for i in range(N)]
    print(f"\nt0={t0:>4}  mean sent-BLEU vs GT = {np.mean(bleus):.1f}")
    print(f"   recon[0]: {rec[0][:90]}")
print("\nHIGH BLEU at low t0 + LOW at t0=2000 => denoiser works locally; "
      "global sampling from noise is the failure (capacity/training).")
print("LOW BLEU even at small t0 => rounding/decode broken.")
