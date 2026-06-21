"""Session 09 / P1.4 (+ cross-attention probe) — inspect the trained checkpoint with
torch ONLY (no transformers / tokenizers / forward pass).

Answers two questions directly from the weights:
  1. Embedding geometry: with init_pretrained=False all token embeddings should have
     ~equal L2 norm (~0.02*sqrt(128)=0.23) and no degenerate cluster.
  2. Does the decoder actually read the encoder? Compare the decoder cross-attention
     (encoder_attn) weight magnitudes against the decoder self-attention. If cross-attn
     is collapsed (~0 relative to self-attn), the decoder cannot use the source.

Usage (from NER_translation/):
    python analysis/probe_checkpoint_weights.py [path_to_ema.pt]
"""
import sys, os, re, torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "session_8", "checkpoints", "ema_0.9999_800000.pt"))

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
print(f"checkpoint: {CKPT}")
print(f"tensors: {len(sd)}\n")


def norm(t):
    return t.float().norm().item()


# ---- 1. Embedding geometry -------------------------------------------------
print("=" * 70)
print("Embedding geometry (init_pretrained=False => expect ~equal norms)")
print("=" * 70)
emb_key = next((k for k in sd if k.endswith("shared.weight")), None)
lm_key = next((k for k in sd if "lm_head.weight" in k), None)
for name, key in (("token embedding (shared)", emb_key), ("lm_head", lm_key)):
    if key is None:
        print(f"  {name}: NOT FOUND"); continue
    w = sd[key].float()
    rn = w.norm(dim=-1)
    print(f"  {name} {tuple(w.shape)}: norm mean={rn.mean():.3f} std={rn.std():.3f} "
          f"min={rn.min():.3f} max={rn.max():.3f}")
    # special tokens 0,1,2
    print(f"      norms[0..4] = {[round(x,3) for x in rn[:5].tolist()]}")
# degenerate-cluster check on lm_head: how many rows have cosine>0.99 with the most common direction
if lm_key is not None:
    w = sd[lm_key].float()
    wn = torch.nn.functional.normalize(w, dim=-1)
    # sample 2000 rows for an O(n^2) cheap estimate of clustering via mean direction
    mean_dir = torch.nn.functional.normalize(wn.mean(0, keepdim=True), dim=-1)
    cos_to_mean = (wn @ mean_dir.T).squeeze()
    print(f"  lm_head cosine-to-mean-direction: mean={cos_to_mean.mean():.3f} "
          f"max={cos_to_mean.max():.3f}  (>0.9 count: {(cos_to_mean>0.9).sum().item()})")


# ---- 2. Cross-attention vs self-attention in the decoder -------------------
print("\n" + "=" * 70)
print("Decoder: cross-attention (encoder_attn) vs self-attention magnitudes")
print("(if cross-attn << self-attn, the decoder ignores the source)")
print("=" * 70)


def layer_attn_norms(kind):
    """Average over decoder layers of the summed q/k/v/out proj weight norms for `kind`."""
    pat = re.compile(rf"decoder\.layers\.(\d+)\.{kind}\.(q|k|v|out)_proj\.weight")
    by_layer = {}
    for k, v in sd.items():
        m = pat.search(k)
        if m:
            by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = norm(v)
    return by_layer


self_l = layer_attn_norms("self_attn")
cross_l = layer_attn_norms("encoder_attn")
if not cross_l:
    print("  no encoder_attn weights found (key naming differs?)")
else:
    print(f"  {'layer':>5} | {'self qkvo':>22} | {'cross qkvo':>22} | cross/self")
    for li in sorted(cross_l):
        s = self_l.get(li, {}); c = cross_l.get(li, {})
        ssum = sum(s.values()); csum = sum(c.values())
        ratio = csum / ssum if ssum else float("nan")
        print(f"  {li:>5} | "
              f"q{c and ''}{s.get('q',0):.2f} k{s.get('k',0):.2f} v{s.get('v',0):.2f} o{s.get('out',0):.2f} | "
              f"q{c.get('q',0):.2f} k{c.get('k',0):.2f} v{c.get('v',0):.2f} o{c.get('out',0):.2f} | {ratio:.2f}")
    # k/v are the ones that read encoder states; q/out are decoder-side
    kv_cross = sum(c.get('k',0)+c.get('v',0) for c in cross_l.values())
    kv_self = sum(s.get('k',0)+s.get('v',0) for s in self_l.values())
    print(f"\n  TOTAL cross-attn k+v norm = {kv_cross:.2f}   self-attn k+v norm = {kv_self:.2f}")
    print(f"  ratio cross/self (k+v) = {kv_cross/kv_self:.3f}")
    print("  k/v are the projections that actually read the ENCODER (source) states.")
