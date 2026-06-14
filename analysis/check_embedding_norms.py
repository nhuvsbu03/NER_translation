"""
Zero-attractor diagnostic: check BART embedding norms in a trained checkpoint.

If BOS/PAD/EOS token norms are much smaller than the overall mean,
the zero-embedding attractor is confirmed (MSE minimum → special tokens → empty output).

Usage (from NER_translation/SeqDiffuSeq/):
    python3 ../analysis/check_embedding_norms.py ckpts/en-ru/model110000.pt
"""
import sys
import torch

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 check_embedding_norms.py <model_checkpoint.pt>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    print(f"Loading {ckpt_path} on CPU...")
    m = torch.load(ckpt_path, map_location='cpu')

    # Find embedding weights — key depends on model structure
    emb_key = 'input_transformers.shared.weight'
    if emb_key not in m:
        # Try decoder embed_tokens
        emb_key = 'input_transformers.decoder.embed_tokens.weight'
    if emb_key not in m:
        print("Available keys containing 'embed':")
        for k in m.keys():
            if 'embed' in k.lower():
                print(f"  {k}: {m[k].shape}")
        sys.exit(1)

    emb = m[emb_key]  # [vocab_size, embedding_dim]
    print(f"\nEmbedding matrix shape: {emb.shape}")

    norms = emb.norm(dim=-1)

    special_tokens = {0: '<s> (BOS)', 1: '<pad>', 2: '</s> (EOS)', 3: '<unk>', 4: '<mask>'}
    print("\nSpecial token L2 norms:")
    for idx, name in special_tokens.items():
        print(f"  [{idx}] {name}: {norms[idx].item():.4f}")

    print(f"\nAll-token statistics:")
    print(f"  Mean norm:   {norms.mean().item():.4f}")
    print(f"  Median norm: {norms.median().item():.4f}")
    print(f"  Min norm:    {norms.min().item():.4f}  (token {norms.argmin().item()})")
    print(f"  Max norm:    {norms.max().item():.4f}  (token {norms.argmax().item()})")
    print(f"  Tokens with norm < 0.1: {(norms < 0.1).sum().item()}")
    print(f"  Tokens with norm < 0.5: {(norms < 0.5).sum().item()}")

    mean_norm = norms.mean().item()
    pad_norm = norms[1].item()
    eos_norm = norms[2].item()
    ratio_pad = pad_norm / mean_norm if mean_norm > 0 else 0
    ratio_eos = eos_norm / mean_norm if mean_norm > 0 else 0

    print(f"\nDiagnosis:")
    if ratio_pad < 0.3 or ratio_eos < 0.3:
        print(f"  ATTRACTOR CONFIRMED: PAD norm is {ratio_pad:.1%} of mean, EOS norm is {ratio_eos:.1%} of mean.")
        print(f"  When model predicts near-zero, rounding hits PAD/EOS → empty string output.")
        print(f"  The rounding loss fix (model_out_discrete_nll) is the correct solution.")
    else:
        print(f"  PAD norm is {ratio_pad:.1%} of mean, EOS norm is {ratio_eos:.1%} of mean.")
        print(f"  Special token norms are not unusually small — attractor may be weaker than expected.")

    # Also check lm_head bias if present
    bias_key = 'lm_head.bias'
    if bias_key in m:
        bias = m[bias_key]
        top5_tokens = torch.topk(bias, k=5).indices.tolist()
        print(f"\nlm_head.bias top-5 tokens: {top5_tokens}")
        print(f"  (if 0/1/2 appear here, model predicts special tokens when input≈0)")

if __name__ == '__main__':
    main()
