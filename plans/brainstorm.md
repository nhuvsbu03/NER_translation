# Brainstorm

Running list of ideas, experiments, and open questions. Move ideas to the next session prep file when you decide to act on them.

---

## Inference Quality

### MBR Decoding (Minimum Bayes Risk)
- **Idea**: Set `mbr_sample=5` — generate 5 candidates per sentence, pick the one with highest average BLEU against the other 4
- **Cost**: 5× inference time
- **Expected gain**: Largest single quality boost for NMT — well established technique
- **Status**: Not tried yet

### generate_by_q
- **Idea**: Set `generate_by_q=True` — at each diffusion step, instead of sampling from learned reverse process, re-encode x_0 prediction through forward process q
- **Effect**: More stable trajectory, enforces consistency with forward diffusion
- **Tradeoff**: Less expressive than full reverse process
- **Status**: Not tried yet

### clip_denoised
- **Idea**: Set `clip_denoised=True` — clamps predicted x_0 embeddings to [-1, 1] during sampling
- **Effect**: Prevents embeddings drifting into extreme regions during inference
- **Combined with**: Better vocab (32k) and pretrained init — might help more now
- **Status**: Currently False — try in session 03

### top_p noise clamping
- **Idea**: Reduce `top_p` from 0.9 to 0.5 — limits noise magnitude during sampling
- **Effect**: More deterministic outputs, less stochasticity
- **Status**: Not tried yet

### NER-guided reranking
- **Idea**: Generate N outputs, run NER on each, score by how well named entities from source appear in output
- **Requires**: NER tagger for ZH
- **Status**: Future idea

---

## Data

### Sequence Length Truncation
- **Problem**: `sequence_len=64` truncates 23.2% of ZH target sentences (median=43, p90=85 chars)
- **Fix**: Increase `sequence_len` to 128 — covers 98.3% of sentences
- **Cost**: ~4× memory for the target sequence self-attention; may need to reduce batch_size 16→8
- **Status**: Not done — planned for session 03

### NER Tag Tokenization
- **Concern**: `[LOC]`, `[ORG]`, `[PER]`, `[`, `]`, `:` should be stable tokens in the 32k BPE vocab
- **Check**: After tokenizer training, verify these strings are not split across multiple tokens
- **Fix if broken**: Add as special tokens before BPE training
- **Status**: Not verified yet

### Data Augmentation
- **Idea**: For sentences with NER tags, create augmented versions swapping entities with random same-type entities
- **Goal**: Force model to learn entity-type structure rather than memorizing specific entities
- **Status**: Future idea

---

## Training

### Sequence Length 128 for Target
- See Data section above — same change
- If `sequence_len=128`, also update inference `--sequence_len 128`

### Longer Training
- Current plan: 100k steps (~4 epochs over 233k examples)
- If BLEU plateaus early, extend to 150k steps
- LR schedule already set for 100k — would need `lr_anneal_steps=150000` if extending

### Curriculum Learning
- **Idea**: Train first on short sentences (seq_len ≤ 32), then gradually increase
- **Goal**: Easier examples first → faster embedding geometry convergence
- **Complexity**: Requires custom data loader
- **Status**: Future idea

---

## Architecture

### Larger Embedding Dimension
- Current: `in_channel=512`, `num_channels=2048`
- Larger embedding → more room for token-specific regions in embedding space
- Cost: significantly more memory and slower training
- Status: Future idea

### NER-conditioned Diffusion
- **Idea**: Extract NER tags from source, create a separate NER embedding, inject as additional conditioning signal into the diffusion process
- **Goal**: Model explicitly knows "this position is a LOC entity" → better entity translation
- **Complexity**: Significant architecture change
- **Status**: Future idea (after baseline is working well)

---

## Open Questions

1. Does `use_ddim=True` actually do anything in this codebase? (Exploration agent said DDIM was removed — verify from training logs)
2. Does `init_pretrained=True` load the embedding layer from BART or just the transformer weights? (Vocab mismatch means embedding layer is likely random — confirm)
3. What is the BLEU ceiling for this dataset/architecture? Run Helsinki-NLP/opus-mt-en-zh as baseline for comparison
4. Are there misaligned pairs beyond the 6,349 we filtered? (The extreme_ratio filter only catches obvious cases)
