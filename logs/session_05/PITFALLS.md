# Session 05 — Training Pitfalls & Lessons Learned

## Overview
EN→RU production training on A100 SXM4 40GB (instance 39898796, $1.07/hr).
Target: 500k steps × batch=128 = 64M samples (paper-equivalent).
Actual: Stopped at step ~273k due to model collapse after disk crash.

---

## Pitfall 1: Disk Full → EMA Checkpoint Corruption (Step 150k)

**What happened:**
- Training ran without disk monitoring. Accumulated 16 model checkpoints × 638MB + 15 EMA checkpoints × 785MB = ~24GB, filling the 32GB disk to 100%.
- PyTorch writes model checkpoint first, then EMA. Disk filled mid-EMA write.
- Result: `ema_0.9999_150000.pt` = 22MB (corrupted). `model150000.pt` = 638MB (intact).
- Training crashed with: `RuntimeError: [enforce fail at inline_container.cc:672] unexpected pos`

**What we lost:**
- All intermediate model checkpoints (0–140k) — were deleted during disk cleanup.
- `model140000.pt` — gone forever. Cannot resume from 140k.

**Fix applied:**
1. Deleted all intermediate model checkpoints (keep only latest).
2. Deleted intermediate EMA checkpoints (keep every 50k + latest).
3. Changed `--save_interval 10000 → 50000`.
4. Added `scripts/monitor_training.sh` — auto-cleans disk at 88% usage, auto-restarts if training crashes.

**Prevention for next run:**
- Always run `monitor_training.sh` from the start.
- Use `--save_interval 50000` from the start.
- Keep only 1 model checkpoint at a time (auto-delete in monitor).
- Keep EMA every 50k steps only.

---

## Pitfall 2: Model Collapse After Restart from model150000.pt (Steps 150k–250k)

**What happened:**
- Restarted training from `model150000.pt` after disk crash.
- Grad norms immediately dropped 100×: `0.0953 → 0.000925`.
- Loss plateaued at `0.013` for the entire 100k steps (150k–250k) — no improvement.
- Inference at step 250k: **empty string outputs** `["", "reference"]`.
- Inference at step 200k: Russian + noise (still some signal).
- Inference at step 140k: Russian + noise (good quality for this stage).

**Root cause (suspected):**
- `model150000.pt` was saved while the disk was critically full.
- The model checkpoint may have been written to a nearly-full disk, potentially with subtle corruption OR the model was already in a very flat loss region that doesn't correspond to good translation.
- After restart, the model converged to a degenerate minimum: low MSE loss (diffusion objective) but predicting near-padding embeddings → empty outputs.
- The EMA reset (reinitialized from model150000.pt instead of prior EMA state) may have contributed.

**Key insight:**
- Diffusion MSE loss ≠ translation quality. Loss can look healthy (0.013) while outputs are garbage.
- Always check inference output at checkpoints, not just loss.
- The LR warmup was NOT the cause — `_anneal_lr()` in `trainer.py` uses `self.step + self.resume_step` so LR correctly continues from step 150k.

**Best checkpoint before collapse:** `ema_0.9999_140000.pt` (Jun 8 04:48) — showed Russian words in output.

---

## Pitfall 3: Watcher Process Dying Silently

**What happened:**
- `watch_checkpoints.sh` was launched in a tmux window, but the window died when tmux session was killed/restarted.
- The watcher missed the 250k checkpoint entirely.
- No alert was raised; had to notice manually.

**Fix applied:**
- Monitor script now auto-restarts training if it crashes.
- But the watcher itself (inside `train` session) still dies when session is killed.

**Prevention:**
- Launch watcher in a separate persistent session (not inside `train` session).
- Or have the monitor also check/restart the watcher.

---

## Pitfall 4: Deleted model140000.pt During Disk Cleanup

**What happened:**
- During disk cleanup after Pitfall 1, deleted ALL intermediate model checkpoints to free space.
- Only kept: `model150000.pt`, `model200000.pt`, `model250000.pt`.
- `model140000.pt` (the best pre-crash checkpoint) was deleted and is unrecoverable.

**Impact:**
- Cannot resume training from the best pre-crash state (140k).
- Next best: `model110000.pt` (saved locally before crash).

**Prevention:**
- Always save at least the last 2 model checkpoints.
- Download model checkpoints to local machine before deleting from remote.

---

## Checkpoint Status After Session 05

| Checkpoint | Location | Quality | Use |
|------------|----------|---------|-----|
| `ema_0.9999_100000.pt` | Local + Remote | Russian + noise (good) | Inference |
| `ema_0.9999_140000.pt` | Local + Remote | Russian + noise (best) | Inference |
| `model110000.pt` | **Local only** | Good (pre-crash) | **Resume training** |
| `ema_0.9999_200000.pt` | Local + Remote | Russian + noise (OK) | Inference |
| `ema_0.9999_250000.pt` | Local + Remote | Mostly empty (collapsed) | Discard |
| `model250000.pt` | Local + Remote | Collapsed | Discard |

---

## Plan for Session 06

**Goal:** Resume from `model110000.pt` and train cleanly to 500k.

**Key changes vs Session 05:**
1. Start `monitor_training.sh` immediately (disk + crash guard).
2. Use `--save_interval 50000` from the start.
3. Delete old model checkpoint immediately after each new save (keep only latest).
4. Run 10-sentence inference every 50k (watcher) + 200-sentence BLEU at every 50k.
5. If disk > 80%, alert. If disk > 88%, auto-clean.
6. Download checkpoints to local after each 50k save.

**Expected timeline from step 110k:**
- Steps remaining: 390k
- Rate: ~3 steps/sec on A100
- Time: ~36 hours
- Cost: ~$38 at $1.07/hr
