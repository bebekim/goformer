# Common Pitfalls

Known, previously-hit traps. Each entry: what happened, and where the
fix lives.

- **macOS's default `/bin/bash` is 3.2 (pre-4.4).** Expanding an
  *empty* array under `set -u` throws "unbound variable" — a real bug
  hit by `run_generations.sh`'s own smoke test. Fix: use plain strings
  with unquoted word-splitting instead of arrays for optional
  command-line flags in shell scripts (see `run_generations.sh`'s
  header comment for the exact pattern).

- **`train.py`: tiny datasets can crash the val split.** A small
  enough dataset makes `int(num_examples * val_fraction)` round down
  to 0, and calling `evaluate()` on an empty val set divides by zero.
  Fixed: bump to 1 example when `num_examples >= 2`, fall back to no
  split at all below that. See `tests/test_train_early_stopping.py`'s
  `TestSmallDatasetValSplitRounding` for the regression coverage.

- **`nn.MultiheadAttention`'s fused "fastpath" kernel returns NaN**
  in `.eval()` mode with a float `attn_mask` (exactly what
  `RelativePositionBias`/`GeometricAttentionBias` inject) — but only
  *after* real training steps; never with a fresh model, never in
  `.train()` mode, which is why it's easy to miss with forward-pass-
  only tests. Fixed via `torch.backends.mha.set_fastpath_enabled(False)`
  in `engine/token_transformer.py`.

- **Comparing training runs at a fixed epoch count understates
  configs that overfit faster.** `go-style-engine/docs/
  board-specification.md` §8b found this directly: a config's val
  loss can bottom out well before a fixed comparison epoch and then
  climb back up, making an earlier read look artificially bad. Always
  compare at each config's own best epoch — `train.py
  --early-stopping-patience` now does this automatically — not a
  fixed epoch count.

- **`sail-subs` delegation failed twice on this repo**, plausibly due
  to `papers/`'s multi-MB git-tracked PDFs bloating a worker's context
  (271K input tokens on one attempt, followed by incoherent output).
  Not confirmed as the cause. See `TODO.md` and
  `Specs/002-bootstrap-knowledge-docs.md`'s Decision Log for the full
  detail before trying `sail-subs` on this repo again.

- **`docs/unattended-generation-loop.md`, if present locally, is
  gitignored on purpose.** It names a personal Sail sailbox ID and
  must never be re-added to tracking — see `.gitignore` in
  `go-style-engine/`. Not every checkout will have this file; that's
  expected, not a bug.
