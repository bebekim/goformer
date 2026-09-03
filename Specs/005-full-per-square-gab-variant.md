# Implement the full per-square GAB variant (gab_per_square_dim>0)

Priority: low
State: ready

## Problem

`GeometricAttentionBias` only implements the cheap mean-pooled variant
(maia3's `gab_per_square_dim=0`) — its own docstring states the full
per-square projection variant (`gab_per_square_dim>0`, used by maia3's
23M/79M models for more expressiveness at higher cost) is "deliberately
not implemented here," with the explicit condition for building it:
"only if the mean-pooled variant is validated to help." §8's validation
found exactly that (GAB's content-dependence measurably helps the value
objective, §8 Finding 1) — the stated precondition for attempting this
is now met, but it remains untried.

## Desired Behavior

Add a `gab_per_square_dim` parameter to `GeometricAttentionBias`,
matching maia3's real implementation (`MHA._sq_bias` in
`~/repositories/individual/deep-learning/maia3/maia3/models.py`): when
`> 0`, project each token individually (`sm1`) before flattening and
concatenating (`num_tokens * gab_per_square_dim` wide) into the
existing `summarize`/`generate` pipeline, instead of mean-pooling first.
When `0` (the current, only implemented behavior), unchanged.

## Non-Goals

- Do not remove or change the mean-pooled path — it stays the default,
  this is a new opt-in mode alongside it, matching how `pos_mode`
  itself is already a set of alternatives, not a replacement chain.
- Do not run a full training comparison as part of this spec's
  acceptance criteria beyond a basic smoke test — a real empirical
  validation (does the extra expressiveness help or just add more
  capacity-mismatch risk, per §8's own capacity/data warnings) is a
  separate follow-up spec once this exists to test.
- Do not change `TokenTransformerNet`'s public `pos_mode` API — this is
  purely a `GeometricAttentionBias` constructor parameter, reachable via
  the existing `gab_gen_size`/`gab_intermediate_dim`-style CLI flags in
  `selfplay.py`/`train.py`.

## Likely Files

- `go-style-engine/engine/token_transformer.py`
  (`GeometricAttentionBias.__init__`/`.forward`)
- `go-style-engine/selfplay.py`, `go-style-engine/train.py` (new
  `--gab-per-square-dim` flag, matching the existing
  `--gab-gen-size`/`--gab-intermediate-dim` pattern)
- `go-style-engine/tests/test_geometric_attention_bias.py` (new
  parametrized coverage: `gab_per_square_dim=0` behaves exactly as
  before — a regression check — and `>0` shape/gradient/ZeroAgent
  coverage matching the existing per-mode test classes)

## Environment

None beyond the existing venv.

## Dependencies

None, though the natural follow-up (an empirical comparison) would
want `004`'s best-epoch-not-fixed-epoch methodology applied from the
start, not bolted on after.

## Edge Cases

- `gab_per_square_dim` changes the parameter count substantially (this
  is the whole point — maia3's own configs jump `gab_intermediate_dim`
  and cost together when this is nonzero). Document the new parameter
  count formula in the class docstring the same way the existing
  mean-pooled cost (`num_tokens² × gen_size`) is documented, so a
  future reader isn't surprised by a much larger model.
- Confirm the existing `torch.backends.mha.set_fastpath_enabled(False)`
  fix (needed for the mean-pooled variant's NaN bug) still applies —
  it's a global backend setting unrelated to `gab_per_square_dim`, so
  it should, but verify with the new regression test rather than
  assume.

## Test Expectations

- `cd go-style-engine && .venv/bin/python -m pytest -q` — full suite
  must still pass, plus new tests for the `gab_per_square_dim>0` path
  specifically (shape, gradient flow, the eval-after-training NaN
  regression check, ZeroAgent integration) mirroring the existing
  per-mode test class structure in `test_geometric_attention_bias.py`.

## Acceptance Criteria

- [ ] `GeometricAttentionBias` supports `gab_per_square_dim>0`,
      matching maia3's real per-square projection logic.
- [ ] `gab_per_square_dim=0` behavior is bit-for-bit unchanged (existing
      tests continue to pass without modification).
- [ ] New tests cover the `>0` path with the same rigor as the existing
      mean-pooled coverage.
- [ ] `selfplay.py`/`train.py` expose the new parameter via CLI flag.
- [ ] `docs/board-specification.md` gets a short note (not a full
      empirical validation — that's a follow-up) recording that this
      variant now exists and what its cost formula is.
- [ ] `CHANGELOG.md` updated.

## Known Risks

Medium-low. The main risk is scope creep into "also validate it
empirically" — resist that; this spec is implementation-only, matching
the Non-Goals. The implementation itself is a faithful port of an
already-proven reference (maia3's real shipped code), so correctness
risk is lower than most net-new work in this repo.

## Decision Log

| # | Proposed | Status | Why |
|---|---|---|---|
| 1 | Build this now, since §8's precondition ("only if the mean-pooled variant is validated to help") is met. | **Accepted, but scoped to implementation-only** | The precondition is met, but §8/§9 also already flag capacity-vs-data mismatch as the dominant unresolved risk in this whole line of work — adding more capacity (this spec) before that risk is better understood is worth doing for the tooling, not worth doing paired with an immediate large training investment. Kept as implementation + smoke test, with real validation deferred to its own spec once this exists. |
