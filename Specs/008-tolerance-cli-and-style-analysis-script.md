# CLI tolerance passthrough + analyze_style.py

Priority: medium
State: ready

## Problem

`docs/style-without-strength-loss.md` §5 lists concrete tooling gaps,
verified still-missing this session (checked directly against the
tree, not assumed): `selfplay.py` already exposes `--rounds-per-move`/
`--c`/`--complexity-weight`/`--safety-lambda`/`--temperature`/
`--dirichlet-epsilon` as overrides, but **not**
`--equal-value-tolerance`, even though `StyleKnobs.equal_value_tolerance`
exists as a field (`engine/mcts.py:54`) and is exactly the knob §4d's
tolerance sweep needs to vary. There is also no `analyze_style.py` —
tournament results from `007` (or any prior/future tournament) can only
be inspected by hand-reading `games.jsonl`.

## Desired Behavior

1. Add `--equal-value-tolerance` to `selfplay.py`'s CLI, following the
   exact existing pattern for the other `StyleKnobs` overrides
   (`build_knobs`'s override dict, `_write_manifest`'s recorded args).
2. Add `analyze_style.py`: reads one or more `runs/*/games.jsonl` files,
   computes win-rate (with the binomial CI already described in
   `style-without-strength-loss.md` §4c) and per-preset histograms/
   summary stats for `complexity`/`variance`/`viable_count`/
   `candidate_count`/`think_time_s` — the three numbers §4c/the doc's
   §3 already names as proving "style-without-tanking": strength inside
   the target band, and separated style telemetry.

## Non-Goals

- Do not build `MetaPreset`/`psychology_fn` (§5's other listed item) —
  a separate, larger design decision (contextual knob modulation based
  on game state), not bundled here.
- Do not build the prior-shaping variant (§5's fourth item) — explicitly
  gated behind "only if post-selection style is too subtle," which
  hasn't been established yet.
- Do not run `007`'s tolerance sweep (§4d) as part of this spec — this
  spec only builds the tools §4d needs; running it is `007`'s job once
  this lands.

## Likely Files

- `go-style-engine/selfplay.py` (`build_knobs`, argparse, manifest)
- `go-style-engine/analyze_style.py` (new)
- `go-style-engine/tests/` (new test file for `analyze_style.py`,
  matching the existing test-coverage bar in this repo — every prior
  CLI addition in this session's history got a real regression test,
  not just manual verification)

## Environment

None beyond the existing venv.

## Dependencies

None. Independent of `007` in build order (007's 4a-4c don't need
this), though `007`'s §4d specifically needs it.

## Edge Cases

- `analyze_style.py` should handle a `games.jsonl` with undecided games
  (`result.winner: null`, per the existing `move_cap_reached` path in
  `selfplay.py`'s `play_one_game`) without crashing — exclude them from
  win-rate the same way `selfplay.py`'s own summary printing already
  does (`decided` count), don't silently miscount them as losses.
- `--equal-value-tolerance` should be validated the same way other
  `StyleKnobs` float overrides are — no new validation needed if the
  existing override mechanism already handles arbitrary floats, but
  confirm (don't assume) it does before calling this done.

## Test Expectations

- New tests for `analyze_style.py`: construct a small synthetic
  `games.jsonl` (a few decided games, a few undecided, varying
  telemetry) and confirm win-rate/CI and per-metric summaries compute
  correctly.
- New test confirming `--equal-value-tolerance` reaches
  `StyleKnobs.equal_value_tolerance` end-to-end (mirroring how other
  overrides are already tested, if such tests exist — check
  `tests/test_resume.py` and similar for the existing pattern before
  writing a new one from scratch).
- `cd go-style-engine && .venv/bin/python -m pytest -q` — full suite
  passes.

## Acceptance Criteria

- [ ] `--equal-value-tolerance` works end-to-end through `selfplay.py`.
- [ ] `analyze_style.py` computes win-rate + CI + style-separation
      summary from a real `games.jsonl`.
- [ ] Both covered by new tests, not just manual verification.
- [ ] `CHANGELOG.md` updated.

## Known Risks

Low. Both pieces are small, well-precedented additions (the CLI flag
follows an existing pattern exactly; the analysis script reads
already-logged fields, computes nothing not already specified in
`style-without-strength-loss.md` §4c).

## Decision Log

| # | Proposed | Status | Why |
|---|---|---|---|
| 1 | Bundle `MetaPreset`/`psychology_fn` into this spec too, since §5 lists it alongside the CLI/analysis items. | **Rejected** | It's a real design decision (how should contextual state — leading/trailing/fatigue — modulate a knob cluster mid-game), not a mechanical tooling gap like the other two. Keeping it out keeps this spec's acceptance criteria unambiguous and quick to execute; it can become its own spec once `007`'s tournament data gives a concrete reason to want it. |
