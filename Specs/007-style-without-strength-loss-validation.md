# Run the style-without-strength-loss validation plan (§4)

Priority: high
State: ready

## Problem

`docs/style-without-strength-loss.md` §4 lays out a complete, specific
experiment answering this project's actual core question: can a
character trait (a `StyleKnobs` cluster) produce a recognizable,
distinguishable Go style while staying within a few points of optimal
strength — the literal game mechanic ("idols develop distinct
personality without win-rate collapsing"). **This plan has never been
run.** Every checkpoint in `go-style-engine/checkpoints/` as of this
spec is from this session's *token-transformer architecture ablation*
work (§7/§8 of `board-specification.md`) — small (9x9), random-net-
generated, single-seed, never intended to play well. `runs/demo.json`
(§4's own stated starting point) is still the random-net degenerate
baseline described in the doc: frequent pass, 12-move games, no signal.
This is the highest-leverage unexecuted work in the whole repo — it
directly tests the thing the project is actually for, using the
simpler, already-purpose-built `GoZeroNet`/`StyleKnobs` mechanism, not
the token-transformer line this session spent most of its time on.

## Desired Behavior

Execute §4a-§4c in full (§4d — the tolerance sweep — depends on `008`
and can follow once that lands; don't block on it). Concretely, from
`go-style-engine/` (paths adjusted from the doc's original
`research/go-style-engine/`, which predates this repo's move):

```bash
# 4a. Base checkpoint (one-time)
.venv/bin/python selfplay.py --board-size 13 --games 200 --black-preset baseline \
  --white-preset baseline --rounds-per-move 200 --out runs/gen0/ --save-experience runs/gen0_exp.npz
.venv/bin/python train.py --experience runs/gen0/ --board-size 13 \
  --out-checkpoint checkpoints/base.pt --epochs 8 --seed 0

# 4b. Per-style fine-tunes (repeat for fighter/builder/adapter/anchor)
.venv/bin/python selfplay.py --board-size 13 --games 120 --black-preset fighter \
  --white-preset fighter --checkpoint checkpoints/base.pt --rounds-per-move 200 --out runs/fighter_gen/
.venv/bin/python train.py --experience runs/fighter_gen/ --in-checkpoint checkpoints/base.pt \
  --out-checkpoint checkpoints/fighter.pt --epochs 3 --seed 1

# 4c. Tournament, knobs-only first (no checkpoint delta)
for preset in fighter builder adapter anchor; do
  .venv/bin/python selfplay.py --board-size 13 --games 60 --black-preset $preset \
    --white-preset baseline --checkpoint checkpoints/base.pt --out runs/tourney_${preset}_vs_base/
done
# then repeat with --checkpoint checkpoints/${preset}.pt for the fighter/etc side,
# to measure the additive effect of the Voice checkpoint on top of knobs alone
```

Compute, per §4c, from each `games.jsonl`: black win-rate vs. baseline
(target 40-60%, binomial 95% CI ±13% at 60 games — "without tanking"
passes if inside that band) and per-preset distributions of
`complexity`/`variance`/`viable_count`/`candidate_count`/`think_time_s`
(already-logged telemetry fields) as the style-separation evidence.

## Non-Goals

- Do not run §4d (tolerance sweep) as part of this spec — it needs
  `--equal-value-tolerance` CLI passthrough, which doesn't exist yet
  (`008`'s job). This spec's acceptance criteria cover 4a-4c only.
- Do not touch the token-transformer (`TokenTransformerNet`) line of
  work — this spec is entirely the `GoZeroNet`/`StyleKnobs` CNN path,
  deliberately separate.
- Do not treat `PRESETS`' current knob values as final — they're
  explicitly documented as placeholders in `selfplay.py`; if the
  tournament shows a preset isn't separable, that's a real result
  (adjust the preset, or record why it doesn't separate), not a bug in
  this spec.
- Do not build `analyze_style.py` as part of this spec — that's `008`;
  compute the win-rate/telemetry numbers by hand or with an ad hoc
  script for this first run if `008` hasn't landed yet.

## Likely Files

- `go-style-engine/runs/gen0/`, `fighter_gen/`, `builder_gen/`,
  `adapter_gen/`, `anchor_gen/`, `tourney_*_vs_base/` (new, gitignored
  — these are data, not tracked)
- `go-style-engine/checkpoints/base.pt`, `fighter.pt`, `builder.pt`,
  `adapter.pt`, `anchor.pt` (new, gitignored)
- `go-style-engine/docs/style-without-strength-loss.md` (record the
  real result — a new section, following that doc's existing style)

## Environment

CPU-only. **This is a large compute job** — 13x13 with 200 rounds/move
is a different cost regime than every 9x9 experiment in this session
(§7/§8 of `board-specification.md` used 9x9/50 rounds; 13x13/200 rounds
is meaningfully more expensive per move on both board size and search
depth). No real timing data exists yet at this board size/round count
combination — budget generously (likely multiple hours total across
4a-4c) and run it unattended (the Sail devbox setup from this session's
history, or `sail-subs`/`sail-swarm`, are exactly what this is for) —
do not attempt this interactively in one sitting.

## Dependencies

`008` (tolerance-sweep tooling) blocks §4d only, not this spec's own
acceptance criteria (4a-4c).

## Edge Cases

- If a preset's win-rate falls outside 40-60% against baseline, that's
  a real finding to record, not something to silently retune away —
  note it plainly, the way this session's own §7/§8 findings were
  reported even when they contradicted the working hypothesis.
- If style separation doesn't show up in the telemetry distributions,
  same treatment — a genuine negative result for the core question,
  worth exactly as much documentation as a positive one.
- `runs/demo.json` (mentioned as the "before" baseline in the doc) may
  no longer exist after the repo move — regenerate it if needed for a
  before/after comparison, don't assume it's present.

## Test Expectations

- No new automated tests. The "test" is the tournament data itself —
  win-rate within target band, style-separation metrics computed and
  recorded.

## Acceptance Criteria

- [ ] `checkpoints/base.pt` exists, trained per 4a, with the expected
      qualitative change from random-net behavior (leaf values spread
      to [-1,1], viable_count drops — see
      `docs/mcts-trace-walkthrough.md`'s "what changes with a trained
      brain" for what to check against).
- [ ] Four per-style checkpoints (`fighter.pt`/`builder.pt`/
      `adapter.pt`/`anchor.pt`) exist per 4b.
- [ ] Tournament run for all four presets vs. baseline, knobs-only and
      with-checkpoint-delta, per 4c.
- [ ] Win-rate and style-separation numbers computed and written into
      `docs/style-without-strength-loss.md` as a new, evidence-based
      section — real numbers, not projections.
- [ ] `CHANGELOG.md` updated.

## Known Risks

The main risk is scale: this is the largest single compute commitment
in this repo's history, with no prior timing data at 13x13/200-rounds
to estimate against. Recommend a small timing probe (a handful of
games) before committing to the full 200+120×4+60×8 game budget, the
same discipline `board-specification.md` §7 used before its own first
large self-play run. A secondary risk is that `PRESETS`' placeholder
knob values simply don't produce separable telemetry at all — plan for
that as a real, useful negative result, not a spec failure.

## Decision Log

| # | Proposed | Status | Why |
|---|---|---|---|
| 1 | Run this on the token-transformer (`TokenTransformerNet`) instead of `GoZeroNet`, since that's what this session's work mostly focused on. | **Rejected** | `StyleKnobs` is net-agnostic (already proven to work with either net via `ZeroAgent`'s shared contract), but `GoZeroNet` is the smaller, simpler, already-purpose-built net with no known overfitting/capacity issues — the right choice for validating the *mechanism* (style knobs) rather than conflating it with the token-transformer's own open questions (§8/§9). |
| 2 | Include §4d (tolerance sweep) in this spec's scope. | **Rejected** | Needs `--equal-value-tolerance` CLI passthrough, which doesn't exist (`008`). Splitting it out means 4a-4c can start immediately rather than waiting on a small, independent CLI change. |
| 3 | Delegate this via `sail-subs`/`sail-swarm` given its size, vs. run it directly/locally. | **Pending** | Genuinely large (likely multi-hour) and a good candidate for unattended execution, but whether it fits `sail_delegate`'s turn/timeout model for a long-running training job (vs. a bounded coding task) hasn't been checked. Whoever picks this up should verify that fit before assuming delegation is the right execution path, not just the right size. |
