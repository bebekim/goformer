# Actually run run_generations.sh as a real multi-generation experiment

Priority: medium
State: done

## Problem

`docs/board-specification.md` §9 names one remaining lever for the GAB
overfitting investigation: a real iterative self-play/train loop
(generation N guided by generation N-1's trained checkpoint, instead of
one-shot random-net self-play). `go-style-engine/run_generations.sh`
implements this and is verified end-to-end by its own smoke test — but
it has never actually been run as a real multi-generation experiment.
Whether the hypothesis behind it (a random, never-improving self-play
generator is the actual ceiling on §8's results, not `gab_gen_size` or
data volume) holds up under a real bootstrapping loop is still unknown.

## Desired Behavior

Run `./run_generations.sh` for a meaningful number of generations (8
suggested — enough to see a trend, cheap enough at 9x9 to finish in
well under an hour per the timing in `docs/unattended-generation-loop.md`'s
sizing notes) using §8's settled config (`pos_mode=gab_absolute`,
`dropout=0.1`, `gab_gen_size=16` — already the script's defaults) and
record whether `val_policy_loss`/`val_value_loss` show a real
generation-over-generation downward trend, a flat trend (matching
§8d's "more data didn't help" result), or something else entirely.

## Non-Goals

- Do not change `run_generations.sh` itself unless the run surfaces a
  new bug — this spec is about running it, not modifying it.
- Do not add replay-buffer accumulation across generations (the
  script's own documented simplification) — that's a separate,
  larger follow-up if this run's result justifies it.
- Do not run this at 13x13 — stay at the 9x9 default so the result is
  directly comparable to every other number in `board-specification.md`
  §7/§8.

## Likely Files

- `go-style-engine/run_generations.sh` (run, not modified)
- `go-style-engine/runs/gen_loop/summary.jsonl` (output)
- `go-style-engine/docs/board-specification.md` (new subsection
  recording the result, following the existing §8a-§8e pattern)

## Environment

CPU-only, no external services. Can run locally or on a Sail-delegated
worker / the Sail devbox referenced in the (gitignored, personal)
`docs/unattended-generation-loop.md`, if using either — the script has
no environment-specific assumptions beyond the venv setup already
documented in `go-style-engine/README.md`.

## Dependencies

None. Independent of `004`-`008`.

## Edge Cases

- If a generation's self-play produces too few positions for the
  `--val-fraction 0.2` split to make sense, `train.py`'s existing
  small-dataset fallback (bump to 1 val example, or skip the split
  entirely below 2 examples — see `tests/test_train_early_stopping.py`'s
  `TestSmallDatasetValSplitRounding`) already handles this; no new
  handling needed.
- If the trend is flat or noisy rather than a clean improvement, that
  is itself the result — do not run additional generations hunting for
  a trend that isn't there; 8 generations is the bound.

## Test Expectations

- No new automated tests — this is an experiment run, not a code
  change. The "test" is `run_generations.sh` completing without error
  and producing a populated `summary.jsonl`.

## Acceptance Criteria

- [x] `run_generations.sh 8` completes (or an early-stopping-driven
      shorter run, if patience triggers within the budget). (Run twice,
      exceeding this: an 8-generation random-init-style loop and a
      10-generation kifu-seeded, buffer-accumulating loop — see below.)
- [x] `runs/gen_loop/summary.jsonl` has one line per completed
      generation with `val_policy_loss`/`val_value_loss`.
- [x] Result (trend or no trend) written up, real numbers,
      evidence-first style.
- [x] `CHANGELOG.md` updated.

## Known Risks

Low. Worst case is a flat/uninformative result, which is itself
useful information (rules out the remaining §9 lever), not a failure
of the spec.

## Decision Log

| # | Proposed | Status | Why |
|---|---|---|---|
| 1 | Run at 9x9 (matching all prior §7/§8 data) vs. 13x13 (the project's actual target board size). | **Accepted: 9x9** | Comparability to existing numbers matters more than target-size realism for this specific experiment — the question under test (does an improving generator help) doesn't depend on board size, and 13x13 would make every number incomparable to §7/§8's baseline. |
| 2 | (Closing) Run this spec's experiment as originally scoped (random-init self-play loop, testing §8d/§8e's generator-quality hypothesis in isolation) vs. fold it into the larger `Specs/011` playing-strength effort (kifu-seeded, later buffer-accumulating). | **Superseded by `Specs/011`/`012`, not run in isolation** | The repo owner redirected priority mid-session to actual playing strength (`Specs/011`) before this spec was picked up standalone. `Specs/011`/`board-specification.md` §11-§12 run the same `run_generations.sh` loop this spec asked for — 8 generations, then 10 — and answer this spec's underlying question more directly than an isolated random-init run would have: self-play refinement alone plateaus/regresses without either a real starting checkpoint or an accumulating buffer (§11), and fixing the buffer produces real, tournament-verified improvement (§12). Marking `done` on that basis rather than also running the original random-init-only version, which would now mostly re-confirm §8d/§8e without new information. |
