# Re-read relative/both/absolute's §7 numbers at their own best epoch

Priority: low
State: ready

## Problem

`docs/board-specification.md` §8b found that comparing configs at a
fixed epoch (40) understates ones that overfit faster — the actual
finding was GAB's true minimum sat at epoch 13, not epoch 40. §7's
original `relative`/`both`/`absolute` comparison (and §7's follow-up
`both`-mode diagnostic) was run and reported *before* this "read at
fixed epoch 40" problem was even identified, so those three configs
were never re-read at their own best epoch either — §9 flags this
explicitly as still open.

## Desired Behavior

For each of `relative`, `both`, and `absolute`, find the epoch with
the lowest combined val loss (`val_policy_loss + val_value_loss`) from
already-existing training logs where available, and report those
numbers alongside §7/§8's existing fixed-epoch numbers in a short new
subsection of `board-specification.md`.

## Non-Goals

- Do not re-run `relative` or `both` — 40-epoch logs already exist
  (`checkpoints/stage2_val_relative_long.pt.meta.json`,
  `checkpoints/stage2_val_both.pt.meta.json`); this is a read-only
  analysis of data that's already on disk.
- Do not change §7's original fixed-epoch numbers in place — add the
  best-epoch numbers alongside them, so the doc shows both and the
  reader can see what changed and why, rather than silently rewriting
  history.
- Do not extend this into a full re-litigation of §7's conclusions —
  the point is just to check whether the direction of §7's findings
  changes, not to redo the whole comparison from scratch.

## Likely Files

- `go-style-engine/checkpoints/stage2_val_relative_long.pt.meta.json` (read)
- `go-style-engine/checkpoints/stage2_val_both.pt.meta.json` (read)
- `go-style-engine/checkpoints/stage2_val_absolute.pt.meta.json` (read —
  only has 12 epochs; see Edge Cases)
- `go-style-engine/docs/board-specification.md` (new subsection)

## Environment

None. Pure local file reads (`json.load` on existing `.meta.json`
sidecars) plus, conditionally, one training run for `absolute` if the
edge case below applies.

## Dependencies

None.

## Edge Cases

- **`absolute` only has 12 epochs logged** (`checkpoints/
  stage2_val_absolute.pt.meta.json`) — no 40-epoch run was ever made,
  because at the time it looked stable/converging already. Before
  writing up a "best epoch" number for `absolute`, check whether its
  val loss is still monotonically improving at epoch 12 (meaning 12
  epochs isn't enough to know its true best epoch) or already showing
  the early signs of climbing back up (meaning 12 is enough). If the
  former, this spec needs one supplementary 40-epoch `absolute` run
  before it can be completed fairly — don't report a
  possibly-artificially-good "best epoch = 12" number for `absolute`
  while `relative`/`both` get the benefit of a full 40-epoch search.

## Test Expectations

- No automated tests — this is a read-only data analysis + doc update.

## Acceptance Criteria

- [ ] Best combined-val-loss epoch identified for `relative` and `both`
      from existing logs.
- [ ] `absolute` either confirmed sufficient at 12 epochs (with the
      reasoning shown) or extended to 40 epochs first.
- [ ] New subsection added to `board-specification.md` (following the
      §8b/§8e format: a table, then a plain-language statement of
      whether this changes §7's conclusions or just refines the
      numbers).
- [ ] `CHANGELOG.md` updated.

## Known Risks

Very low — read-only analysis, at most one small supplementary training
run. The only real risk is drawing a stronger conclusion than the data
supports (e.g. claiming a "trend reversal" from noise) — cross-check
against §8b/§8e's own stated epistemic caution (one seed, no sweep)
before writing anything definitive.

## Decision Log

| # | Proposed | Status | Why |
|---|---|---|---|
| 1 | Re-run all three configs from scratch for a clean, uniform comparison vs. reuse existing logs where available and only fill the gap for `absolute`. | **Accepted: reuse existing logs** | `relative`/`both` already have full 40-epoch data sitting on disk unused — re-running them would waste compute re-deriving numbers already available, and this spec is explicitly scoped as cheap/low-priority precisely because most of the data already exists. |
