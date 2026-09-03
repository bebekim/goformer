# Does attention actually learn Go group/liberty topology?

Priority: low
State: needs-clarification

## Problem

`docs/board-specification.md` §1 deliberately left liberty/group/eye
features out of the token encoding, framing the open research question
as "does attention recover group topology from raw adjacency + history
on its own." §9 restates this as still open: everything measured
through §8 is value/policy *loss*, never whether the model's internal
attention actually organizes around real Go structure (groups, shared
liberties, capturing races) or just fits the training distribution
through some other route entirely.

This is the one item from `board-specification.md`'s original research
agenda that has never been touched, and it's the most valuable one for
the project's actual stated purpose (per the "assess where we are"
discussion elsewhere in this repo's history) — but it's also
qualitatively different from every other spec in this queue:
`003`-`005` are "run this" or "build this," this one requires deciding
*how you would even know* before there's anything to execute.

## Desired Behavior

Not yet specifiable. Candidate approaches, none committed to:

- Probe classifiers: train a small linear/shallow probe to predict
  group membership or liberty count from a trained model's
  intermediate token representations; if the probe generalizes well,
  that's evidence the representation encodes it, even if not proof the
  *policy* uses it causally.
- Attention-pattern correlation: for `RelativePositionBias`/GAB's
  learned bias, check whether high attention weight between two tokens
  correlates with those two intersections being in the same connected
  group (using `engine/goboard.py`'s real group/liberty computation as
  ground truth) more than chance/distance alone would predict.
- Ablation: zero out or scramble a specific attention head/bias
  component and measure whether performance degrades specifically on
  capture-heavy positions vs. uniformly — a causal test, not just
  correlational.
- Follow Chessformer's own interpretability methodology directly (cited
  in `docs/board-specification.md`'s references,
  `papers/chessformer-2605.19091.pdf`) rather than inventing a new one.

## Non-Goals (provisional — will firm up once scoped)

- Not a retraining task — should reuse existing checkpoints where
  possible, not require new training runs, unless the chosen method
  specifically needs representations from a much better-trained model
  than currently exists (in which case that dependency must be named
  explicitly, likely on `007`'s CNN track or a properly-trained
  token-transformer, neither of which exist yet either).

## Likely Files

Unknown until scoped — likely a new `go-style-engine/analysis/` or
similar directory, not existing `engine/` code.

## Environment

Unknown until scoped.

## Dependencies

Soft dependency on having *any* meaningfully-trained checkpoint to
probe — right now every checkpoint in this repo comes from tiny
(1,200-4,400 position), single-seed, architecture-ablation runs, not
anything trained to actually play well. Whether that's good enough to
probe, or whether this needs `007`'s (or a token-transformer
equivalent's) real training investment first, is itself part of what
needs clarifying before this spec can move to `ready`.

## Edge Cases

Unknown until scoped.

## Test Expectations

Unknown until scoped.

## Acceptance Criteria

Unknown until scoped — this spec cannot be marked `ready` until a
specific method (from the candidates above or another) is chosen and
the rest of the template filled in for real.

## Known Risks

The main risk is scope explosion: "does attention learn group
topology" can expand into an open-ended research program. Whoever
picks this up should resolve it into one small, falsifiable first
question (e.g. "does a linear probe on layer-N token representations
predict liberty count above chance, on the best checkpoint currently
available") rather than a general investigation.

## Decision Log

| # | Proposed | Status | Why |
|---|---|---|---|
| 1 | Write this as a fully-scoped `State: ready` spec now, guessing at a specific method. | **Rejected** | `SPEC_PREFLIGHT.md`'s own discipline: "if implementation would require an unstated requirement... do not infer it silently." Which method, what checkpoint, what counts as evidence — these are real methodology decisions the repo owner should make, not ones to guess through to hit a numbering deadline. |
| 2 | Leave this out of `Specs/` entirely, as a `TODO.md` note instead, since it's not executable yet. | **Rejected** | It's a genuine, previously-identified research question (§1/§9 of `board-specification.md`), not a passing observation — it deserves the same durable tracking as every other deferred item, explicitly marked `needs-clarification` rather than silently dropped or buried in `TODO.md`'s lower-ceremony format. |
