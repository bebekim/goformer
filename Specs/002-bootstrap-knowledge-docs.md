# Fill the five agent knowledge-doc stubs

Priority: high
State: done

## Problem

`AGENTS.md` routes agents to `docs/architecture.md`, `docs/testing.md`,
`docs/domain.md`, `docs/style-guide.md`, and `docs/common-pitfalls.md` —
none of these files exist yet. This repo was bootstrapped into the
night-shift baseline (`Specs/`, `AGENT_LOOP.md`, ground-loop guard) but
never had its domain knowledge filled in, so any future spec's preflight
check ("are there contradictions with AGENTS.md/AGENT_LOOP.md/docs?")
has nothing real to check against, and any agent working here — human,
Claude, or a `sail-subs` worker — has to rediscover the same context
from scratch every time.

## Desired Behavior

Create the five files below under `docs/`, each with real, verified,
repo-specific content — not generic best practice. Every claim in each
doc must be checkable against the actual current tree (file paths, test
commands, line numbers where cited) rather than asserted from memory.

### `docs/architecture.md`

- Three sibling top-level areas: `go-style-engine/` (the actual code —
  Python, PyTorch, pytest), `papers/` (a local PDF mirror with its own
  README explaining why each paper is kept), `convo-style-engine/`
  (currently empty — placeholder for a future dialogue/coaching-text
  counterpart, not yet started).
- Inside `go-style-engine/`: `engine/` (the library — rules engine
  ported from `dlgo/`, the CNN net + MCTS/style-knobs, and the
  token-Transformer variant), `selfplay.py`/`train.py` (CLI entry
  points), `tests/`, `docs/` (the real design documents — this repo has
  *two* `docs/` directories, one at root for the night-shift knowledge
  stubs, one under `go-style-engine/` for the actual technical specs;
  name both explicitly so this isn't confused later).
- Two parallel network architectures exist on purpose, not as
  leftover cruft: `engine/network.py` (`GoZeroNet`, CNN, AlphaZero-style,
  the older/simpler baseline) and `engine/token_transformer.py`
  (`TokenTransformerNet`, the Transformer/GAB line of work). Both
  implement the same external contract (`encoder.encode`/
  `decode_move_index`/`num_moves()`, `model.predict(state_tensor,
  device) -> (priors, value)`) so `engine/mcts.py`'s `ZeroAgent` works
  with either without modification — state this contract explicitly, it's
  the single most important architectural fact for extending either net.
- Point to `go-style-engine/docs/board-specification.md` as the primary
  design document for the token/Transformer line — it's long (10
  sections plus subsections) and already contains the real history;
  don't duplicate it here, summarize its existence and what it covers.
- Point to `go-style-engine/docs/from-transformer-to-trans-go-former.md`
  as the from-zero conceptual explainer, and
  `go-style-engine/docs/mcts-trace-walkthrough.md` and
  `go-style-engine/docs/style-without-strength-loss.md` as the
  CNN/MCTS-side design docs.
- `run_generations.sh` at the `go-style-engine/` root: what it does
  (iterative self-play/train loop) and that it's meant to be run
  unattended, typically off-laptop.

### `docs/testing.md`

- Canonical test command:
  `cd go-style-engine && .venv/bin/python -m pytest -q`
  — verify this against `go-style-engine/.sandcastle`... (no, that's
  gone) — verify against `.sandcastle` is wrong, verify by actually
  running it and recording the real pass/fail/xfail counts at spec
  completion time, not by copying a stale number from this spec.
- Explain the `xfail(strict=True)` convention in `tests/test_rules.py`
  (executable spec for a future Swift port — xfails are intentional
  documentation of not-yet-implemented behavior, not failures to fix).
- Note environment setup: `python3 -m venv .venv && .venv/bin/pip
  install -r requirements.txt` — and the known Python-version pin issue
  (`requirements.txt` pins `numpy==2.5.2`/`torch==2.13.0`, which need
  Python 3.12+; on a host with only 3.11, install unpinned-but-compatible
  versions instead: `.venv/bin/pip install -q numpy torch pytest`).
- List the test files and roughly what each covers (rules engine,
  resume logic, token encoder, token transformer, relative-position
  bias, geometric attention bias, train.py's early-stopping/val-split
  behavior) — a one-line-per-file index, not a restatement of each
  file's contents.

### `docs/domain.md`

- Baduk/Go vocabulary an agent needs without prior Go knowledge:
  board, intersection, stone, group, liberty, capture, ko, pass, komi,
  self-play, MCTS, policy/value head, checkpoint, epoch.
- The project's actual goal, one paragraph, honestly scoped: this is
  architecture research toward eventually supporting personality-
  conditioned play (distinct playing styles at equal skill, coaching
  that visibly changes behavior) for a coaching/idol game concept — but
  state plainly that as of this spec, none of the personality/coaching
  work has been built yet; everything in `go-style-engine/` so far is
  board-representation and training-architecture research.
- The two style-conditioning mechanisms that already exist and how
  they relate: `StyleKnobs` (`engine/mcts.py`, search-time re-ranking,
  net-agnostic, works with either `GoZeroNet` or `TokenTransformerNet`)
  vs. the Transformer's `pos_mode` variants (`absolute`/`relative`/
  `gab`/`gab_absolute`/etc., an architecture-level concern, not a
  style-conditioning mechanism itself — don't conflate the two).

### `docs/style-guide.md`

- This is a research codebase: prefer clear, well-commented empirical
  findings in doc `.md` files over terse code comments; code itself
  follows plain PEP 8-ish conventions already visible in `engine/`
  (no enforced formatter/linter currently configured — note that
  explicitly rather than inventing a convention that isn't enforced).
- Every new positional-info mechanism, training lever, or CLI flag
  gets: a docstring/comment explaining *why* (not just what), a test,
  and — if it's a genuine empirical finding, not just plumbing — a
  section in `go-style-engine/docs/board-specification.md` with real
  numbers, not just an implementation.
- Commit message convention already established: explain what was
  found/why, not just what changed — see recent `git log` in
  `go-style-engine/` for the actual pattern in practice.

### `docs/common-pitfalls.md`

Known, previously-hit traps, each with a one-line description and
where the fix lives:

- macOS's default `/bin/bash` is 3.2 (pre-4.4): expanding an *empty*
  array under `set -u` throws "unbound variable" — use plain strings
  with unquoted word-splitting instead of arrays for optional flags in
  shell scripts (see `run_generations.sh`'s header comment).
- `train.py`: a small enough dataset can make
  `int(num_examples * val_fraction)` round down to 0, and calling
  `evaluate()` on an empty val set divides by zero — fixed by bumping
  to 1 example when `num_examples >= 2`, falling back to no split
  otherwise (see `tests/test_train_early_stopping.py`'s
  `TestSmallDatasetValSplitRounding`).
- `nn.MultiheadAttention`'s fused "fastpath" kernel returns NaN in
  `.eval()` mode with a float `attn_mask`, but only *after* real
  training steps, never with a fresh model or in `.train()` mode —
  fixed via `torch.backends.mha.set_fastpath_enabled(False)` in
  `engine/token_transformer.py`.
- Comparing training runs at a fixed epoch count understates
  configurations that overfit faster (see
  `go-style-engine/docs/board-specification.md` §8b) — always compare
  at each config's own best epoch (`train.py --early-stopping-patience`
  now supports this directly), not a fixed epoch.
- `docs/unattended-generation-loop.md` (if it exists locally) is
  gitignored on purpose — it names a personal sailbox ID and must never
  be re-added to tracking; see `.gitignore`.

## Non-Goals

- Do not write new code, tests, or fix any bug while doing this — this
  spec is documentation-only, sourced from what's already true in the
  tree.
- Do not invent conventions that aren't actually enforced (e.g. don't
  claim a linter/formatter is required if none is configured).
- Do not duplicate `go-style-engine/docs/board-specification.md`'s
  content — point to it, don't restate its findings.
- Do not touch `go-style-engine/README.md` or any file under
  `go-style-engine/docs/` — those are already accurate and current;
  this spec only creates the five root-level `docs/*.md` stubs.

## Likely Files

- `docs/architecture.md` (new)
- `docs/testing.md` (new)
- `docs/domain.md` (new)
- `docs/style-guide.md` (new)
- `docs/common-pitfalls.md` (new)

## Environment

No credentials, services, or fixtures needed. Requires read access to
the full repo tree (already checked out) and the ability to run
`cd go-style-engine && .venv/bin/python -m pytest -q` to verify the
real current pass count before writing it into `docs/testing.md`
(the `.venv` must already exist — if missing, `python3 -m venv .venv &&
.venv/bin/pip install -r requirements.txt` first, or the unpinned
fallback noted above if the Python version is 3.11 or below 3.12).

## Dependencies

None. This is the first spec for this repo.

## Edge Cases

- If the actual pytest pass/fail/xfail count differs from what's
  written above (132 passed, 7 xfailed as of this spec's authoring),
  use the real observed count, not the number in this spec — it will
  drift as the repo changes.
- If `docs/unattended-generation-loop.md` doesn't exist locally for
  whoever executes this spec (it's gitignored, personal, may not be
  present on every machine/box), skip that bullet in
  `common-pitfalls.md` rather than asserting a file that isn't there.

## Test Expectations

- No new tests — this is a documentation-only spec.
- Verification is manual: each doc's factual claims (file paths, test
  commands, line references) must be checked against the actual tree
  before the spec is marked `State: done`.

## Acceptance Criteria

- [x] All five `docs/*.md` files exist with real, repo-specific content
      per the outlines above (not generic boilerplate).
- [x] Every file path, command, and line-number reference in the new
      docs actually resolves in the current tree (spot-checked
      directly: xfail convention, `.gitignore` rule, `papers/README.md`,
      `board-specification.md`'s §8a-§8e, all five new files present).
- [x] `cd go-style-engine && .venv/bin/python -m pytest -q` was
      actually run (2026-09-03), and its real output — `132 passed,
      7 xfailed` — is what's recorded in `docs/testing.md`, not a
      number copied from this spec's own illustrative text.
- [x] `CHANGELOG.md` updated.
- [x] This spec's `State:` set to `done` in the same commit as the docs.

## Known Risks

- Low risk — documentation-only, no code paths touched, nothing to
  regress. The main risk is staleness: these docs will drift as
  `go-style-engine/` keeps changing, and nothing currently keeps them
  in sync automatically. Whoever next hits a stale doc should correct
  it in place rather than let inaccuracy compound (per
  `HARNESS_PRINCIPLES.md`'s "garbage collect continuously").

## Decision Log

| # | Proposed | Status | Why |
|---|---|---|---|
| 1 | Execute this spec directly (fast, free, no delegation exercised) vs. delegate it via `sail-subs` (slower, first real trial of the pipeline `001-adopt-sail-delegation.md` set up). | **Accepted: `sail-subs`** | Repo owner: "settle that." Deliberately low-risk (docs-only, easy to review) and self-contained enough to be a fair first `sail-subs` trial — exactly the task this pipeline was built to prove out. |
| 2 | First `sail-subs` attempt (delegation `d-20260903-7f0ec231384e`), after fixing the stale-`CLAUDE_PROJECT_DIR` blocker (see `TODO.md`). | **Failed — real, not infra** | The delegation launched correctly (setup command passed, repo checkout worked), but the worker made zero edits and returned a generic "I don't see a specific question in your message" response — it never engaged with the task, `AGENTS.md`, `AGENT_LOOP.md`, or this spec at all, despite using only 3 of a 24-turn budget. `required_checks` failed (no docs created). Likely a worker-model comprehension failure (default worker: DeepSeek V4 Flash 0731, a fast/cheap default) on a moderately long, context-heavy task prompt, not a pipeline or environment problem. |
| 3 | Second `sail-subs` attempt (delegation `d-20260903-16bad6e7e4a5`), with a much shorter, terser task prompt, to isolate whether prompt length was the cause of #2. | **Failed differently — worse** | This time the worker genuinely explored the repo (`ls`, `git log`, checked authors/Python version, **read `papers/README.md`**), using 12 of 24 turns and 34 tool calls — but consumed **271,074 input tokens** (114,944 cached) doing it, then produced completely incoherent, repetitive gibberish as its final summary and made zero file edits. `required_checks` failed again. The token count is the real clue: `papers/` holds several multi-MB PDFs, git-tracked — if the worker's exploration pulled PDF content into context as raw text, that plausibly explains both the token blowout and the coherence collapse in a fast/cheap default model. Not confirmed, but a strong, specific, testable hypothesis (see `TODO.md`). Concluded two failures with two different failure modes, real token cost on the second, is enough evidence to stop retrying blind and fall back to direct execution for this spec — see #4. |
| 4 | Retry a third time (e.g. with `model_role="implementation"` or an explicit stronger `model`) vs. execute this spec directly instead. | **Accepted: execute directly** | Two failed attempts, meaningfully different failure modes, real cost already spent (271K tokens on attempt 2 alone) — further blind retries aren't a good use of either compute or time. `TODO.md` tracks the `papers/`-PDF hypothesis as a real follow-up to test *before* trying `sail-subs` again on this repo, rather than repeating the same likely mistake a third time. |
