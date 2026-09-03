# Split kifu fetching/parsing into a separate go-gibo-ingestion repo

Priority: high
State: done

## Problem

`Specs/009` originally planned to own fetching CWI's Go game archives
and parsing SGF directly inside `go-style-engine/data/`. Partway
through drafting `009`, the repo owner raised a broader question: this
project is likely to need a *consistent* data ingestion pattern for
kifu, not a one-off script bolted onto the model repo — pointing at
`~/repositories/individual/nem-forecast-orchestration` as a precedent,
which pulls ingestion (cron feeds, dbt bronze/silver/gold transforms,
Dagster orchestration) out of `nem-forecast` entirely, into its own
repo, for the same reason: ingestion concerns don't belong in a
model/API repository.

This is a retroactive spec (per `Specs/README.md`'s Decision Log
convention) documenting that decision and the resulting build, written
after the work was done — the decision otherwise would only exist in
chat history, which `HARNESS_PRINCIPLES.md`'s "repository knowledge is
the system of record" rule treats as not actually decided.

## Desired Behavior

Create `~/repositories/individual/go-gibo-ingestion` as a standalone
sibling repo, decoupled from `trans-go-former`, that:

- Fetches Go game archives from public sources (currently CWI's
  `homepages.cwi.nl/~aeb/go/games/`) into a gitignored `data/raw/`.
- Parses SGF into a normalized, per-game JSON record (`game_id`,
  `category`, `board_size`, `komi`, `players`, `result`, `moves`,
  `has_variations`) written as `data/parsed/*.jsonl` — a
  project-neutral "silver" layer with no board-encoding or
  training-target logic, so any future consumer (not just
  `trans-go-former`) can build on it without re-parsing SGF.
- Stays deliberately lighter than `nem-forecast-orchestration`'s
  architecture: plain Python scripts (`fetch.py`, `sgf_reader.py`,
  `ingest.py`), no Dagster/dbt/DuckDB orchestrator, no bronze/silver/
  gold medallion layering beyond the one raw→parsed split. The
  precedent repo's orchestration weight is justified by several live,
  continuously-updating cron feeds needing scheduling and backfills;
  kifu ingestion today is one static, occasionally-fetched source, so
  matching that weight would be building for a problem that doesn't
  exist yet.

## Non-Goals

- Not a full `nem-forecast-orchestration`-style build (Dagster, dbt,
  DuckDB, bronze/silver/gold) — explicitly rejected for now, see
  Decision Log #1.
- Not a night-shift `Specs/`/`AGENTS.md` bootstrap for the new repo —
  out of scope for this spec; a follow-up if the repo grows enough to
  need its own spec-driven workflow independent of `trans-go-former`.
- Not a general-purpose SGF library — `sgf_reader.py` only handles
  what the CWI 9x9 archive's games actually need (main line only, no
  handicap support); see that module's own docstring.
- Does not decide which categories of games count as "usable" training
  data (e.g. excluding AI self-play) — that's left as consumer-side
  policy (`category` is preserved, not filtered, at ingest time); see
  `Specs/009`.

## Likely Files

All in the new repo, not `trans-go-former`:
`~/repositories/individual/go-gibo-ingestion/{fetch.py,sgf_reader.py,
ingest.py,README.md,requirements.txt,.gitignore,tests/}`.

In `trans-go-former`: `Specs/009` rewritten to consume this repo's
output instead of owning fetch/parse itself;
`go-style-engine/data/fetch_go_games.sh` removed (superseded);
`go-style-engine/data/README.md` updated to point here.

## Environment

CPU-only, stdlib-only beyond `pytest` for tests (no `numpy`/`torch`
dependency — this repo doesn't touch model code at all).

## Dependencies

None. Independent, standalone repo.

## Edge Cases

- **Coordinate convention mismatch**: `go-gibo-ingestion` emits
  0-indexed `(col, row)` per SGF's own convention; `trans-go-former`'s
  `engine/goboard.py` uses 1-indexed `Point(row, col)`. This is a
  deliberate seam, not something to unify across repos — documented in
  both `sgf_reader.py`'s docstring and `Specs/009`'s Edge Cases, with
  the mapping owned by `Specs/009`'s adapter script, not this repo.
- **`category` is not a human/AI label**: the CWI 9x9 archive's
  `Minigo` subdirectory (419/517 games) turned out to be AI self-play,
  not human kifu, discovered only by actually running the pipeline
  end-to-end and inspecting the category breakdown — not something
  that would have been caught by spec-writing alone. Documented
  prominently in the new repo's `README.md` specifically so a future
  consumer doesn't repeat the same wrong assumption.

## Test Expectations

- `cd ~/repositories/individual/go-gibo-ingestion && .venv/bin/python -m pytest -q`
  — 14 tests (SGF parsing edge cases: normal game, pass move, score vs.
  resign result, missing/malformed/unusable `RE[]`, non-9x9 `SZ[]`,
  handicap rejection, variation main-line extraction, escaped bracket/
  backslash values; batch ingestion: valid/invalid mix, category
  assignment, JSONL output).
- Verified against the real archive, not just synthetic fixtures:
  `fetch.py` + `ingest.py` run against the live CWI 9x9 archive parsed
  all 517 games with 0 rejections.

## Acceptance Criteria

- [x] `go-gibo-ingestion` repo created, git-initialized, first commit
      made.
- [x] `fetch.py` fetches the real CWI 9x9 archive (and, on request,
      the full 96,058-game archive).
- [x] `sgf_reader.py`/`ingest.py` parse SGF into normalized JSONL,
      verified against the real archive (517/517 parsed, 0
      rejections) as well as synthetic edge-case fixtures.
- [x] `README.md` documents usage, output schema, and the
      category/human-vs-AI finding.
- [x] `Specs/009` (in `trans-go-former`) updated to consume this
      repo's output instead of owning ingestion itself.
- [x] `trans-go-former`'s superseded `fetch_go_games.sh` removed,
      `data/README.md` repointed.
- [x] `CHANGELOG.md` (in `trans-go-former`) updated.

## Known Risks

Low, now that the build is done and verified. The main risk going
forward is drift between the two repos' assumptions (e.g. if
`go-gibo-ingestion`'s output schema changes without `Specs/009`'s
adapter being updated to match) — mitigated by keeping the schema
contract documented in one place (`go-gibo-ingestion/README.md`) that
`Specs/009` explicitly references rather than restating.

## Decision Log

| # | Proposed | Status | Why |
|---|---|---|---|
| 1 | Mirror `nem-forecast-orchestration`'s full architecture (Dagster, dbt, DuckDB, bronze/silver/gold) in the new repo now, vs. a much lighter plain-script pipeline, vs. not splitting the repo at all and keeping `009`'s original in-place plan. | **Accepted: light repo, split from trans-go-former** | The separation itself (ingestion out of the model repo) is worth adopting now — same reasoning as the precedent repo, and kifu sources are only going to grow. The orchestration *weight* of that precedent is not worth adopting yet — it's justified there by several live, continuously-updating feeds needing scheduling/backfills, which doesn't describe kifu ingestion today (one static, occasionally-fetched source). Revisit if/when a second live, scheduled source shows up. |
| 2 | Give the new repo its own night-shift `Specs/`/`AGENTS.md` bootstrap immediately, matching `trans-go-former`'s discipline, vs. defer that. | **Rejected for now** | The repo is small enough (3 scripts, ~250 lines) that the spec-preflight/review overhead isn't earning its cost yet; revisit once it grows past a size where ad hoc changes start risking drift or regressions. |
