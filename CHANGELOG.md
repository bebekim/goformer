# Changelog

Repository changes worth human review should be summarized here.

## Unreleased

- Split kifu fetching/SGF-parsing into a separate sibling repo,
  `go-gibo-ingestion`, rather than owning it in
  `go-style-engine/data/` — decouples ingestion from this
  model-training repo (`Specs/009`'s Decision Log #4). That repo's
  first real run against the CWI 9x9 archive found 419 of 517 games
  are Minigo AI self-play, not human kifu; only 80 are — `Specs/009`
  was re-scoped to a human-only default in response
  (`Specs/009`'s Decision Log #2).
- Filled the five root `docs/*.md` agent-knowledge stubs
  (`architecture.md`, `testing.md`, `domain.md`, `style-guide.md`,
  `common-pitfalls.md`) that `AGENTS.md` routes to — none existed
  before this (`Specs/002-bootstrap-knowledge-docs.md`). Executed
  directly after two failed `sail-subs` delegation attempts; see that
  spec's Decision Log and `TODO.md` for the failure detail and the
  `papers/*.pdf` context-bloat hypothesis.
