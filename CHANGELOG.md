# Changelog

Repository changes worth human review should be summarized here.

## Unreleased

- Executed `Specs/009`: built `go-style-engine/data/kifu_to_experience.py`
  (go-gibo-ingestion JSONL -> train.py-compatible `.npz`, human-only
  80-game default) and re-ran §7/§8's GAB comparison on real human
  kifu instead of self-play data. Real finding, not the expected
  noisy-but-directional read: the value head fails to learn *at all*
  on real human data in every `pos_mode`/`gab_gen_size` config tried
  (train loss flat at the constant-predictor baseline), sharply
  contradicting §7/§8's clean self-play-data value learning —
  documented in `docs/board-specification.md` §8f, with a stated,
  untested hypothesis (self-play from a weak generator may make the
  value target easier than real, closely-contested human games) and a
  concrete next lever (more human game data). `Specs/009` marked
  `done`.
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
