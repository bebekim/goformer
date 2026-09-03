# Changelog

Repository changes worth human review should be summarized here.

## Unreleased

- Executed `Specs/011`, the first attempt to get `trans-go-former`
  actually playing (not just architecture-comparison checkpoints):
  kifu-pretrain (`pos_mode='both'`, §8g's corrected patience) then
  8 generations of self-play refinement (`run_generations.sh`, now
  supporting `INIT_CKPT` to seed from a real checkpoint instead of
  random init). Result, honestly mixed: gen8 clearly beats a
  random-init net (73.3% win rate, +8.2 mean score margin) but does
  *not* clearly beat generation 1 (46.2%, a coin flip) — the loop's
  gains came from kifu-pretraining plus the first few generations,
  then plateaued/regressed as each generation trained on a shrinking,
  non-accumulating dataset (`run_generations.sh`'s own documented
  simplification). Qualitatively: sane opening/midgame play, but a
  real endgame-termination weakness (games often hit the move cap
  instead of double-passing). Full account in
  `docs/board-specification.md` §11, including a determinism bug in
  the first evaluation attempt (caught and fixed, same pitfall §7
  already documented once). Diagnosed next lever: accumulate a
  self-play replay buffer across generations.
- **Correction to the entry below**: the "value head fails to learn at
  all on real human data" finding was wrong. Extending to a real
  13x13 human dataset (110 games via `go-gibo-ingestion`'s newly
  board-size-agnostic pipeline) surfaced one config (`pos_mode='both'`)
  that broke through where others didn't, which traced back to
  `--early-stopping-patience 3` (tuned for self-play data in §8c)
  cutting most configs off epochs before a real, later breakthrough —
  not a genuine learning failure. Re-running `absolute` with a much
  longer patience budget confirmed the breakthrough at *both* board
  sizes, including the original 9x9 dataset. Full corrected account,
  the 13x13 extension, and what's still genuinely unresolved (whether
  `gab_absolute` shares this pattern) in `docs/board-specification.md`
  §8g; §8f kept as-written with a correction notice rather than
  rewritten, per this repo's evidence-first convention. `Specs/009`
  Decision Log #6.
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
