# Get trans-go-former to play semi-good 9x9 Go: kifu-pretrain, then self-play refine

Priority: high
State: done

## Problem

Every checkpoint in this repo so far exists to answer an architecture
question (§7/§8's `pos_mode` comparisons, §8g's patience investigation)
— none was trained with the goal of actually playing well. The repo
owner has now set the real near-term goal explicitly: get
`trans-go-former` (the token-transformer path, not `GoZeroNet`) to
play semi-good-enough Go, as phase 1, before phase 2 (natural-language
"skill"-style conditioning of playing style, not yet scoped).

§8d/§8e found random-net self-play alone plateaus (the generator never
improves, so more self-play data doesn't help). §8f/§8g found real
human kifu data *does* teach the value function, contrary to §8f's
original (wrong, corrected) conclusion — but only ~80-110 games exist
per board size, likely not enough alone for real strength.
`run_generations.sh` (from §9's `003`) exists and works but has only
ever been run from random init, the exact regime §8d/§8e showed
plateaus.

The synthesis this spec runs: combine both correctly-understood pieces
— kifu pretraining to escape the random-init generator problem, then
self-play refinement to go beyond what ~80 games alone can teach.

## Desired Behavior

**Board size: 9x9** (decided with the repo owner — cheaper/faster
iteration, matches every prior experiment in this repo; 13x13 is a
follow-up once this recipe is shown to work at all).

1. **Pretrain** on the existing `runs/kifu_exp` (80 human 9x9 games,
   3,878 positions, `Specs/009`), `pos_mode='both'` (not
   `gab_absolute` — see Decision Log #1), proper patience per §8g's
   correction (not self-play-tuned `patience=3`):
   ```
   train.py --experience runs/kifu_exp --board-size 9 --net-type token \
     --pos-mode both --seed 0 --val-fraction 0.2 \
     --epochs 60 --early-stopping-patience 15 \
     --out-checkpoint checkpoints/base_kifu9x9.pt
   ```
2. **Extend `run_generations.sh`** to accept an initial checkpoint
   (currently `PREV_CKPT=""` is hardcoded before the generation loop,
   always random-init for generation 1) via an `INIT_CKPT` env var,
   defaulting to empty (unchanged behavior when unset — this is a
   strict extension, not a behavior change for existing callers).
3. **Run the refinement loop**, seeded from step 1's checkpoint,
   `POS_MODE=both` (matching step 1 — switching `pos_mode` mid-lineage
   would invalidate the warm start), proper patience:
   ```
   INIT_CKPT=checkpoints/base_kifu9x9.pt POS_MODE=both PATIENCE=15 \
     BOARD_SIZE=9 ./run_generations.sh 8
   ```
4. **Evaluate** the final generation's checkpoint two ways:
   - Qualitative: play/inspect a handful of games (`trace_move.py` or
     a small `selfplay.py` run), check for basic non-degenerate play
     (not passing immediately, plausible-looking shape, actually
     completing captures) — the same "what changes with a trained
     brain" checklist `docs/mcts-trace-walkthrough.md` already uses
     for `GoZeroNet`.
   - Quantitative: a small tournament (final generation vs. generation
     1, and vs. a random-init baseline) to get an actual win-rate
     number — loss curves alone don't establish playing strength.

## Non-Goals

- Not 13x13 — 9x9 first, per the board-size decision above.
- Not `GoZeroNet`/`StyleKnobs` — that's `Specs/007`'s track,
  deliberately separate; this spec is entirely the token-transformer
  line, since the repo owner specifically wants "trans-go-former" to
  be the thing that plays.
- Not the natural-language style-conditioning phase 2 — explicitly
  deferred, not yet scoped, mentioned here only for context on why
  phase 1 matters.
- Not resolving `gab_absolute`'s divergence problem (§8g) — sidestepped
  by using `pos_mode='both'` instead, not solved. A separate follow-up
  if `gab_absolute` turns out to matter later.
- Not accumulating a growing self-play replay buffer across
  generations — `run_generations.sh`'s existing documented
  simplification (warm-starts weights, not data, each generation)
  stays as-is; revisit only if this run's result justifies the added
  complexity.

## Likely Files

- `go-style-engine/checkpoints/base_kifu9x9.pt` (new, gitignored)
- `go-style-engine/run_generations.sh` (small edit: `INIT_CKPT` env var)
- `go-style-engine/runs/gen_loop/` (new run, gitignored)
- `go-style-engine/docs/board-specification.md` (result written up,
  following the §8a-§8g pattern) or a new doc section specifically
  about playing strength, if the result doesn't fit naturally into
  the architecture-validation narrative §8 already tells
- `CHANGELOG.md`

## Environment

CPU-only. Pretraining is cheap (seconds, per this session's own kifu
runs). The 8-generation refinement loop at 9x9/30 games/50 rounds is
the same cost regime `Specs/003` already estimated as "well under an
hour" — can run locally.

## Dependencies

Builds directly on `Specs/009` (kifu data + pipeline) and `Specs/003`
(the generation-loop script) — both `done`. Independent of `004`-`008`.

## Edge Cases

- If `run_generations.sh`'s self-play step (guided by a real,
  non-random checkpoint for the first time) produces meaningfully
  different game lengths/shapes than every prior random-init run in
  this repo, that's expected and worth noting, not a bug to chase.
- If the refinement loop's val losses look *worse* than the pretrained
  base checkpoint alone (self-play-generated data diluting real
  signal with noisier targets), that's a real, useful negative result
  — report it plainly rather than tuning around it blindly.
- Small dataset caveat carries over from `Specs/009`: 80 games is
  small; treat any single run's result as directional, not definitive.

## Test Expectations

- No new automated tests for the training runs themselves (experiment,
  not code). `run_generations.sh`'s `INIT_CKPT` addition should get a
  quick manual smoke check (does it correctly skip the `--checkpoint`/
  `--in-checkpoint` flags when unset, matching existing behavior) —
  automated coverage optional given the script's existing smoke-test
  precedent, not a hard requirement.
- `cd go-style-engine && .venv/bin/python -m pytest -q` — full suite
  still passes (no `engine/`/`train.py`/`selfplay.py` changes expected).

## Acceptance Criteria

- [x] `checkpoints/base_kifu9x9.pt` exists, trained per step 1. (Real
      policy/value tradeoff found and documented, not silently
      resolved — see `board-specification.md` §11.)
- [x] `run_generations.sh` accepts `INIT_CKPT`, unchanged behavior
      when unset.
- [x] 8-generation refinement loop completes, seeded from the
      pretrained checkpoint.
- [x] Qualitative game review completed and recorded: sane, varied
      opening/midgame play; a specific, real endgame-termination
      weakness found (games often run to the move cap instead of
      double-passing) and reported, not glossed over.
- [x] A concrete win-rate number exists — two, in fact: gen8 vs.
      random-init (73.3%, clear signal) and gen8 vs. gen1 (46.2%,
      no clear improvement). A determinism bug in the first evaluation
      attempt (identical games per color, §7's own already-documented
      pitfall recurring) was caught and fixed before either number was
      trusted.
- [x] Result written up in `board-specification.md` §11, evidence-first
      style, including the honest negative half of the result
      (generations 5-8 plateaued/regressed) and a concrete, diagnosed
      next lever (accumulate a self-play replay buffer across
      generations — `run_generations.sh`'s own documented
      simplification, not a new open question).
- [x] `CHANGELOG.md` updated.

## Known Risks

Medium. The core uncertainty is whether ~80 human games is enough of
a seed for self-play refinement to meaningfully compound on — it might
not be, in which case the result is "kifu-pretrain helps some, but
still plateaus," itself a useful, honest finding pointing at "need
more kifu data" as the next real lever (this time for strength, not
just as a §8f-style architecture-comparison footnote). Not a spec
failure either way, per this whole doc's evidence-first convention.

## Decision Log

| # | Proposed | Status | Why |
|---|---|---|---|
| 1 | Use `pos_mode='gab_absolute'` (§8's overall "best" per the pre-correction narrative) vs. `'both'` for this pretrain-then-refine effort. | **Accepted: `'both'`** | `gab_absolute` has an unresolved catastrophic policy-divergence problem on real kifu data (§8g) — untested whether it ever recovers. `'both'` is proven to learn cleanly on real data at both board sizes tried, with no open questions blocking it. The goal here is "get something that plays," not "use the theoretically most sophisticated positional mechanism" — resolving GAB's real-data behavior is a separate, deferred investigation. |
| 2 | Board size 9x9 vs. 13x13 (the project's actual target) for this first playing-strength push. | **Accepted: 9x9** | Decided with the repo owner. Cheaper/faster iteration to validate the pretrain-then-refine recipe works at all before spending 13x13-scale compute on it. |
| 3 | Accumulate a growing self-play replay buffer across generations (real AlphaZero-style) vs. keep `run_generations.sh`'s existing per-generation-fresh-data simplification. | **Rejected for now** | Bigger scope change than this spec needs to test the core hypothesis (does kifu-seeded refinement beat random-init refinement); revisit only if this run's result justifies the added complexity, per `run_generations.sh`'s own existing documented stance. |
