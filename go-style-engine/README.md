# go-style-engine

A PyTorch, CPU-only port of the AlphaGo-Zero-style engine from
[maxpumperla/deep_learning_and_the_game_of_go](https://github.com/maxpumperla/deep_learning_and_the_game_of_go)
(MIT licensed; `engine/gotypes.py`, `engine/zobrist.py`, `engine/goboard.py`,
`engine/scoring.py`, `engine/encoder.py` are ported from that repo's
`dlgo/` package with minimal changes — see file headers). The MCTS agent
(`engine/mcts.py`) and network (`engine/network.py`) are original PyTorch
implementations sized for **13x13** rather than the book's 9x9 demos, per
the "9x9 is too cramped to express a style" call — see
`specs/2026-08-31-embedded-go-engine-research.md` for the related KataGo
on-device spike, which this is a complementary, cheaper alternative to
(no C++/CoreML bridge, no GPU, trains from scratch).

## Docs

- `docs/mcts-trace-walkthrough.md` — a from-zero explanation of how the
  engine picks a move, built around an instrumented run of
  `trace_move.py`. Start here if the search/knob machinery is new to you.

## What this is for

Before porting anything to Swift, this answers one question empirically:
**do the coaching "style knobs" (pace, candidate breadth, complexity
preference, risk tolerance) actually produce statistically distinguishable
match telemetry** — the win-probability curve, junction density, think
time — or is the mechanism too weak/noisy to show up at this board size
and net capacity? See `selfplay.py`'s module docstring for the knob
definitions and how they map onto the PRD's Voice/Psychology state
families.

This is not the app's real match-simulation engine. It's a standalone
research/validation harness, run from the command line, that produces JSON
telemetry logs. Nothing here is wired into the iOS app or the
`PrototypeMatch`/`TelemetryPoint`/`Junction` contracts described in
`specs/2026-08-31-coaching-interaction-layer-design.md` — that's a
deliberate follow-up decision, not an oversight.

## Setup

```bash
cd research/go-style-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running self-play

```bash
python selfplay.py --board-size 13 --games 20 \
    --black-preset baseline --white-preset fighter \
    --rounds-per-move 200 --out runs/baseline_vs_fighter/
```

`--out` is now a RUN DIRECTORY, not a single JSON file. The run writes:

- `<out>/manifest.json` -- full args, seed, git SHA, torch version, start
  timestamp, board size, presets, resolved knob dicts.
- `<out>/games.jsonl` -- one JSON line per completed game, flushed
  immediately (so a laptop shutdown mid-run loses only the in-flight
  game, not the whole run).
- `<out>/experience/game_{idx:04d}.npz` -- one experience shard per game
  (always written; contains state/visit-count/reward triples).
- `<out>/summary.json` -- written on full completion; top-level info plus
  per-game index references.

Runs are **resumable**: if `<out>/games.jsonl` already exists at start,
already-completed game indices are skipped (the loop still runs
`range(args.games)` but skips completed indices). If args disagree with
the manifest (different games count, board size, presets, seed,
checkpoint), a loud warning is printed but the CURRENT args are used.

`--save-experience` still controls whether a final combined .npz is
written (its value is the output path); shards are always written under
`<out>/experience/` regardless. At full completion, all shards are
combined into the requested path using the existing `combine_experience`
logic (loading shard arrays directly).

`PRESETS` in `selfplay.py` has one illustrative starting point per PRD
trainee archetype (`baseline`, `fighter`, `builder`, `adapter`, `anchor`).
These are placeholders, not calibrated numbers — the point of running this
is to check whether settings like these separate in the output, then tune
from what the data shows, not to treat the presets as final.

Any individual knob can be overridden on the command line
(`--complexity-weight`, `--safety-lambda`, `--c`, `--temperature`,
`--dirichlet-epsilon`, `--rounds-per-move`) and applies to both sides.

With `--checkpoint` omitted, the net is randomly initialized — fine for
checking that the search mechanism itself produces separable statistics,
but the policy/value estimates won't reflect real Go skill. Self-play
training (below) is what makes the checkpoint actually play.

## Self-play training (optional, to get a non-random checkpoint)

```bash
python selfplay.py --board-size 13 --games 200 \
    --black-preset baseline --white-preset baseline \
    --rounds-per-move 200 --out runs/gen0/ \
    --save-experience runs/gen0_exp.npz

python train.py --experience runs/gen0_exp.npz --board-size 13 \
    --out-checkpoint checkpoints/gen1.pt --epochs 5
```

Or -- if `runs/gen0/` is a run directory with shards -- train.py can
read it directly:

```bash
python train.py --experience runs/gen0/ --board-size 13 \
    --out-checkpoint checkpoints/gen1.pt --epochs 5
```

`train.py` now supports `--seed` (default 0): both torch and numpy RNGs
are seeded before training, making runs reproducible. Checkpoints are
saved after EACH epoch (`<out-stem>.epoch{N}.pt`) as well as the final
`--out-checkpoint`. A `<out-checkpoint>.meta.json` sidecar records args,
seed, in-checkpoint path, experience path(s), git SHA, torch version,
final epoch losses, and timestamp.

Repeat with `--in-checkpoint checkpoints/gen1.pt` to keep improving.
Per-trainee "Voice" checkpoints (see the earlier design discussion) are
made the same way: generate self-play games with that trainee's knob
preset, then fine-tune a copy of the base checkpoint on that data --
`--in-checkpoint checkpoints/gen1.pt --out-checkpoint checkpoints/fighter.pt`.

## Sizing notes

Default network is 6 residual blocks, 64 channels (`GoZeroNet` in
`engine/network.py`) — the size judged appropriate for 13x13 in the prior
design discussion (9x9's 4-conv-layer net is too shallow to see across a
13x13 board; 19x19's 20+ block towers are a different cost regime
entirely, deferred). `--channels`/`--blocks` are exposed on both scripts
if that needs revisiting once real timing/quality numbers come back.

Self-play is single-threaded and CPU-only by design (matches the "no
GPU" constraint). It is not fast — treat `--games` counts as something to
run in the background, not interactively.
