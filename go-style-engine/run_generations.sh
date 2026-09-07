#!/usr/bin/env bash
set -euo pipefail

# Iterative self-play/train loop -- generation N's self-play is guided by
# generation (N-1)'s trained checkpoint instead of random weights. Tests
# docs/board-specification.md's §8e conclusion: the random, never-improving
# self-play generator (not gab_gen_size or data volume) was the actual
# ceiling on every §8 result. If that's right, generations should show a
# real, improving trend in val_policy_loss/val_value_loss; if it isn't,
# the trend will be flat, same as §8d's "more data didn't help" result.
#
# Buffer accumulation (default ON, BUFFER=0 to disable): each
# generation's fresh self-play shards are copied into a persistent
# $OUT_DIR/buffer/ directory (generation-prefixed filenames, so nothing
# from an earlier generation is overwritten), and every generation
# trains on the WHOLE buffer so far -- not just its own fresh batch --
# warm-started from the previous generation's weights, same as before.
# This replaced an earlier version of this script that discarded each
# generation's data after one use; Specs/011 found that version's gains
# stalled/regressed after generation 4 as later generations trained on
# a shrinking dataset (games resolving faster = fewer positions/game),
# not a growing one. BUFFER=0 restores the original discard-each-
# generation behavior, for comparison.
#
# Usage:
#   ./run_generations.sh [NUM_GENERATIONS]
#
# All settings below default to §8's settled config
# (docs/board-specification.md §8e) and can be overridden via env vars,
# e.g.: BOARD_SIZE=13 GAMES=60 ./run_generations.sh 8
#
# WORKERS (default: 1, sequential, unchanged behavior): play this many
# self-play games in parallel per generation (selfplay.py --workers).
# engine/mcts.py evaluates one board at a time with no batching, so a
# bigger/multi-core box only actually helps once this is set > 1 --
# GPU doesn't help this bottleneck either, since it's many tiny
# sequential calls, not one big batch.
#
# INIT_CKPT (default: unset, i.e. random init, the original behavior):
# seed generation 1's self-play from an existing checkpoint instead of
# random weights -- e.g. a kifu-pretrained base (Specs/011), so
# refinement starts from real signal instead of the random-generator
# regime §8d/§8e showed plateaus. Must match POS_MODE/GAB_GEN_SIZE/
# GAB_INTERMEDIATE_DIM/DROPOUT/BOARD_SIZE exactly, same as any
# --in-checkpoint/--checkpoint use elsewhere in this codebase --
# mismatched architecture will fail to load, not silently misbehave.
# GAB_GEN_SIZE and GAB_INTERMEDIATE_DIM are two DIFFERENT parameters
# (GeometricAttentionBias's `summarize` output width vs. `generate`'s)
# that both default to 16 here but do not have to match each other --
# a checkpoint trained by hand with only --gab-gen-size set (leaving
# --gab-intermediate-dim at train.py's own 64 default) needs
# GAB_INTERMEDIATE_DIM=64 here explicitly, not left to follow
# GAB_GEN_SIZE.
#
# START_GEN (default: 1): resume an interrupted run at a specific
# generation number instead of restarting from 1 -- combine with
# INIT_CKPT pointing at the last completed generation's checkpoint and
# the SAME OUT_DIR/CKPT_DIR (so the existing buffer/ is reused, not
# recreated) to continue exactly where a stopped run left off, without
# clobbering or discarding the generations already done.
#
# BUFFER_WINDOW (default: 0, unbounded -- unchanged behavior): the
# buffer's own real cost. Accumulating every generation's games
# forever means training set size grows without limit, so per-
# generation TRAINING time grows without limit too -- a real run
# (Specs/011's follow-up, a Sail box) took 36+ minutes to train
# generation 10 alone once the buffer reached 600 games, versus
# seconds at generation 2's ~120 games, with no sign of leveling off.
# BUFFER_WINDOW=K keeps only the most recent K generations' shards in
# the buffer (older ones deleted after each generation, real
# AlphaZero-style replay-buffer eviction, not just a read-side
# filter) -- bounding per-generation training cost roughly flat
# indefinitely, at the cost of the model no longer training on
# arbitrarily old self-play data. Requires BUFFER=1 (the default).
#
# TEMP_CUTOFF (default: unset, disabled -- unchanged behavior): passed
# straight through as selfplay.py --temperature-cutoff. Added after
# real qualitative play-testing rated the resulting checkpoints
# "18k-level" -- every self-play game used temperature=1.0/dirichlet
# noise for its ENTIRE length (this script's own self-play call always
# passed --dirichlet-epsilon 0.25 --temperature 1.0 unconditionally),
# meaning the training data stayed noisy in the midgame/endgame too,
# not just the opening, unlike standard AlphaZero-style annealing.
# TEMP_CUTOFF=N switches both self-play sides to greedy after N plies.

NUM_GENERATIONS="${1:-5}"
BOARD_SIZE="${BOARD_SIZE:-9}"
GAMES="${GAMES:-30}"
WORKERS="${WORKERS:-1}"
ROUNDS="${ROUNDS:-50}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-5}"
POS_MODE="${POS_MODE:-gab_absolute}"
GAB_GEN_SIZE="${GAB_GEN_SIZE:-16}"
GAB_INTERMEDIATE_DIM="${GAB_INTERMEDIATE_DIM:-$GAB_GEN_SIZE}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
DROPOUT="${DROPOUT:-0.1}"
OUT_DIR="${OUT_DIR:-runs/gen_loop}"
CKPT_DIR="${CKPT_DIR:-checkpoints/gen_loop}"
SEED_BASE="${SEED_BASE:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
INIT_CKPT="${INIT_CKPT:-}"
BUFFER="${BUFFER:-1}"
BUFFER_WINDOW="${BUFFER_WINDOW:-0}"
TEMP_CUTOFF="${TEMP_CUTOFF:-}"
START_GEN="${START_GEN:-1}"

mkdir -p "$OUT_DIR" "$CKPT_DIR"
SUMMARY="$OUT_DIR/summary.jsonl"
BUFFER_DIR="$OUT_DIR/buffer"
if [ "$BUFFER" -eq 1 ]; then
  mkdir -p "$BUFFER_DIR"
fi

echo "Starting $NUM_GENERATIONS-generation loop:"
echo "  board_size=$BOARD_SIZE games=$GAMES rounds=$ROUNDS epochs=$EPOCHS patience=$PATIENCE"
echo "  pos_mode=$POS_MODE gab_gen_size=$GAB_GEN_SIZE dropout=$DROPOUT"
echo "  out_dir=$OUT_DIR ckpt_dir=$CKPT_DIR"
echo "  summary=$SUMMARY"
echo

PREV_CKPT="$INIT_CKPT"
for gen in $(seq "$START_GEN" "$NUM_GENERATIONS"); do
  GEN_TAG="gen${gen}"
  SEED=$((SEED_BASE + gen))
  SP_OUT="$OUT_DIR/${GEN_TAG}_selfplay"
  EXP_FILE="$OUT_DIR/${GEN_TAG}_exp.npz"
  CKPT_FILE="$CKPT_DIR/${GEN_TAG}.pt"

  echo "=== Generation $gen/$NUM_GENERATIONS: self-play (seed=$SEED, checkpoint=${PREV_CKPT:-random init}) ==="
  # Plain strings + unquoted word-splitting, not arrays: macOS's default
  # /bin/bash is 3.2 (pre-4.4), where expanding an EMPTY array under
  # `set -u` throws "unbound variable" -- a real bug this script hit
  # during its own smoke test. Safe here since none of these paths
  # contain spaces.
  CKPT_FLAG=""
  if [ -n "$PREV_CKPT" ]; then
    CKPT_FLAG="--checkpoint $PREV_CKPT"
  fi
  TEMP_CUTOFF_FLAG=""
  if [ -n "$TEMP_CUTOFF" ]; then
    TEMP_CUTOFF_FLAG="--temperature-cutoff $TEMP_CUTOFF"
  fi
  "$PYTHON" selfplay.py --board-size "$BOARD_SIZE" --games "$GAMES" \
    --rounds-per-move "$ROUNDS" --net-type token --pos-mode "$POS_MODE" \
    --gab-gen-size "$GAB_GEN_SIZE" --gab-intermediate-dim "$GAB_INTERMEDIATE_DIM" \
    --dropout "$DROPOUT" --dirichlet-epsilon 0.25 --temperature 1.0 --workers "$WORKERS" \
    --seed "$SEED" $CKPT_FLAG $TEMP_CUTOFF_FLAG \
    --out "$SP_OUT" --save-experience "$EXP_FILE"

  TRAIN_SRC="$EXP_FILE"
  if [ "$BUFFER" -eq 1 ]; then
    # Copy this generation's per-game shards into the persistent
    # buffer, renamed so every generation's files stay unique and so
    # the name still starts with "game_" (train.py's directory loader
    # requires that prefix -- see its _load_experience docstring).
    for shard in "$SP_OUT"/experience/game_*.npz; do
      [ -e "$shard" ] || continue
      base="$(basename "$shard")"
      cp "$shard" "$BUFFER_DIR/game_gen${gen}_${base#game_}"
    done

    if [ "$BUFFER_WINDOW" -gt 0 ]; then
      # Evict shards from generations older than the window -- real
      # eviction (files removed), not just excluded from this
      # generation's read. Shard names are "game_gen<N>_<rest>.npz";
      # extract N and compare against the oldest generation still
      # allowed in the window.
      min_gen=$((gen - BUFFER_WINDOW + 1))
      if [ "$min_gen" -gt 0 ]; then
        pruned=0
        for f in "$BUFFER_DIR"/game_gen*_*.npz; do
          [ -e "$f" ] || continue
          fbase="$(basename "$f")"
          shard_gen="${fbase#game_gen}"
          shard_gen="${shard_gen%%_*}"
          if [ "$shard_gen" -lt "$min_gen" ] 2>/dev/null; then
            rm -f "$f"
            pruned=$((pruned + 1))
          fi
        done
        if [ "$pruned" -gt 0 ]; then
          echo "Pruned $pruned shard(s) from generations before $min_gen (BUFFER_WINDOW=$BUFFER_WINDOW)."
        fi
      fi
    fi

    TRAIN_SRC="$BUFFER_DIR"
    n_shards=$(find "$BUFFER_DIR" -name '*.npz' | wc -l | tr -d ' ')
    echo "Buffer now has $n_shards shards (through generation $gen)."
  fi

  echo "=== Generation $gen/$NUM_GENERATIONS: train ==="
  INCKPT_FLAG=""
  if [ -n "$PREV_CKPT" ]; then
    INCKPT_FLAG="--in-checkpoint $PREV_CKPT"
  fi
  "$PYTHON" train.py --experience "$TRAIN_SRC" --board-size "$BOARD_SIZE" \
    --net-type token --pos-mode "$POS_MODE" --gab-gen-size "$GAB_GEN_SIZE" \
    --gab-intermediate-dim "$GAB_INTERMEDIATE_DIM" --dropout "$DROPOUT" \
    --grad-clip "$GRAD_CLIP" \
    $INCKPT_FLAG --val-fraction 0.2 --epochs "$EPOCHS" \
    --early-stopping-patience "$PATIENCE" --seed "$SEED" \
    --out-checkpoint "$CKPT_FILE"

  "$PYTHON" - "$CKPT_FILE" "$gen" "$SUMMARY" <<'PYEOF'
import json
import sys

ckpt_file, gen, summary_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
meta = json.load(open(f'{ckpt_file}.meta.json'))
losses = meta['final_epoch_losses']
best = next(r for r in losses if r['epoch'] == meta['best_epoch'])
record = {
    'generation': gen,
    'checkpoint': ckpt_file,
    'best_epoch': meta['best_epoch'],
    'val_policy_loss': best.get('val_policy_loss'),
    'val_value_loss': best.get('val_value_loss'),
}
with open(summary_path, 'a') as f:
    f.write(json.dumps(record) + '\n')
print(f'Generation {gen} summary:', record)
PYEOF

  PREV_CKPT="$CKPT_FILE"
  echo
done

echo "Loop complete ($NUM_GENERATIONS generations)."
echo "Per-generation summary ($SUMMARY):"
cat "$SUMMARY"
