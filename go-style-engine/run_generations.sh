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
# INIT_CKPT (default: unset, i.e. random init, the original behavior):
# seed generation 1's self-play from an existing checkpoint instead of
# random weights -- e.g. a kifu-pretrained base (Specs/011), so
# refinement starts from real signal instead of the random-generator
# regime §8d/§8e showed plateaus. Must match POS_MODE/GAB_GEN_SIZE/
# DROPOUT/BOARD_SIZE, same as any --in-checkpoint/--checkpoint use
# elsewhere in this codebase -- mismatched architecture will fail to
# load, not silently misbehave.
#
# START_GEN (default: 1): resume an interrupted run at a specific
# generation number instead of restarting from 1 -- combine with
# INIT_CKPT pointing at the last completed generation's checkpoint and
# the SAME OUT_DIR/CKPT_DIR (so the existing buffer/ is reused, not
# recreated) to continue exactly where a stopped run left off, without
# clobbering or discarding the generations already done.

NUM_GENERATIONS="${1:-5}"
BOARD_SIZE="${BOARD_SIZE:-9}"
GAMES="${GAMES:-30}"
ROUNDS="${ROUNDS:-50}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-5}"
POS_MODE="${POS_MODE:-gab_absolute}"
GAB_GEN_SIZE="${GAB_GEN_SIZE:-16}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
DROPOUT="${DROPOUT:-0.1}"
OUT_DIR="${OUT_DIR:-runs/gen_loop}"
CKPT_DIR="${CKPT_DIR:-checkpoints/gen_loop}"
SEED_BASE="${SEED_BASE:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
INIT_CKPT="${INIT_CKPT:-}"
BUFFER="${BUFFER:-1}"
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
  "$PYTHON" selfplay.py --board-size "$BOARD_SIZE" --games "$GAMES" \
    --rounds-per-move "$ROUNDS" --net-type token --pos-mode "$POS_MODE" \
    --gab-gen-size "$GAB_GEN_SIZE" --gab-intermediate-dim "$GAB_GEN_SIZE" \
    --dropout "$DROPOUT" --dirichlet-epsilon 0.25 --temperature 1.0 \
    --seed "$SEED" $CKPT_FLAG \
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
    --gab-intermediate-dim "$GAB_GEN_SIZE" --dropout "$DROPOUT" \
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
