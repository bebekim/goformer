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
# Simplification, stated plainly: this does NOT accumulate a growing
# replay buffer across generations the way real AlphaZero training does.
# Each generation trains from scratch on only its own fresh self-play
# data, warm-started from the previous generation's WEIGHTS (not its
# data). This is the simplest version that still tests the core
# hypothesis -- accumulating a buffer is a reasonable next step if this
# shows a real trend worth investing more in.
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

NUM_GENERATIONS="${1:-5}"
BOARD_SIZE="${BOARD_SIZE:-9}"
GAMES="${GAMES:-30}"
ROUNDS="${ROUNDS:-50}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-5}"
POS_MODE="${POS_MODE:-gab_absolute}"
GAB_GEN_SIZE="${GAB_GEN_SIZE:-16}"
DROPOUT="${DROPOUT:-0.1}"
OUT_DIR="${OUT_DIR:-runs/gen_loop}"
CKPT_DIR="${CKPT_DIR:-checkpoints/gen_loop}"
SEED_BASE="${SEED_BASE:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
INIT_CKPT="${INIT_CKPT:-}"

mkdir -p "$OUT_DIR" "$CKPT_DIR"
SUMMARY="$OUT_DIR/summary.jsonl"

echo "Starting $NUM_GENERATIONS-generation loop:"
echo "  board_size=$BOARD_SIZE games=$GAMES rounds=$ROUNDS epochs=$EPOCHS patience=$PATIENCE"
echo "  pos_mode=$POS_MODE gab_gen_size=$GAB_GEN_SIZE dropout=$DROPOUT"
echo "  out_dir=$OUT_DIR ckpt_dir=$CKPT_DIR"
echo "  summary=$SUMMARY"
echo

PREV_CKPT="$INIT_CKPT"
for gen in $(seq 1 "$NUM_GENERATIONS"); do
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

  echo "=== Generation $gen/$NUM_GENERATIONS: train ==="
  INCKPT_FLAG=""
  if [ -n "$PREV_CKPT" ]; then
    INCKPT_FLAG="--in-checkpoint $PREV_CKPT"
  fi
  "$PYTHON" train.py --experience "$EXP_FILE" --board-size "$BOARD_SIZE" \
    --net-type token --pos-mode "$POS_MODE" --gab-gen-size "$GAB_GEN_SIZE" \
    --gab-intermediate-dim "$GAB_GEN_SIZE" --dropout "$DROPOUT" \
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
