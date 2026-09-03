#!/usr/bin/env python3
"""Convert go-gibo-ingestion's parsed kifu JSONL into a train.py-
compatible experience buffer -- per Specs/009. Reads one JSON record
per game (schema: see ~/repositories/individual/go-gibo-ingestion's
README.md), replays each game's moves through engine.GameState, and
writes one game_*.npz shard per game in the exact (states,
visit_counts, rewards) format engine/experience.py/selfplay.py already
produce, so `train.py --experience <output dir>` works unmodified.

Two real differences from selfplay.py's own experience buffers, both
intentional (see Specs/009's Edge Cases):
  - visit_counts here is a ONE-HOT vector at the move actually played,
    not a soft MCTS visit distribution -- there is no search over a
    human/recorded game, only the single move chosen.
  - reward is the game's real final outcome (+1/-1 from the mover's
    color, per the archive's own recorded result), not a value derived
    from this engine's own play.

Coordinate mapping: go-gibo-ingestion's SGF-derived moves are
0-indexed (col, row); engine.Point is 1-indexed (row, col). This
module is the seam between the two conventions -- see
_sgf_move_to_engine_move.

Default game selection is HUMAN-ONLY (Specs/009 Decision Log #2):
most of the CWI 9x9 archive (419/517 games, category "Minigo") is AI
self-play, not human kifu, and mixing that back in would defeat the
point of testing "real human signal vs. this project's own weak
random-net self-play." Override with --categories to include others
explicitly (e.g. for a deliberately different, future experiment).

Usage:
    python3 data/kifu_to_experience.py \\
        --input ../go-gibo-ingestion/data/parsed/9x9.jsonl \\
        --output runs/kifu_exp
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Runnable directly (`python3 data/kifu_to_experience.py`, per this
# module's own Usage) as well as as a module (`python -m
# data.kifu_to_experience`) or imported from tests -- direct
# invocation needs go-style-engine's root (one level up) on sys.path
# for `import engine` to resolve, since this script's own directory
# (data/) is what Python puts there by default.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import BoardSpec, GameState, Move, Point, TokenEncoder

HUMAN_CATEGORIES = ('NHK', 'ProPairgo', 'Misc', 'Go_Seigen')

DEFAULT_INPUT = str(
    Path(__file__).resolve().parents[2] / 'go-gibo-ingestion' / 'data' / 'parsed' / '9x9.jsonl')


def _sgf_move_to_engine_move(m):
    if m.get('pass'):
        return Move.pass_turn()
    return Move.play(Point(row=m['row'] + 1, col=m['col'] + 1))


def convert_game(record, encoder):
    """Replay one game record through `encoder`, returning
    (states, visit_counts, rewards) lists -- one entry per move, state
    encoded BEFORE that move is applied, matching how
    engine/mcts.py's ZeroAgent records decisions during self-play.
    Raises ValueError (with a short reason) if the game can't be
    replayed -- a real, if rare, possibility for archival data (a
    move landing on an already-occupied point, which this engine's
    Board.place_stone asserts against rather than silently ignoring)."""
    winner = record['result']['winner']
    num_moves_total = encoder.num_moves()
    game = GameState.new_game(record['board_size'])

    states, visit_counts, rewards = [], [], []
    for m in record['moves']:
        move = _sgf_move_to_engine_move(m)
        states.append(encoder.encode(game))
        target = np.zeros(num_moves_total, dtype=np.float32)
        target[encoder.encode_move(move)] = 1.0
        visit_counts.append(target)
        rewards.append(1.0 if m['color'] == winner else -1.0)
        try:
            game = game.apply_move(move)
        except AssertionError:
            raise ValueError('illegal_move_replay')

    return states, visit_counts, rewards


def convert_jsonl(input_path, output_dir, categories=HUMAN_CATEGORIES,
                   board_size=9, history_depth=7):
    encoder = TokenEncoder(BoardSpec(board_size=board_size, history_depth=history_depth))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    total_positions = 0
    skipped = {}

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if categories is not None and record['category'] not in categories:
                continue

            try:
                states, visit_counts, rewards = convert_game(record, encoder)
            except ValueError as e:
                skipped.setdefault(str(e), []).append(record['game_id'])
                continue
            if not states:
                skipped.setdefault('empty_game', []).append(record['game_id'])
                continue

            np.savez_compressed(
                output_dir / f'game_{shard_idx:05d}.npz',
                states=np.array(states, dtype=np.float32),
                visit_counts=np.array(visit_counts, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
            )
            shard_idx += 1
            total_positions += len(states)

    return {'games_kept': shard_idx, 'positions': total_positions, 'skipped': skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=DEFAULT_INPUT,
                         help='go-gibo-ingestion JSONL path '
                              '(default: ../go-gibo-ingestion/data/parsed/9x9.jsonl)')
    parser.add_argument('--output', default='runs/kifu_exp')
    parser.add_argument('--categories', nargs='+', default=None,
                         help='Game categories to include. Default: human-only '
                              f'({", ".join(HUMAN_CATEGORIES)}). Pass explicit '
                              'categories (e.g. --categories Minigo) to override.')
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--history-depth', type=int, default=7)
    args = parser.parse_args(argv)

    categories = args.categories if args.categories is not None else list(HUMAN_CATEGORIES)
    summary = convert_jsonl(
        args.input, args.output, categories=categories,
        board_size=args.board_size, history_depth=args.history_depth)

    print(f"Converted {summary['games_kept']} games "
          f"({summary['positions']} positions) -> {args.output}")
    total_skipped = sum(len(ids) for ids in summary['skipped'].values())
    if total_skipped:
        print(f'Skipped {total_skipped} games:')
        for reason, ids in summary['skipped'].items():
            print(f'  {reason}: {len(ids)}')

    return summary


if __name__ == '__main__':
    main()
