"""Tests for data/kifu_to_experience.py -- the Specs/009 adapter that
turns go-gibo-ingestion's parsed kifu JSONL into a train.py-compatible
experience buffer. Uses tiny synthetic records matching that repo's
real output schema, not a dependency on go-gibo-ingestion being cloned
or fetched at test time.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data'))
import kifu_to_experience as kte  # noqa: E402

import train
from engine import BoardSpec, TokenEncoder
from engine.goboard import Move


def _record(game_id, category, winner, moves):
    return {
        'game_id': game_id,
        'category': category,
        'board_size': 9,
        'komi': 5.5,
        'date': '2000-01-01',
        'event': 'test event',
        'players': {'black': 'A', 'black_rank': '5d',
                    'white': 'B', 'white_rank': '6d'},
        'result': {'winner': winner, 'method': 'resign', 'margin': None},
        'moves': moves,
        'has_variations': False,
    }


SHORT_GAME = _record('9x9/NHK/g1.sgf', 'NHK', 'B', [
    {'color': 'B', 'pass': False, 'col': 2, 'row': 2},
    {'color': 'W', 'pass': False, 'col': 3, 'row': 3},
    {'color': 'B', 'pass': True},
    {'color': 'W', 'pass': False, 'col': 4, 'row': 4},
])

MINIGO_GAME = _record('9x9/Minigo/g2.sgf', 'Minigo', 'W', [
    {'color': 'B', 'pass': False, 'col': 0, 'row': 0},
    {'color': 'W', 'pass': False, 'col': 1, 'row': 1},
])

ILLEGAL_REPLAY_GAME = _record('9x9/NHK/bad.sgf', 'NHK', 'B', [
    {'color': 'B', 'pass': False, 'col': 2, 'row': 2},
    # W plays on the exact same point B just occupied -- Board.place_stone
    # asserts on this rather than silently ignoring it.
    {'color': 'W', 'pass': False, 'col': 2, 'row': 2},
])


def test_sgf_move_to_engine_move_pass():
    move = kte._sgf_move_to_engine_move({'pass': True})
    assert move.is_pass


def test_sgf_move_to_engine_move_coordinate_mapping():
    # 0-indexed (col=2, row=3) -> 1-indexed Point(row=4, col=3)
    move = kte._sgf_move_to_engine_move({'pass': False, 'col': 2, 'row': 3})
    assert move.is_play
    assert move.point.row == 4
    assert move.point.col == 3


def test_convert_game_shapes_and_targets():
    spec = BoardSpec(board_size=9, history_depth=1)
    encoder = TokenEncoder(spec)
    states, visit_counts, rewards = kte.convert_game(SHORT_GAME, encoder)

    assert len(states) == len(visit_counts) == len(rewards) == 4
    assert states[0].shape == (spec.num_tokens, spec.token_dim)
    for vc in visit_counts:
        assert vc.shape == (spec.num_moves,)
        assert vc.sum() == pytest.approx(1.0)  # one-hot

    # SHORT_GAME's result.winner is 'B': black moves get +1, white -1.
    assert rewards == [1.0, -1.0, 1.0, -1.0]


def test_convert_game_pass_move_target_is_pass_index():
    spec = BoardSpec(board_size=9, history_depth=1)
    encoder = TokenEncoder(spec)
    _, visit_counts, _ = kte.convert_game(SHORT_GAME, encoder)
    pass_index = encoder.encode_move(Move.pass_turn())
    assert visit_counts[2][pass_index] == 1.0


def test_convert_game_illegal_replay_raises():
    spec = BoardSpec(board_size=9, history_depth=1)
    encoder = TokenEncoder(spec)
    with pytest.raises(ValueError, match='illegal_move_replay'):
        kte.convert_game(ILLEGAL_REPLAY_GAME, encoder)


def test_convert_jsonl_default_is_human_only(tmp_path):
    jsonl_path = tmp_path / 'games.jsonl'
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps(SHORT_GAME) + '\n')
        f.write(json.dumps(MINIGO_GAME) + '\n')

    out_dir = tmp_path / 'exp'
    summary = kte.convert_jsonl(jsonl_path, out_dir, history_depth=1)

    assert summary['games_kept'] == 1  # only the NHK (human) game
    shards = sorted(out_dir.glob('game_*.npz'))
    assert len(shards) == 1


def test_convert_jsonl_categories_override_includes_minigo(tmp_path):
    jsonl_path = tmp_path / 'games.jsonl'
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps(SHORT_GAME) + '\n')
        f.write(json.dumps(MINIGO_GAME) + '\n')

    out_dir = tmp_path / 'exp'
    summary = kte.convert_jsonl(
        jsonl_path, out_dir, categories=('NHK', 'Minigo'), history_depth=1)

    assert summary['games_kept'] == 2


def test_convert_jsonl_skips_and_counts_illegal_replay(tmp_path):
    jsonl_path = tmp_path / 'games.jsonl'
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps(SHORT_GAME) + '\n')
        f.write(json.dumps(ILLEGAL_REPLAY_GAME) + '\n')

    out_dir = tmp_path / 'exp'
    summary = kte.convert_jsonl(
        jsonl_path, out_dir, categories=('NHK',), history_depth=1)

    assert summary['games_kept'] == 1
    assert summary['skipped'] == {'illegal_move_replay': ['9x9/NHK/bad.sgf']}


def test_output_shards_are_train_compatible(tmp_path):
    """End-to-end: convert -> train.py runs one epoch without crashing,
    net-type=token, unmodified train.py, per Specs/009's acceptance
    criteria."""
    jsonl_path = tmp_path / 'games.jsonl'
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps(SHORT_GAME) + '\n')

    out_dir = tmp_path / 'exp'
    kte.convert_jsonl(jsonl_path, out_dir, categories=('NHK',), history_depth=1)

    ckpt = tmp_path / 'model.pt'
    train.main([
        '--experience', str(out_dir),
        '--board-size', '9',
        '--net-type', 'token',
        '--history-depth', '1',
        '--out-checkpoint', str(ckpt),
        '--epochs', '1',
    ])
    assert ckpt.exists()
