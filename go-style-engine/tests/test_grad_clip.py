"""Tests for train.py's --grad-clip flag -- added after §8g found
gab_absolute's policy loss diverging catastrophically (3.7 -> 12+
within a few epochs) on real kifu data, and no gradient clipping
existed anywhere in the training loop to guard against it.
"""
from unittest.mock import patch

import numpy as np
import pytest

import train
from engine.token_encoder import BoardSpec, TokenEncoder
from engine.goboard import GameState, Move
from engine.gotypes import Point


def _write_tiny_experience(path, board_size=5, history_depth=1, n=20, seed=0):
    spec = BoardSpec(board_size=board_size, history_depth=history_depth)
    encoder = TokenEncoder(spec)
    rng = np.random.default_rng(seed)

    states, visit_counts, rewards = [], [], []
    for _ in range(n):
        game = GameState.new_game(board_size)
        for p in (Point(1, 1), Point(2, 2)):
            move = Move.play(p)
            if game.is_valid_move(move):
                game = game.apply_move(move)
        states.append(encoder.encode(game))
        vc = rng.integers(0, 5, size=spec.num_moves).astype(np.float32)
        visit_counts.append(vc)
        rewards.append(float(rng.choice([-1.0, 1.0])))

    np.savez_compressed(
        path,
        states=np.stack(states),
        visit_counts=np.stack(visit_counts),
        rewards=np.array(rewards, dtype=np.float32),
    )


@pytest.fixture
def tiny_experience(tmp_path):
    path = tmp_path / 'exp.npz'
    _write_tiny_experience(path)
    return path


def test_grad_clip_disabled_by_default(tiny_experience, tmp_path):
    out = tmp_path / 'model.pt'
    with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
        train.main([
            '--experience', str(tiny_experience), '--board-size', '5',
            '--net-type', 'token', '--history-depth', '1',
            '--out-checkpoint', str(out), '--epochs', '1',
        ])
    mock_clip.assert_not_called()


def test_grad_clip_applied_when_set(tiny_experience, tmp_path):
    out = tmp_path / 'model.pt'
    with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
        train.main([
            '--experience', str(tiny_experience), '--board-size', '5',
            '--net-type', 'token', '--history-depth', '1',
            '--out-checkpoint', str(out), '--epochs', '1',
            '--grad-clip', '1.0',
        ])
    assert mock_clip.called
    _, kwargs = mock_clip.call_args
    called_max_norm = mock_clip.call_args[0][1] if len(mock_clip.call_args[0]) > 1 else kwargs.get('max_norm')
    assert called_max_norm == 1.0


def test_grad_clip_recorded_in_meta(tiny_experience, tmp_path):
    out = tmp_path / 'model.pt'
    train.main([
        '--experience', str(tiny_experience), '--board-size', '5',
        '--net-type', 'token', '--history-depth', '1',
        '--out-checkpoint', str(out), '--epochs', '1',
        '--grad-clip', '2.5',
    ])
    import json
    meta = json.load(open(f'{out}.meta.json'))
    assert meta['args']['grad_clip'] == 2.5
