"""Tests for train.py's early-stopping / best-checkpoint tracking --
added by docs/board-specification.md §8b's finding that every earlier
comparison in that doc read a fixed epoch instead of each config's own
best epoch. Calls train.main(argv=[...]) in-process (train.py's main()
takes an optional argv, refactored for exactly this) rather than
shelling out, so these run at normal pytest speed.
"""
import json

import numpy as np
import pytest

import train
from engine.token_encoder import BoardSpec, TokenEncoder
from engine.goboard import GameState, Move
from engine.gotypes import Point


def _write_tiny_experience(path, board_size=5, history_depth=1, n=40, seed=0):
    """A small but real experience file: real TokenEncoder states from
    real (if short, semi-random) games, real-shaped random visit-count/
    reward targets. Enough to exercise train.py's actual data path
    without needing a real self-play run."""
    spec = BoardSpec(board_size=board_size, history_depth=history_depth)
    encoder = TokenEncoder(spec)
    rng = np.random.default_rng(seed)

    states, visit_counts, rewards = [], [], []
    for _ in range(n):
        game = GameState.new_game(board_size)
        # A couple of legal moves so states aren't all identical.
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


class TestBestCheckpointTracking:
    def test_no_val_split_keeps_old_behavior(self, tiny_experience, tmp_path):
        out = tmp_path / 'model.pt'
        train.main([
            '--experience', str(tiny_experience), '--board-size', '5',
            '--net-type', 'token', '--pos-mode', 'gab', '--history-depth', '1',
            '--gab-gen-size', '4',
            '--gab-intermediate-dim', '4', '--d-model', '8', '--nhead', '2',
            '--num-layers', '1', '--dim-feedforward', '16',
            '--epochs', '2', '--out-checkpoint', str(out),
        ])
        assert out.exists()
        assert not (tmp_path / 'model.best.pt').exists()
        meta = json.loads((tmp_path / 'model.pt.meta.json').read_text())
        assert meta['best_epoch'] is None
        assert meta['best_val_loss'] is None

    def test_val_split_tracks_and_writes_best(self, tiny_experience, tmp_path):
        out = tmp_path / 'model.pt'
        train.main([
            '--experience', str(tiny_experience), '--board-size', '5',
            '--net-type', 'token', '--pos-mode', 'gab', '--history-depth', '1',
            '--gab-gen-size', '4',
            '--gab-intermediate-dim', '4', '--d-model', '8', '--nhead', '2',
            '--num-layers', '1', '--dim-feedforward', '16',
            '--epochs', '3', '--val-fraction', '0.25', '--out-checkpoint', str(out),
        ])
        assert (tmp_path / 'model.best.pt').exists()
        meta = json.loads((tmp_path / 'model.pt.meta.json').read_text())
        assert meta['best_epoch'] in (1, 2, 3)
        assert meta['best_val_loss'] is not None

        # --out-checkpoint's bytes must match the best checkpoint's,
        # not necessarily the last epoch trained.
        assert out.read_bytes() == (tmp_path / 'model.best.pt').read_bytes()

    def test_out_checkpoint_matches_the_actual_best_epoch_file(self, tiny_experience, tmp_path):
        # Stronger than the byte-equality check above: confirm the
        # recorded best_epoch really is the epoch checkpoint with the
        # lowest combined val loss, by re-reading the per-epoch log.
        out = tmp_path / 'model.pt'
        train.main([
            '--experience', str(tiny_experience), '--board-size', '5',
            '--net-type', 'token', '--pos-mode', 'gab', '--history-depth', '1',
            '--gab-gen-size', '4',
            '--gab-intermediate-dim', '4', '--d-model', '8', '--nhead', '2',
            '--num-layers', '1', '--dim-feedforward', '16',
            '--epochs', '4', '--val-fraction', '0.25', '--out-checkpoint', str(out),
        ])
        meta = json.loads((tmp_path / 'model.pt.meta.json').read_text())
        losses = meta['final_epoch_losses']
        combined = [(r['epoch'], r['val_policy_loss'] + r['val_value_loss']) for r in losses]
        expected_best_epoch = min(combined, key=lambda pair: pair[1])[0]
        assert meta['best_epoch'] == expected_best_epoch


class TestEarlyStoppingPatience:
    def test_requires_val_fraction(self, tiny_experience, tmp_path):
        out = tmp_path / 'model.pt'
        with pytest.raises(SystemExit):
            train.main([
                '--experience', str(tiny_experience), '--board-size', '5',
                '--net-type', 'token', '--pos-mode', 'gab',
                '--early-stopping-patience', '2', '--out-checkpoint', str(out),
            ])

    def test_stops_before_requested_epochs_when_patience_exceeded(self, tiny_experience, tmp_path):
        out = tmp_path / 'model.pt'
        # A tiny, undertrained net on 4 examples of noise-target data
        # will plateau almost immediately -- patience=1 should trigger
        # well before a generous --epochs budget completes.
        train.main([
            '--experience', str(tiny_experience), '--board-size', '5',
            '--net-type', 'token', '--pos-mode', 'gab', '--history-depth', '1',
            '--gab-gen-size', '4',
            '--gab-intermediate-dim', '4', '--d-model', '8', '--nhead', '2',
            '--num-layers', '1', '--dim-feedforward', '16',
            '--epochs', '50', '--val-fraction', '0.25',
            '--early-stopping-patience', '1', '--out-checkpoint', str(out),
        ])
        epoch_checkpoints = list(tmp_path.glob('model.epoch*.pt'))
        assert len(epoch_checkpoints) < 50

    def test_patience_zero_disables_early_stopping(self, tiny_experience, tmp_path):
        out = tmp_path / 'model.pt'
        train.main([
            '--experience', str(tiny_experience), '--board-size', '5',
            '--net-type', 'token', '--pos-mode', 'gab', '--history-depth', '1',
            '--gab-gen-size', '4',
            '--gab-intermediate-dim', '4', '--d-model', '8', '--nhead', '2',
            '--num-layers', '1', '--dim-feedforward', '16',
            '--epochs', '3', '--val-fraction', '0.25',
            '--early-stopping-patience', '0', '--out-checkpoint', str(out),
        ])
        # patience=0 (default): always run the full --epochs, even
        # though best-checkpoint tracking is still active.
        epoch_checkpoints = list(tmp_path.glob('model.epoch*.pt'))
        assert len(epoch_checkpoints) == 3
        assert (tmp_path / 'model.best.pt').exists()
