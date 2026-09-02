"""
Tests for crash-safe, resumable self-play and reproducible training.

Uses tiny fast configs (board-size 5 or 9, rounds-per-move 10-20,
max-moves ~10, games 3) so the suite stays under ~90s. Never loads a
checkpoint (random init, but seeded).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# The selfplay/train scripts live one level up from this tests/ dir.
RESEARCH_DIR = Path(__file__).resolve().parent.parent
SELFPALY_SCRIPT = RESEARCH_DIR / 'selfplay.py'
TRAIN_SCRIPT = RESEARCH_DIR / 'train.py'


def _run_selfplay(args, env=None):
    """Run selfplay.py as a subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(SELFPALY_SCRIPT)] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                          env=env)
    return proc.returncode, proc.stdout, proc.stderr


def _run_train(args, env=None):
    """Run train.py as a subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(TRAIN_SCRIPT)] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                          env=env)
    return proc.returncode, proc.stdout, proc.stderr


def _game_record_fingerprint(rec):
    """Return a hashable fingerprint of a game record for equality checks,
    EXCLUDING think_time_s (which is timing-dependent)."""
    out = {
        'game_index': rec['game_index'],
        'board_size': rec['board_size'],
        'black_preset': rec['black_preset'],
        'white_preset': rec['white_preset'],
        'result': rec['result'],
        'moves': [],
    }
    for m in rec['moves']:
        out['moves'].append({
            'move_index': m['move_index'],
            'player': m['player'],
            'move': m['move'],
            'win_prob_black': m['win_prob_black'],
            'complexity': m['complexity'],
            'variance': m['variance'],
            'viable_count': m['viable_count'],
            'candidate_count': m['candidate_count'],
            # think_time_s deliberately excluded
        })
    return out


def _load_games_jsonl(path):
    """Read games.jsonl and return list of game records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _games_equal_except_timing(a, b):
    """Compare two game records ignoring think_time_s fields."""
    return _game_record_fingerprint(a) == _game_record_fingerprint(b)


# ---------------------------------------------------------------------------
# test_resume_reproduces_uninterrupted_run
# ---------------------------------------------------------------------------

class TestResumeReproducesUninterruptedRun:
    """Verify that a resumable run (games=1 then games=3) reproduces
    an uninterrupted run (games=3) bit-for-bit (except timing)."""

    @pytest.fixture
    def tmp_path(self):
        d = tempfile.mkdtemp(prefix='test_resume_')
        yield Path(d)
        shutil.rmtree(d, ignore_errors=True)

    def test_resume_reproduces_uninterrupted_run(self, tmp_path):
        # Configuration tuned for speed: small board, few rounds, few moves.
        base_args = [
            '--board-size', '5',
            '--black-preset', 'baseline',
            '--white-preset', 'baseline',
            '--rounds-per-move', '10',
            '--max-moves', '10',
            '--seed', '42',
        ]

        # Run A: full uninterrupted 3-game run
        out_a = tmp_path / 'run_a'
        rc_a, out_a_stdout, err_a = _run_selfplay(
            base_args + ['--games', '3', '--out', str(out_a)])
        assert rc_a == 0, f'selfplay A failed:\n{err_a}'
        games_a = _load_games_jsonl(out_a / 'games.jsonl')
        assert len(games_a) == 3, f'Expected 3 games in A, got {len(games_a)}'

        # Run B: first a 1-game run (partial), then resume with 3 games
        out_b = tmp_path / 'run_b'
        rc_b1, _, err_b1 = _run_selfplay(
            base_args + ['--games', '1', '--out', str(out_b)])
        assert rc_b1 == 0, f'selfplay B part 1 failed:\n{err_b1}'
        assert (out_b / 'games.jsonl').exists()

        rc_b2, _, err_b2 = _run_selfplay(
            base_args + ['--games', '3', '--out', str(out_b)])
        assert rc_b2 == 0, f'selfplay B part 2 (resume) failed:\n{err_b2}'

        games_b = _load_games_jsonl(out_b / 'games.jsonl')
        assert len(games_b) == 3, f'Expected 3 games in B after resume, got {len(games_b)}'

        # Assert all 3 games in B equal A for all fields except think_time_s
        for i, (ga, gb) in enumerate(zip(games_a, games_b)):
            assert ga['game_index'] == gb['game_index'], \
                f'game_index mismatch at index {i}'
            assert _games_equal_except_timing(ga, gb), \
                f'Game {i} differs between A and B (excluding timing)'


# ---------------------------------------------------------------------------
# test_train_reproducible
# ---------------------------------------------------------------------------

class TestTrainReproducible:
    """Verify that training with the same --seed produces byte-identical
    state_dict files."""

    @pytest.fixture
    def tmp_path(self):
        d = tempfile.mkdtemp(prefix='test_train_')
        yield Path(d)
        shutil.rmtree(d, ignore_errors=True)

    def test_train_reproducible(self, tmp_path):
        # Step 1: generate a tiny experience .npz via a 1-game selfplay run
        # (shards are written; the final combined .npz is what train.py consumes)
        selfplay_out = tmp_path / 'sp_out'
        rc_sp, _, err_sp = _run_selfplay([
            '--board-size', '5',
            '--black-preset', 'baseline',
            '--white-preset', 'baseline',
            '--rounds-per-move', '10',
            '--max-moves', '10',
            '--seed', '42',
            '--games', '1',
            '--out', str(selfplay_out),
            '--save-experience', str(tmp_path / 'exp.npz'),
        ])
        assert rc_sp == 0, f'selfplay for training test failed:\n{err_sp}'
        exp_path = tmp_path / 'exp.npz'
        assert exp_path.exists(), 'Expected combined experience .npz'

        # Step 2: run train.py twice with the same --seed to different outputs
        ckpt1 = tmp_path / 'ckpt1.pt'
        ckpt2 = tmp_path / 'ckpt2.pt'
        rc1, _, err1 = _run_train([
            '--experience', str(exp_path),
            '--board-size', '5',
            '--epochs', '2',
            '--batch-size', '16',
            '--seed', '42',
            '--out-checkpoint', str(ckpt1),
        ])
        assert rc1 == 0, f'train run 1 failed:\n{err1}'

        rc2, _, err2 = _run_train([
            '--experience', str(exp_path),
            '--board-size', '5',
            '--epochs', '2',
            '--batch-size', '16',
            '--seed', '42',
            '--out-checkpoint', str(ckpt2),
        ])
        assert rc2 == 0, f'train run 2 failed:\n{err2}'

        # Step 3: assert the two state_dict files are byte-identical
        assert ckpt1.exists() and ckpt2.exists()
        import torch
        sd1 = torch.load(ckpt1, map_location='cpu')
        sd2 = torch.load(ckpt2, map_location='cpu')
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), \
                f'Training reproducibility failed: param {k} differs'


# ---------------------------------------------------------------------------
# test_manifest_and_sidecars_exist
# ---------------------------------------------------------------------------

class TestManifestAndSidecarsExist:
    """Verify that manifest.json, games.jsonl, experience shards, and
    train .meta.json are written with the required keys."""

    @pytest.fixture
    def tmp_path(self):
        d = tempfile.mkdtemp(prefix='test_sidecar_')
        yield Path(d)
        shutil.rmtree(d, ignore_errors=True)

    def test_selfplay_manifest_and_sidecars(self, tmp_path):
        out_dir = tmp_path / 'run'
        rc, _, err = _run_selfplay([
            '--board-size', '5',
            '--black-preset', 'baseline',
            '--white-preset', 'baseline',
            '--rounds-per-move', '10',
            '--max-moves', '10',
            '--seed', '7',
            '--games', '2',
            '--out', str(out_dir),
            '--save-experience', str(tmp_path / 'combined.npz'),
        ])
        assert rc == 0, f'selfplay failed:\n{err}'

        # manifest.json
        manifest_path = out_dir / 'manifest.json'
        assert manifest_path.exists(), 'manifest.json missing'
        manifest = json.loads(manifest_path.read_text())
        required_manifest_keys = [
            'args', 'seed', 'git_sha', 'torch_version',
            'start_timestamp', 'board_size', 'presets',
            'black_knobs', 'white_knobs',
        ]
        for k in required_manifest_keys:
            assert k in manifest, f'manifest.json missing key: {k}'
        assert 'games' in manifest['args']
        assert manifest['args']['board_size'] == 5

        # games.jsonl
        games_path = out_dir / 'games.jsonl'
        assert games_path.exists(), 'games.jsonl missing'
        games = _load_games_jsonl(games_path)
        assert len(games) == 2, f'Expected 2 games, got {len(games)}'
        for g in games:
            for k in ['game_index', 'board_size', 'black_preset',
                      'white_preset', 'moves', 'result']:
                assert k in g, f'games.jsonl record missing key: {k}'

        # experience shards
        exp_dir = out_dir / 'experience'
        assert exp_dir.exists(), 'experience/ directory missing'
        shard_files = sorted(exp_dir.glob('game_*.npz'))
        assert len(shard_files) == 2, f'Expected 2 shards, got {len(shard_files)}'
        # Verify each shard is loadable
        for sf in shard_files:
            data = np.load(sf)
            for arr_name in ['states', 'visit_counts', 'rewards']:
                assert arr_name in data, f'shard {sf.name} missing {arr_name}'

        # summary.json
        summary_path = out_dir / 'summary.json'
        assert summary_path.exists(), 'summary.json missing'
        summary = json.loads(summary_path.read_text())
        for k in ['board_size', 'black_preset', 'white_preset',
                  'black_knobs', 'white_knobs', 'seed', 'games']:
            assert k in summary, f'summary.json missing key: {k}'

        # combined .npz from --save-experience
        combined_path = tmp_path / 'combined.npz'
        assert combined_path.exists(), 'combined experience .npz missing'
        data = np.load(combined_path)
        for arr_name in ['states', 'visit_counts', 'rewards']:
            assert arr_name in data, f'combined.npz missing {arr_name}'

    def test_train_meta_json(self, tmp_path):
        # Generate a tiny experience buffer first
        sp_out = tmp_path / 'sp'
        rc_sp, _, err_sp = _run_selfplay([
            '--board-size', '5',
            '--black-preset', 'baseline',
            '--white-preset', 'baseline',
            '--rounds-per-move', '10',
            '--max-moves', '10',
            '--seed', '42',
            '--games', '1',
            '--out', str(sp_out),
            '--save-experience', str(tmp_path / 'exp.npz'),
        ])
        assert rc_sp == 0, f'selfplay failed:\n{err_sp}'

        ckpt_path = tmp_path / 'model.pt'
        rc_tr, _, err_tr = _run_train([
            '--experience', str(tmp_path / 'exp.npz'),
            '--board-size', '5',
            '--epochs', '1',
            '--batch-size', '16',
            '--seed', '42',
            '--out-checkpoint', str(ckpt_path),
        ])
        assert rc_tr == 0, f'train failed:\n{err_tr}'

        # .meta.json sidecar
        meta_path = Path(str(ckpt_path) + '.meta.json')
        assert meta_path.exists(), '.meta.json missing'
        meta = json.loads(meta_path.read_text())
        required_meta_keys = [
            'args', 'seed', 'in_checkpoint', 'experience_paths',
            'git_sha', 'torch_version', 'final_epoch_losses', 'timestamp',
        ]
        for k in required_meta_keys:
            assert k in meta, f'.meta.json missing key: {k}'
        assert isinstance(meta['final_epoch_losses'], list)
        assert len(meta['final_epoch_losses']) == 1
        assert 'policy_loss' in meta['final_epoch_losses'][0]
        assert 'value_loss' in meta['final_epoch_losses'][0]

        # epoch checkpoint files
        epoch1 = tmp_path / 'model.epoch1.pt'
        assert epoch1.exists(), 'epoch1 checkpoint missing'
