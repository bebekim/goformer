#!/usr/bin/env python3
"""Self-play harness: run games between two style-knob configurations
on an NxN board (default 13x13) and dump per-move telemetry so the
knob-distinguishability question can actually be checked against data,
before any of this gets ported to Swift/Core ML.

Example -- baseline vs. a "Fighter"-leaning configuration:

    python selfplay.py --board-size 13 --games 20 \
        --black-preset baseline --white-preset fighter \
        --rounds-per-move 200 --out runs/baseline_vs_fighter/

Output is a RUN DIRECTORY (not a single JSON file). Each completed game
is written as one JSON line to <out>/games.jsonl (flushed immediately,
so a laptop shutdown mid-run loses only the in-flight game, not everything).
Experience is saved as one shard per game under <out>/experience/; if
--save-experience is also given, all shards are combined into that final
.npz at the end. Run directories are resumable: if games.jsonl already
exists, already-completed indices are skipped.

The per-game results are reproducible from (args.seed, game_idx, knobs,
model) alone -- see module-level notes for the determinism assumptions
and the one caveat (train.py's torch.randperm)."""
import argparse
import copy
import dataclasses
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from engine import GameState, Player, StyleKnobs, ZeroAgent, ZeroEncoder, GoZeroNet
from engine.experience import ZeroExperienceCollector, combine_experience

# Illustrative starting points for the four PRD trainee archetypes.
# These are placeholders, not calibrated values -- the point of this
# harness is to check whether settings like these actually produce
# separable telemetry, then tune from what the data shows.
PRESETS = {
    'baseline': StyleKnobs(),
    'fighter': StyleKnobs(
        complexity_weight=0.6, safety_lambda=0.08, rounds_per_move=150),
    'builder': StyleKnobs(
        complexity_weight=-0.5, safety_lambda=-0.15, rounds_per_move=260),
    'adapter': StyleKnobs(
        c=3.0, complexity_weight=0.2, rounds_per_move=200),
    'anchor': StyleKnobs(
        safety_lambda=-0.25, complexity_weight=-0.1, rounds_per_move=220),
}


def build_knobs(preset_name, overrides):
    base = copy.deepcopy(PRESETS[preset_name])
    for field in dataclasses.fields(base):
        cli_value = overrides.get(field.name)
        if cli_value is not None:
            setattr(base, field.name, cli_value)
    return base


def _git_sha():
    """Return git rev-parse HEAD, tolerating failure (e.g. not a repo)."""
    try:
        proc = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def play_one_game(board_size, black_agent, white_agent, max_moves,
                   black_collector=None, white_collector=None):
    game = GameState.new_game(board_size)
    agents = {Player.black: black_agent, Player.white: white_agent}
    telemetry = []
    move_index = 0

    if black_collector is not None:
        black_agent.set_collector(black_collector)
        black_collector.begin_episode()
    if white_collector is not None:
        white_agent.set_collector(white_collector)
        white_collector.begin_episode()

    while not game.is_over() and move_index < max_moves:
        mover = game.next_player
        agent = agents[mover]
        move, diag = agent.select_move(game)

        root_value = diag['root_value']  # from mover's perspective
        win_prob_black = (root_value + 1) / 2 if mover == Player.black \
            else (1 - root_value) / 2

        telemetry.append({
            'move_index': move_index,
            'player': 'black' if mover == Player.black else 'white',
            'move': str(move),
            'win_prob_black': win_prob_black,
            'complexity': diag['complexity'],
            'variance': diag['variance'],
            'viable_count': diag['viable_count'],
            'candidate_count': diag['candidate_count'],
            'think_time_s': diag['think_time_s'],
        })

        game = game.apply_move(move)
        move_index += 1

    result = None
    if game.is_over() and game.last_move is not None and not game.last_move.is_resign:
        from engine.scoring import compute_game_result
        gr = compute_game_result(game)
        result = {'winner': 'black' if gr.winner == Player.black else 'white',
                   'black_score': gr.b, 'white_score': gr.w, 'komi': gr.komi,
                   'margin': gr.winning_margin, 'reason': 'score'}
    else:
        result = {'winner': None, 'reason': 'move_cap_reached', 'move_count': move_index}

    if black_collector is not None or white_collector is not None:
        black_reward = 1 if result['winner'] == 'black' else (-1 if result['winner'] == 'white' else 0)
        if black_collector is not None:
            black_collector.complete_episode(black_reward)
        if white_collector is not None:
            white_collector.complete_episode(-black_reward)

    return telemetry, result


def _write_manifest(out_dir, args, black_knobs, white_knobs, sha, torch_version, start_ts):
    manifest = {
        'args': {
            'board_size': args.board_size,
            'games': args.games,
            'black_preset': args.black_preset,
            'white_preset': args.white_preset,
            'rounds_per_move': args.rounds_per_move,
            'c': args.c,
            'complexity_weight': args.complexity_weight,
            'safety_lambda': args.safety_lambda,
            'temperature': args.temperature,
            'dirichlet_epsilon': args.dirichlet_epsilon,
            'checkpoint': args.checkpoint,
            'channels': args.channels,
            'blocks': args.blocks,
            'max_moves': args.max_moves,
            'device': args.device,
            'seed': args.seed,
            'save_experience': args.save_experience,
        },
        'seed': args.seed,
        'git_sha': sha,
        'torch_version': torch_version,
        'start_timestamp': start_ts,
        'board_size': args.board_size,
        'presets': {
            'black': args.black_preset,
            'white': args.white_preset,
        },
        'black_knobs': dataclasses.asdict(black_knobs),
        'white_knobs': dataclasses.asdict(white_knobs),
    }
    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _check_resume_compatibility(manifest, args):
    """Return list of warning strings if args disagree with manifest."""
    warnings = []
    m = manifest['args']
    checks = [
        ('games', args.games, m.get('games')),
        ('board_size', args.board_size, m.get('board_size')),
        ('black_preset', args.black_preset, m.get('black_preset')),
        ('white_preset', args.white_preset, m.get('white_preset')),
        ('seed', args.seed, m.get('seed')),
        ('checkpoint', args.checkpoint, m.get('checkpoint')),
    ]
    for name, current, stored in checks:
        if current != stored:
            warnings.append(
                f'WARNING: manifest says {name}={stored!r}, '
                f'current args say {name}={current!r} -- proceeding with current args')
    return warnings


def _load_completed_indices(out_dir):
    """Read games.jsonl and return set of completed game indices."""
    games_path = out_dir / 'games.jsonl'
    if not games_path.exists():
        return set()
    completed = set()
    with open(games_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completed.add(rec['game_index'])
    return completed


def _load_shard_arrays(out_dir, game_idx):
    """Load (states, visit_counts, rewards) from a single game shard."""
    shard_path = out_dir / 'experience' / f'game_{game_idx:04d}.npz'
    data = np.load(shard_path)
    return data['states'], data['visit_counts'], data['rewards']


def _combine_shards_to_npz(out_dir, shard_indices, dest_path, flush_every=None):
    """Load shards for the given game indices and combine into one .npz.

    Uses the same collector-free array concatenation as the old
    combine_experience path (no need for ZeroExperienceCollector here).
    """
    all_states = []
    all_visit_counts = []
    all_rewards = []
    for idx in shard_indices:
        states, visit_counts, rewards = _load_shard_arrays(out_dir, idx)
        all_states.append(states)
        all_visit_counts.append(visit_counts)
        all_rewards.append(rewards)
    combined_states = np.concatenate(all_states, axis=0)
    combined_visits = np.concatenate(all_visit_counts, axis=0)
    combined_rewards = np.concatenate(all_rewards, axis=0)
    np.savez_compressed(dest_path,
                        states=combined_states,
                        visit_counts=combined_visits,
                        rewards=combined_rewards)
    return combined_states.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--board-size', type=int, default=13)
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--black-preset', choices=PRESETS.keys(), default='baseline')
    parser.add_argument('--white-preset', choices=PRESETS.keys(), default='baseline')
    parser.add_argument('--rounds-per-move', type=int, default=None)
    parser.add_argument('--c', type=float, default=None)
    parser.add_argument('--complexity-weight', type=float, default=None)
    parser.add_argument('--safety-lambda', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--dirichlet-epsilon', type=float, default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                         help='PyTorch state_dict to load for both agents; random init if omitted')
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--max-moves', type=int, default=None,
                         help='default: 2 * board_size^2')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--save-experience', type=str, default=None,
                         help='optional path to save a combined .npz training buffer '
                              '(state/visit-count/reward triples) for train.py; '
                              'shards are always written under <out>/experience/')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    overrides = {
        'rounds_per_move': args.rounds_per_move,
        'c': args.c,
        'complexity_weight': args.complexity_weight,
        'safety_lambda': args.safety_lambda,
        'temperature': args.temperature,
        'dirichlet_epsilon': args.dirichlet_epsilon,
    }
    black_knobs = build_knobs(args.black_preset, overrides)
    white_knobs = build_knobs(args.white_preset, overrides)

    encoder = ZeroEncoder(args.board_size)
    model = GoZeroNet(args.board_size, channels=args.channels, num_blocks=args.blocks)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    max_moves = args.max_moves or (2 * args.board_size * args.board_size)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sha = _git_sha()
    torch_version = torch.__version__
    start_ts = time.time()

    manifest = _write_manifest(out_dir, args, black_knobs, white_knobs,
                               sha, torch_version, start_ts)

    # Resume logic
    completed = _load_completed_indices(out_dir)
    resume_from = len(completed)
    total_games = args.games
    if resume_from > 0:
        # Count lines in manifest to know total target
        print(f'resuming from game {resume_from}/{total_games}')
        warnings = _check_resume_compatibility(manifest, args)
        for w in warnings:
            print(w)
    else:
        print(f'starting fresh run: {total_games} games')

    # Open games.jsonl for appending and keep it open
    games_path = out_dir / 'games.jsonl'
    games_file = open(games_path, 'a')

    # Experience shard directory
    exp_dir = out_dir / 'experience'
    exp_dir.mkdir(parents=True, exist_ok=True)

    black_wins = 0
    decided = 0
    t0 = start_ts

    for game_idx in range(args.games):
        if game_idx in completed:
            # Skip already-completed games on resume
            continue

        black_agent = ZeroAgent(model, encoder, black_knobs, device=args.device,
                                seed=args.seed * 1000 + game_idx)
        white_agent = ZeroAgent(model, encoder, white_knobs, device=args.device,
                                seed=args.seed * 1000 + game_idx + 500)

        black_collector = white_collector = None
        if args.save_experience or True:
            # Always create collectors so we can write per-game shards;
            # the --save-experience flag now controls whether the FINAL
            # combined .npz is written (in addition to shards).
            black_collector = ZeroExperienceCollector()
            white_collector = ZeroExperienceCollector()

        telemetry, result = play_one_game(
            args.board_size, black_agent, white_agent, max_moves,
            black_collector=black_collector, white_collector=white_collector)

        # Write per-game JSON line to games.jsonl
        game_record = {
            'game_index': game_idx,
            'board_size': args.board_size,
            'black_preset': args.black_preset,
            'white_preset': args.white_preset,
            'moves': telemetry,
            'result': result,
        }
        games_file.write(json.dumps(game_record) + '\n')
        games_file.flush()

        # Save experience shard for this game (always, since collectors exist)
        if black_collector is not None:
            black_collector.save(exp_dir / f'game_{game_idx:04d}.npz')

        if result['winner'] is not None:
            decided += 1
            if result['winner'] == 'black':
                black_wins += 1

        avg_complexity = np.mean([m['complexity'] for m in telemetry]) if telemetry else 0.0
        avg_think = np.mean([m['think_time_s'] for m in telemetry]) if telemetry else 0.0
        print(f'game {game_idx + 1}/{args.games}: {len(telemetry)} moves, '
              f'result={result.get("winner")}, avg_complexity={avg_complexity:.3f}, '
              f'avg_think_s={avg_think:.3f}')

    games_file.close()

    elapsed = time.time() - t0
    print(f'\n{args.games} games in {elapsed:.1f}s '
          f'({elapsed / max(1, args.games):.1f}s/game)')
    if decided:
        print(f'Black win rate: {black_wins}/{decided} decided games '
              f'({100 * black_wins / decided:.1f}%)')

    # Write summary.json
    completed_indices = sorted(_load_completed_indices(out_dir))
    summary = {
        'board_size': args.board_size,
        'black_preset': args.black_preset,
        'white_preset': args.white_preset,
        'black_knobs': dataclasses.asdict(black_knobs),
        'white_knobs': dataclasses.asdict(white_knobs),
        'seed': args.seed,
        'games': [{'game_index': idx} for idx in completed_indices],
    }
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {summary_path}')

    # If --save-experience was given, combine all shards into the requested path
    if args.save_experience:
        combined_count = _combine_shards_to_npz(
            out_dir, completed_indices, Path(args.save_experience))
        print(f'Wrote experience buffer: {args.save_experience} '
              f'({combined_count} positions)')


if __name__ == '__main__':
    main()