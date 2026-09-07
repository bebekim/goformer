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

Per-game MOVE SELECTION is deterministic given (args.seed, game_idx,
knobs, model) when dirichlet_epsilon=temperature=0 (no exploration
randomness) -- but this does NOT mean two separate process runs of the
identical command reproduce byte-identical games. Checked directly
while adding --workers: two plain sequential (--workers 1) runs of the
same command, same seed, same checkpoint, zero exploration knobs,
produced different games. Forcing single-threaded BLAS
(OMP_NUM_THREADS=1) didn't fix it either. The remaining source is the
model's own forward-pass floating-point output varying slightly
between separate process launches -- a known PyTorch CPU limitation
(non-deterministic reduction order in some ops without
torch.use_deterministic_algorithms), not a bug in this file, and not
something --workers introduces (it was already true before --workers
existed). Games are still valid/legal either way; just don't rely on
byte-for-byte reruns for anything."""
import argparse
import copy
import dataclasses
import json
import multiprocessing
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from engine import (GameState, Player, StyleKnobs, ZeroAgent, ZeroEncoder, GoZeroNet,
                    BoardSpec, TokenEncoder, TokenTransformerNet)
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


def build_encoder_and_model(args):
    """net_type='cnn' (default) is the original ZeroEncoder/GoZeroNet
    path. net_type='token' builds the trans-go-former body instead
    (engine/token_encoder.py, engine/token_transformer.py) -- see
    docs/board-specification.md. Both sides honor the same
    encode/decode_move_index/predict contract, so nothing else in this
    file (play_one_game, ZeroAgent) needs to know which one it got."""
    if args.net_type == 'cnn':
        encoder = ZeroEncoder(args.board_size)
        model = GoZeroNet(args.board_size, channels=args.channels, num_blocks=args.blocks)
    else:
        spec = BoardSpec(board_size=args.board_size, history_depth=args.history_depth)
        encoder = TokenEncoder(spec)
        model = TokenTransformerNet(
            spec, d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
            dropout=args.dropout, pos_mode=args.pos_mode,
            gab_gen_size=args.gab_gen_size, gab_intermediate_dim=args.gab_intermediate_dim)
    return encoder, model


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
                   black_collector=None, white_collector=None,
                   temperature_cutoff=None):
    """temperature_cutoff: standard AlphaZero-style annealing -- after
    this many plies, both agents switch to temperature=0/
    dirichlet_epsilon=0 (greedy, no exploration noise) for the rest of
    the game. None (default) means never anneal -- temperature/
    dirichlet stay whatever the agents' own knobs say, for the whole
    game, the original behavior. Without this, EVERY move of EVERY
    self-play game (not just the opening) samples among near-tied
    options at temperature=1.0 -- the setting needed for opening
    diversity, but which also makes the training data noisy for the
    entire game, including the midgame/endgame where the network may
    already have a clear, correct best move. Reassigns each agent's
    OWN .knobs (dataclasses.replace, not mutation) rather than the
    shared StyleKnobs instance callers construct once and reuse across
    every game in a run -- mutating that in place would leak the
    annealed settings into every subsequent game too."""
    game = GameState.new_game(board_size)
    agents = {Player.black: black_agent, Player.white: white_agent}
    telemetry = []
    move_index = 0
    cutoff_applied = False

    if black_collector is not None:
        black_agent.set_collector(black_collector)
        black_collector.begin_episode()
    if white_collector is not None:
        white_agent.set_collector(white_collector)
        white_collector.begin_episode()

    while not game.is_over() and move_index < max_moves:
        if (temperature_cutoff is not None and not cutoff_applied
                and move_index >= temperature_cutoff):
            for a in (black_agent, white_agent):
                a.knobs = dataclasses.replace(a.knobs, temperature=0.0, dirichlet_epsilon=0.0)
            cutoff_applied = True

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


def _play_and_package_game(model, encoder, black_knobs, white_knobs, args, game_idx):
    """Play one game and package everything needed to write it to disk
    (game record + combined-color experience arrays), without touching
    any shared file handles -- the one piece of per-game work that's
    identical whether it runs inline (--workers 1, the original
    behavior) or inside a worker process (--workers > 1, new)."""
    black_agent = ZeroAgent(model, encoder, black_knobs, device=args.device,
                            seed=args.seed * 1000 + game_idx)
    white_agent = ZeroAgent(model, encoder, white_knobs, device=args.device,
                            seed=args.seed * 1000 + game_idx + 500)

    black_collector = ZeroExperienceCollector()
    white_collector = ZeroExperienceCollector()

    max_moves = args.max_moves or (2 * args.board_size * args.board_size)
    telemetry, result = play_one_game(
        args.board_size, black_agent, white_agent, max_moves,
        black_collector=black_collector, white_collector=white_collector,
        temperature_cutoff=args.temperature_cutoff)

    game_record = {
        'game_index': game_idx,
        'board_size': args.board_size,
        'black_preset': args.black_preset,
        'white_preset': args.white_preset,
        'moves': telemetry,
        'result': result,
    }
    states, visit_counts, rewards = combine_experience(
        [black_collector, white_collector]).to_arrays()
    return game_idx, game_record, states, visit_counts, rewards


# Populated once per worker PROCESS by _init_worker (multiprocessing's
# 'spawn' start method re-imports this module fresh in each worker, so
# a plain module-level dict is safe -- no cross-process sharing).
_worker_ctx = {}


def _init_worker(args, black_knobs, white_knobs):
    # Each spawned worker is a fresh process/interpreter -- main()'s own
    # top-level seeding doesn't propagate to it. Only matters when
    # --checkpoint is omitted (random-init self-play, still used for the
    # very first pre-kifu-seeded generation): without this, every worker
    # would construct a DIFFERENTLY random-initialized model, breaking
    # this module's own documented "reproducible from (seed, game_idx,
    # knobs, model) alone" guarantee. With --checkpoint given (the usual
    # case), this is a no-op -- the loaded weights are identical either way.
    # PyTorch defaults each process's intra-op thread pool to one thread
    # per (perceived) core -- fine for a single process, but with a Pool
    # of N worker processes each independently spinning up that many
    # threads, N processes end up oversubscribing the machine N-fold
    # (measured: 8 workers x 15 threads = 120 threads contending for 8
    # physical cores, load average ~40 on an 8-core box). Each worker
    # here only ever runs one game at a time, so it needs exactly one
    # thread for its own inference -- the parallelism already comes from
    # the process pool, not from each process's internal thread pool.
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder, model = build_encoder_and_model(args)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device)
    model.eval()  # inference, not training -- dropout must not be live here.
                  # Was missing everywhere in this file (train.py and
                  # play_server.py both already call this); found via a
                  # reproducibility check when adding --workers. Every
                  # self-play run with --dropout > 0 (the standard setting,
                  # 0.1, throughout this session's refinement loops) had
                  # genuinely stochastic forward passes from live dropout,
                  # an uncontrolled noise source on top of the intended
                  # dirichlet/temperature exploration, and in violation of
                  # this module's own documented reproducibility guarantee.
    _worker_ctx['model'] = model
    _worker_ctx['encoder'] = encoder
    _worker_ctx['black_knobs'] = black_knobs
    _worker_ctx['white_knobs'] = white_knobs
    _worker_ctx['args'] = args


def _play_game_worker(game_idx):
    return _play_and_package_game(
        _worker_ctx['model'], _worker_ctx['encoder'],
        _worker_ctx['black_knobs'], _worker_ctx['white_knobs'],
        _worker_ctx['args'], game_idx)


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
            'temperature_cutoff': args.temperature_cutoff,
            'checkpoint': args.checkpoint,
            'net_type': args.net_type,
            'channels': args.channels,
            'blocks': args.blocks,
            'pos_mode': args.pos_mode,
            'history_depth': args.history_depth,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'gab_gen_size': args.gab_gen_size,
            'gab_intermediate_dim': args.gab_intermediate_dim,
            'dropout': args.dropout,
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
        ('net_type', args.net_type, m.get('net_type')),
        ('pos_mode', args.pos_mode, m.get('pos_mode')),
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
    parser.add_argument('--temperature-cutoff', type=int, default=None,
                         help='standard AlphaZero-style annealing: switch to greedy '
                              '(temperature=0, dirichlet_epsilon=0) after this many '
                              'plies, for the rest of the game. Default: disabled -- '
                              'temperature/dirichlet stay constant the whole game '
                              '(the original behavior). Without this, every self-play '
                              'game is noisy for its full length, not just the opening, '
                              'even in the midgame/endgame where the network may '
                              'already have a clear best move -- a real, plausible '
                              'contributor to weak resulting play, not just data volume.')
    parser.add_argument('--checkpoint', type=str, default=None,
                         help='PyTorch state_dict to load for both agents; random init if omitted')
    parser.add_argument('--net-type', choices=('cnn', 'token'), default='cnn',
                         help="'cnn' = ZeroEncoder/GoZeroNet (default); "
                              "'token' = TokenEncoder/TokenTransformerNet "
                              "(docs/board-specification.md)")
    parser.add_argument('--channels', type=int, default=64,
                         help='net-type=cnn only')
    parser.add_argument('--blocks', type=int, default=6,
                         help='net-type=cnn only')
    parser.add_argument('--pos-mode',
                         choices=('absolute', 'relative', 'both', 'gab', 'gab_absolute'),
                         default='absolute',
                         help='net-type=token only: TokenTransformerNet positional-info '
                              'mode, see docs/board-specification.md §6-7')
    parser.add_argument('--history-depth', type=int, default=7,
                         help='net-type=token only: BoardSpec.history_depth')
    parser.add_argument('--d-model', type=int, default=64,
                         help='net-type=token only')
    parser.add_argument('--nhead', type=int, default=4,
                         help='net-type=token only')
    parser.add_argument('--num-layers', type=int, default=4,
                         help='net-type=token only')
    parser.add_argument('--dim-feedforward', type=int, default=128,
                         help='net-type=token only')
    parser.add_argument('--gab-gen-size', type=int, default=64,
                         help="net-type=token, pos-mode=gab*/gab_absolute only: "
                              "GeometricAttentionBias's template-library size "
                              "(64 matches maia3-3m/5m's cheap configs)")
    parser.add_argument('--gab-intermediate-dim', type=int, default=64,
                         help='net-type=token, pos-mode=gab*/gab_absolute only')
    parser.add_argument('--dropout', type=float, default=0.0,
                         help='net-type=token only: nn.TransformerEncoderLayer dropout. '
                              'Was a constructor arg on TokenTransformerNet but never '
                              'wired to a flag until docs/board-specification.md §8 found '
                              'GAB overfitting the policy head at small self-play scale.')
    parser.add_argument('--max-moves', type=int, default=None,
                         help='default: 2 * board_size^2')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--workers', type=int, default=1,
                         help='play this many games in parallel (separate '
                              'processes, each with its own model copy). '
                              '1 (default) = original sequential behavior, '
                              'unchanged. Self-play games are independent '
                              'and CPU-bound with no batching in engine/mcts.py '
                              '(one board evaluated at a time), so this is '
                              'the lever that actually uses multiple cores -- '
                              'a bigger box only helps once this is set > 1.')
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

    encoder, model = build_encoder_and_model(args)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device)
    model.eval()  # inference, not training -- dropout must not be live here.
                  # Was missing everywhere in this file (train.py and
                  # play_server.py both already call this); found via a
                  # reproducibility check when adding --workers. Every
                  # self-play run with --dropout > 0 (the standard setting,
                  # 0.1, throughout this session's refinement loops) had
                  # genuinely stochastic forward passes from live dropout,
                  # an uncontrolled noise source on top of the intended
                  # dirichlet/temperature exploration, and in violation of
                  # this module's own documented reproducibility guarantee.
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

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

    def _record_completed_game(game_idx, game_record, states, visit_counts, rewards):
        nonlocal black_wins, decided
        games_file.write(json.dumps(game_record) + '\n')
        games_file.flush()
        np.savez_compressed(exp_dir / f'game_{game_idx:04d}.npz',
                            states=states, visit_counts=visit_counts, rewards=rewards)

        result = game_record['result']
        if result['winner'] is not None:
            decided += 1
            if result['winner'] == 'black':
                black_wins += 1

        telemetry = game_record['moves']
        avg_complexity = np.mean([m['complexity'] for m in telemetry]) if telemetry else 0.0
        avg_think = np.mean([m['think_time_s'] for m in telemetry]) if telemetry else 0.0
        print(f'game {game_idx + 1}/{args.games}: {len(telemetry)} moves, '
              f'result={result.get("winner")}, avg_complexity={avg_complexity:.3f}, '
              f'avg_think_s={avg_think:.3f}')

    remaining = [i for i in range(args.games) if i not in completed]

    if args.workers > 1 and remaining:
        print(f'running {len(remaining)} games across {args.workers} worker processes')
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(args, black_knobs, white_knobs)) as pool:
            for game_idx, game_record, states, visit_counts, rewards in \
                    pool.imap_unordered(_play_game_worker, remaining):
                _record_completed_game(game_idx, game_record, states, visit_counts, rewards)
    else:
        for game_idx in remaining:
            _, game_record, states, visit_counts, rewards = _play_and_package_game(
                model, encoder, black_knobs, white_knobs, args, game_idx)
            _record_completed_game(game_idx, game_record, states, visit_counts, rewards)

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