#!/usr/bin/env python3
"""Local simulation of a multi-Sail-box actor/learner self-play loop --
validates the ORCHESTRATION LOGIC (fan out N actors, pull their data to
one coordinator, train once, push the new checkpoint back out, repeat)
using plain local file copies in place of `sail box cp`/`sail_fanout`,
before spending money standing up real Sail boxes.

Each "actor" is a separate local directory + its own selfplay.py
subprocess -- standing in for a separate sailbox with its own
filesystem (no shared storage assumed, matching sail box cp's real,
confirmed local<->remote-only behavior: `sail: only one of SRC/DST may
be remote`). The coordinator step (pull -> train -> push) uses
shutil.copy where a real version would use `sail box cp`; the round
loop itself (dispatch all actors, wait, pull, train once, push) is
exactly what a real multi-box version would do -- swapping the copy
primitive later is a small, mechanical change, not a redesign.

Actors run concurrently (subprocess.Popen, not sequential) to mirror
what N real sailboxes running in parallel would actually look like,
timing-wise -- not just a correctness check.

Usage:
    .venv/bin/python distributed_gen_loop.py --actors 3 --rounds 3 \\
        --init-checkpoint checkpoints/base_kifu9x9.pt \\
        --games-per-actor 10 --rounds-per-move 15 --epochs 60 --patience 15
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable


def _run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def dispatch_actor(actor_dir, checkpoint, args, round_idx, seed):
    """Simulates fanning one self-play job out to a sailbox: copy the
    current checkpoint INTO the actor's own directory first (simulating
    `sail box cp` coordinator->box, so the actor genuinely only reads
    its own local copy, never the coordinator's file directly), then
    launch selfplay.py as a background process against that copy."""
    actor_dir.mkdir(parents=True, exist_ok=True)
    actor_ckpt = actor_dir / 'current.pt'
    shutil.copy(checkpoint, actor_ckpt)

    out_dir = actor_dir / f'round{round_idx}_selfplay'
    cmd = [
        PYTHON, 'selfplay.py', '--board-size', str(args.board_size),
        '--games', str(args.games_per_actor), '--rounds-per-move', str(args.rounds_per_move),
        '--net-type', 'token', '--pos-mode', args.pos_mode,
        '--gab-gen-size', str(args.gab_gen_size),
        '--gab-intermediate-dim', str(args.gab_intermediate_dim),
        '--dropout', str(args.dropout), '--dirichlet-epsilon', '0.25', '--temperature', '1.0',
        '--workers', str(args.workers_per_actor),
        '--checkpoint', str(actor_ckpt), '--seed', str(seed),
        '--out', str(out_dir),
    ]
    print(f"  dispatching actor {actor_dir.name} (seed={seed})")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), out_dir


def pull_actor_shards(out_dir, actor_name, round_idx, coordinator_buffer):
    """Simulates `sail box cp` box->coordinator: copy this actor's
    shards into the shared coordinator buffer, prefixed by actor+round
    so nothing from different actors/rounds ever collides."""
    exp_dir = out_dir / 'experience'
    if not exp_dir.is_dir():
        return 0
    n = 0
    for shard in sorted(exp_dir.glob('game_*.npz')):
        dest = coordinator_buffer / f'game_{actor_name}_r{round_idx}_{shard.name[len("game_"):]}'
        shutil.copy(shard, dest)
        n += 1
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--actors', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--init-checkpoint', type=str, required=True)
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--games-per-actor', type=int, default=10)
    parser.add_argument('--rounds-per-move', type=int, default=15)
    parser.add_argument('--pos-mode', default='gab_absolute')
    parser.add_argument('--gab-gen-size', type=int, default=16)
    parser.add_argument('--gab-intermediate-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--workers-per-actor', type=int, default=1,
                         help='selfplay.py --workers WITHIN each simulated actor -- '
                              'a second, independent layer of parallelism from --actors')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--out-dir', default='runs/dist_sim')
    args = parser.parse_args(argv)

    out_root = Path(args.out_dir)
    coordinator_buffer = out_root / 'coordinator' / 'buffer'
    coordinator_ckpts = out_root / 'coordinator' / 'checkpoints'
    coordinator_buffer.mkdir(parents=True, exist_ok=True)
    coordinator_ckpts.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / 'coordinator' / 'summary.jsonl'

    actor_dirs = [out_root / f'actor_{i}' for i in range(args.actors)]

    current_ckpt = args.init_checkpoint

    for round_idx in range(1, args.rounds + 1):
        print(f'\n=== Round {round_idx}/{args.rounds}: dispatching {args.actors} actors ===')
        procs = []
        for i, actor_dir in enumerate(actor_dirs):
            seed = round_idx * 1000 + i
            proc, out_dir = dispatch_actor(actor_dir, current_ckpt, args, round_idx, seed)
            procs.append((proc, out_dir, actor_dir.name))

        print(f'=== Round {round_idx}/{args.rounds}: awaiting {len(procs)} actors ===')
        for proc, out_dir, name in procs:
            stdout, _ = proc.communicate()
            if proc.returncode != 0:
                print(stdout)
                raise RuntimeError(f'actor {name} failed (exit {proc.returncode})')
            last_line = [l for l in stdout.strip().splitlines() if l.strip()][-1] if stdout.strip() else ''
            print(f'  actor {name} done: {last_line}')

        print(f'=== Round {round_idx}/{args.rounds}: pulling shards to coordinator ===')
        total_pulled = 0
        for proc, out_dir, name in procs:
            n = pull_actor_shards(out_dir, name, round_idx, coordinator_buffer)
            total_pulled += n
        n_buffer = len(list(coordinator_buffer.glob('*.npz')))
        print(f'  pulled {total_pulled} shards this round; buffer now has {n_buffer} total')

        print(f'=== Round {round_idx}/{args.rounds}: training on coordinator buffer ===')
        round_ckpt = coordinator_ckpts / f'round{round_idx}.pt'
        train_cmd = [
            PYTHON, 'train.py', '--experience', str(coordinator_buffer),
            '--board-size', str(args.board_size), '--net-type', 'token',
            '--pos-mode', args.pos_mode, '--gab-gen-size', str(args.gab_gen_size),
            '--gab-intermediate-dim', str(args.gab_intermediate_dim),
            '--dropout', str(args.dropout), '--grad-clip', str(args.grad_clip),
            '--in-checkpoint', current_ckpt,
            '--val-fraction', '0.2', '--epochs', str(args.epochs),
            '--early-stopping-patience', str(args.patience),
            '--seed', str(round_idx), '--out-checkpoint', str(round_ckpt),
        ]
        _run(train_cmd)

        meta = json.load(open(f'{round_ckpt}.meta.json'))
        best = next(r for r in meta['final_epoch_losses'] if r['epoch'] == meta['best_epoch'])
        record = {
            'round': round_idx, 'checkpoint': str(round_ckpt),
            'best_epoch': meta['best_epoch'],
            'val_policy_loss': best.get('val_policy_loss'),
            'val_value_loss': best.get('val_value_loss'),
            'buffer_size': n_buffer,
        }
        with open(summary_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(f'Round {round_idx} summary: {record}')

        # PUSH step (simulating `sail box cp` coordinator->box): each
        # actor's next round starts from this checkpoint, pulled fresh
        # into its own directory at the top of dispatch_actor() above.
        current_ckpt = str(round_ckpt)

    print(f'\nDistributed-simulation loop complete ({args.rounds} rounds, {args.actors} actors).')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
