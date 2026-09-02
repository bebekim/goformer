#!/usr/bin/env python3
"""Fine-tune a checkpoint on an experience buffer produced by
`selfplay.py --save-experience`. Policy target is the (normalized)
MCTS visit-count distribution; value target is the game's final
outcome from the mover's perspective -- the standard AlphaZero loss.

    python train.py --experience runs/exp.npz --board-size 13 \
        --in-checkpoint checkpoints/base.pt \
        --out-checkpoint checkpoints/fighter.pt --epochs 3

Training is reproducible when --seed is set: both torch and numpy RNGs
are seeded before the epoch loop. Checkpoints are saved after every
epoch (<out-stem>.epochN.pt) as well as at the final --out-checkpoint,
and a .meta.json sidecar records args, seed, paths, and final losses.

NOTE: selfplay.py writes per-game shards under <out>/experience/ and
can combine them into a final .npz. train.py accepts either a single
.npz (from --save-experience) or a directory of shards; when given a
directory it loads all game_*.npz shards and concatenates them."""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from engine import GoZeroNet, BoardSpec, TokenTransformerNet


def _git_sha():
    try:
        proc = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _load_experience(experience_path):
    """Load (states, visit_counts, rewards) from a .npz file or a
    directory of game_*.npz shards (combined via concatenation)."""
    path = experience_path
    if os.path.isdir(path):
        shards = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if f.startswith('game_') and f.endswith('.npz')
        ])
        all_states, all_visits, all_rewards = [], [], []
        for s in shards:
            data = np.load(s)
            all_states.append(data['states'])
            all_visits.append(data['visit_counts'])
            all_rewards.append(data['rewards'])
        states = np.concatenate(all_states, axis=0)
        visit_counts = np.concatenate(all_visits, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        return states, visit_counts, rewards
    else:
        data = np.load(path)
        return data['states'], data['visit_counts'], data['rewards']


def build_model(args):
    """Same net_type/pos_mode split as selfplay.py's
    build_encoder_and_model -- see docs/board-specification.md."""
    if args.net_type == 'cnn':
        return GoZeroNet(args.board_size, channels=args.channels, num_blocks=args.blocks)
    spec = BoardSpec(board_size=args.board_size, history_depth=args.history_depth)
    return TokenTransformerNet(
        spec, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        pos_mode=args.pos_mode)


def evaluate(model, states, policy_targets, rewards, batch_size, device):
    """Mean policy/value loss over a fixed set, no grad, no shuffling --
    used for the held-out split so runs are exactly comparable."""
    model.eval()
    total_policy_loss = total_value_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            batch_states = states[start:start + batch_size].to(device)
            batch_policy = policy_targets[start:start + batch_size].to(device)
            batch_value = rewards[start:start + batch_size].to(device)

            policy_logits, value_pred = model(batch_states)
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(batch_policy * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred, batch_value)

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
    model.train()
    return total_policy_loss / num_batches, total_value_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experience', type=str, required=True)
    parser.add_argument('--board-size', type=int, default=13)
    parser.add_argument('--net-type', choices=('cnn', 'token'), default='cnn',
                         help="must match the net-type the experience was "
                              "generated with -- state tensor shapes differ")
    parser.add_argument('--channels', type=int, default=64,
                         help='net-type=cnn only')
    parser.add_argument('--blocks', type=int, default=6,
                         help='net-type=cnn only')
    parser.add_argument('--pos-mode', choices=('absolute', 'relative', 'both'), default='absolute',
                         help='net-type=token only, see docs/board-specification.md §6')
    parser.add_argument('--history-depth', type=int, default=7,
                         help='net-type=token only')
    parser.add_argument('--d-model', type=int, default=64,
                         help='net-type=token only')
    parser.add_argument('--nhead', type=int, default=4,
                         help='net-type=token only')
    parser.add_argument('--num-layers', type=int, default=4,
                         help='net-type=token only')
    parser.add_argument('--dim-feedforward', type=int, default=128,
                         help='net-type=token only')
    parser.add_argument('--in-checkpoint', type=str, default=None,
                         help='start from this checkpoint; random init if omitted')
    parser.add_argument('--out-checkpoint', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--val-fraction', type=float, default=0.0,
                         help='holdout fraction for a validation split reported '
                              'each epoch alongside training loss (0 = no split, '
                              'trains on everything, matches old behavior)')
    parser.add_argument('--seed', type=int, default=0,
                         help='seed for torch and numpy RNGs (0 = default)')
    args = parser.parse_args()

    # Reproducibility: seed both torch and numpy before training
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    states, visit_counts, rewards = _load_experience(args.experience)

    states = torch.from_numpy(states).float()
    visit_counts = torch.from_numpy(visit_counts).float()
    rewards = torch.from_numpy(rewards).float()

    visit_sums = visit_counts.sum(dim=1, keepdim=True).clamp(min=1.0)
    policy_targets = visit_counts / visit_sums

    # Train/val split, seeded so two runs given the same --seed (e.g.
    # comparing net-type=token --pos-mode=absolute vs --pos-mode=relative
    # on the same --experience file) get the IDENTICAL split -- otherwise
    # a loss difference could just be "different val set", not the
    # architecture change under test.
    num_examples = states.shape[0]
    if args.val_fraction > 0:
        split_perm = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(num_examples, generator=split_perm)
        num_val = int(num_examples * args.val_fraction)
        val_idx, train_idx = perm[:num_val], perm[num_val:]
        val_states, val_policy, val_rewards = (
            states[val_idx], policy_targets[val_idx], rewards[val_idx])
        states, policy_targets, rewards = (
            states[train_idx], policy_targets[train_idx], rewards[train_idx])
    else:
        val_states = val_policy = val_rewards = None

    model = build_model(args)
    if args.in_checkpoint:
        model.load_state_dict(torch.load(args.in_checkpoint, map_location=args.device))
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_examples = states.shape[0]
    print(f'{num_examples} training examples'
          + (f', {val_states.shape[0]} val examples' if val_states is not None else '')
          + f', {sum(p.numel() for p in model.parameters()):,} model params')

    out_path = args.out_checkpoint
    out_stem = os.path.splitext(out_path)[0]

    meta = {
        'args': {
            'experience': args.experience,
            'board_size': args.board_size,
            'net_type': args.net_type,
            'channels': args.channels,
            'blocks': args.blocks,
            'pos_mode': args.pos_mode,
            'history_depth': args.history_depth,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'in_checkpoint': args.in_checkpoint,
            'out_checkpoint': args.out_checkpoint,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'device': args.device,
            'val_fraction': args.val_fraction,
            'seed': args.seed,
        },
        'seed': args.seed,
        'in_checkpoint': args.in_checkpoint,
        'experience_paths': [args.experience],
        'git_sha': _git_sha(),
        'torch_version': torch.__version__,
    }

    model.train()
    epoch_losses = []
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(num_examples)
        total_policy_loss = total_value_loss = 0.0
        num_batches = 0

        for start in range(0, num_examples, args.batch_size):
            idx = perm[start:start + args.batch_size]
            batch_states = states[idx].to(args.device)
            batch_policy = policy_targets[idx].to(args.device)
            batch_value = rewards[idx].to(args.device)

            policy_logits, value_pred = model(batch_states)
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(batch_policy * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred, batch_value)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        avg_policy = total_policy_loss / num_batches
        avg_value = total_value_loss / num_batches
        epoch_record = {'epoch': epoch + 1,
                        'policy_loss': avg_policy,
                        'value_loss': avg_value,
                        'total_loss': avg_policy + avg_value}
        log_line = (f'epoch {epoch + 1}/{args.epochs}: '
                    f'policy_loss={avg_policy:.4f}, value_loss={avg_value:.4f}')

        if val_states is not None:
            val_policy_loss, val_value_loss = evaluate(
                model, val_states, val_policy, val_rewards, args.batch_size, args.device)
            epoch_record['val_policy_loss'] = val_policy_loss
            epoch_record['val_value_loss'] = val_value_loss
            log_line += (f' | val_policy_loss={val_policy_loss:.4f}, '
                         f'val_value_loss={val_value_loss:.4f}')

        epoch_losses.append(epoch_record)
        print(log_line)

        # Save epoch checkpoint (besides the final one)
        epoch_path = f'{out_stem}.epoch{epoch + 1}.pt'
        torch.save(model.state_dict(), epoch_path)
        print(f'Wrote epoch checkpoint: {epoch_path}')

    elapsed = time.time() - t0
    print(f'\nTraining complete in {elapsed:.1f}s')

    # Final checkpoint
    torch.save(model.state_dict(), out_path)
    print(f'Wrote {out_path}')

    # Write .meta.json sidecar
    meta['final_epoch_losses'] = epoch_losses
    meta['timestamp'] = time.time()
    meta_path = f'{out_path}.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Wrote {meta_path}')


if __name__ == '__main__':
    main()