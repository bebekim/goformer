#!/usr/bin/env python3
"""Watch one MCTS move decision happen, round by round.

Plays a short fixed opening on 9x9, then instruments ZeroAgent so every
step of the search loop in engine/mcts.py is printed as it happens:
branch selection (PUCT), leaf expansion (network eval), value backup,
and the final style-knob decision among near-equal moves.

    .venv/bin/python trace_move.py

The network here is randomly initialized -- we're watching the search
machinery, not the quality of its judgment.
"""
import math

import numpy as np
import torch

from engine import GameState, StyleKnobs, ZeroAgent, ZeroEncoder, GoZeroNet
from engine.goboard import Move
from engine.gotypes import Point

ROUNDS = 30
BOARD_SIZE = 13
# Fighter-style knobs so the style term is visible in the final table.
KNOBS = StyleKnobs(rounds_per_move=ROUNDS, complexity_weight=0.6,
                   safety_lambda=0.08)


class TracedAgent(ZeroAgent):
    """ZeroAgent with print hooks in the two places the search loop
    makes decisions (mcts.py select_branch / create_node).
    Behavior is unchanged -- it only observes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._round = 0
        self.root = None

    @staticmethod
    def _depth(node):
        d = 0
        while node.parent is not None:
            node = node.parent
            d += 1
        return d

    def create_node(self, game_state, move=None, parent=None, is_root=False):
        node = super().create_node(game_state, move=move, parent=parent,
                                   is_root=is_root)
        if is_root:
            self.root = node
            print(f'[setup] root position evaluated by net: '
                  f'value={node.value:+.3f} for {game_state.next_player.name}')
        else:
            self._round += 1
            print(f'      expand {move}: net values this leaf '
                  f'{node.value:+.3f}; backed up to parent as '
                  f'{-node.value:+.3f}')
        return node

    def select_branch(self, node):
        chosen = super().select_branch(node)
        depth = self._depth(node)
        if depth == 0:
            print(f'round {self._round + 1}:')
        if chosen is not None:
            b = node.branches[chosen]
            explore = (self.knobs.c * b.prior *
                       math.sqrt(node.total_visit_count) / (b.visit_count + 1))
            print(f'  {"  " * depth}pick {chosen}  '
                  f'Q={b.mean_value:+.3f} prior={b.prior:.3f} '
                  f'n={b.visit_count} explore=+{explore:.3f}')
        return chosen


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Fixed 4-move opening (a small corner fight), then trace black's 5th.
    game = GameState.new_game(BOARD_SIZE)
    print('opening:')
    for r, c in [(4, 4), (3, 3), (4, 3), (3, 4)]:
        mv = Move.play(Point(r, c))
        game = game.apply_move(mv)
        print(f'  {game.next_player.other.name:5s} plays {mv}')
    print(f'\nposition: {game.next_player.name} to move. '
          f'Tracing {ROUNDS} search rounds...\n')

    encoder = ZeroEncoder(BOARD_SIZE)
    model = GoZeroNet(BOARD_SIZE)  # random init
    agent = TracedAgent(model, encoder, KNOBS, seed=0)
    move, diag = agent.select_move(game)

    # ---- final root statistics, mirroring choose_from_root's view ----
    root = agent.root
    visited = [m for m in root.moves() if root.visit_count(m) > 0]
    best_q = max(root.expected_value(m) for m in visited)

    def complexity_of(m):
        if root.has_child(m):
            child = root.get_child(m)
            max_entropy = math.log(max(2, len(child.branches)))
            return child.policy_entropy() / max_entropy if max_entropy else 0.0
        return 0.0

    print('\n=== root branch table (top 10 by visits) ===')
    print(f'{"move":>10} {"prior":>6} {"n":>3} {"Q":>7} {"var":>5} '
          f'{"cplx":>5} {"viable":>7} {"style_score":>11}')
    ranked = sorted(visited, key=lambda m: -root.visit_count(m))[:10]
    for m in ranked:
        b = root.branches[m]
        cplx = complexity_of(m)
        viable = root.expected_value(m) >= best_q - KNOBS.equal_value_tolerance
        style = (b.mean_value + KNOBS.complexity_weight * cplx +
                 KNOBS.safety_lambda * b.variance)
        print(f'{str(m):>10} {b.prior:6.3f} {b.visit_count:3d} '
              f'{b.mean_value:+7.3f} {b.variance:5.3f} {cplx:5.3f} '
              f'{"yes" if viable else "":>7} {style:+11.3f}')

    print('\n=== decision ===')
    print('diagnostics from select_move:', diag)
    print('chosen move:', move)


if __name__ == '__main__':
    main()
