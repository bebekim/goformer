"""AlphaZero-style MCTS agent, ported from dlgo/zero/agent.py to
PyTorch, with the coaching "style knob" layer added on top of the
original PUCT search:

  rounds_per_move   -- pace: how many playouts before committing to a move
  c                 -- candidate breadth: PUCT exploration constant.
                        Low c concentrates visits on the top branch
                        (narrow/deep reading); high c spreads visits
                        across more candidates (broad reading).
  complexity_weight -- among moves whose value is within
                        equal_value_tolerance of the best (i.e.
                        genuinely near-equal, not just noise), prefer
                        higher-complexity (positive) or lower-
                        complexity (negative) continuations. Complexity
                        is the entropy of the resulting position's own
                        policy distribution -- a flatter/more open
                        policy at the child means more live options,
                        which is the cheapest complexity signal already
                        sitting in the search tree (no extra net calls).
  safety_lambda     -- risk tolerance: among the same near-equal set,
                        prefer lower-variance (negative lambda, safe)
                        or higher-variance (positive lambda, risk-
                        seeking) branches, using the variance of the
                        backed-up values MCTS already collected for
                        that branch.
  temperature       -- 0 picks the top-scoring viable branch
                        deterministically; >0 samples from a softmax
                        over viable-branch scores (controllable
                        unpredictability, e.g. for a rattled trainee).
  dirichlet_epsilon -- standard AlphaZero root exploration noise.

Voice/Policy (stable per-trainee identity) is meant to live in the
*checkpoint* (fine-tuned weights, see train.py); Psychology (contextual
state -- leading/trailing/fatigue/captaincy) is meant to modulate these
knobs move-to-move or game-to-game. This module only implements the
mechanism; which trainee gets which numbers is a content decision made
by the caller (see selfplay.py).
"""
import math
import time
from dataclasses import dataclass

import numpy as np

from .goboard import Move


@dataclass
class StyleKnobs:
    rounds_per_move: int = 200
    c: float = 2.0
    complexity_weight: float = 0.0
    safety_lambda: float = 0.0
    equal_value_tolerance: float = 0.05
    temperature: float = 0.0
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.0


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.total_value_sq = 0.0

    @property
    def mean_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def variance(self):
        if self.visit_count == 0:
            return 0.0
        mean = self.mean_value
        return max(0.0, self.total_value_sq / self.visit_count - mean * mean)


class ZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {
            move: Branch(p)
            for move, p in priors.items()
            if state.is_valid_move(move)
        }
        self.children = {}

    def moves(self):
        return self.branches.keys()

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]

    def record_visit(self, move, value):
        self.total_visit_count += 1
        branch = self.branches[move]
        branch.visit_count += 1
        branch.total_value += value
        branch.total_value_sq += value * value

    def expected_value(self, move):
        return self.branches[move].mean_value

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0

    def policy_entropy(self):
        """Entropy of this node's own prior distribution -- used by a
        parent branch as a cheap 'how open is this position' proxy."""
        priors = np.array([b.prior for b in self.branches.values()])
        priors = priors[priors > 0]
        if len(priors) == 0:
            return 0.0
        return float(-(priors * np.log(priors)).sum())


class ZeroAgent:
    def __init__(self, model, encoder, knobs=None, device='cpu', seed=None):
        self.model = model
        self.encoder = encoder
        self.knobs = knobs or StyleKnobs()
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.collector = None

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        start = time.perf_counter()
        root = self.create_node(game_state, is_root=True)

        for _ in range(self.knobs.rounds_per_move):
            node = root
            next_move = self.select_branch(node)
            while next_move is not None and node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)

            if next_move is None:
                # No legal continuation from this node (shouldn't normally
                # happen since pass is always legal, but guard anyway).
                continue

            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state, move=next_move, parent=node)

            move = next_move
            value = -1 * child_node.value
            backup_node = node
            while backup_node is not None:
                backup_node.record_visit(move, value)
                move = backup_node.last_move
                backup_node = backup_node.parent
                value = -1 * value

        elapsed = time.perf_counter() - start

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx))
                for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(root_state_tensor, visit_counts)

        chosen_move, diagnostics = self.choose_from_root(root)
        diagnostics['think_time_s'] = elapsed
        diagnostics['root_value'] = root.value
        diagnostics['rounds'] = self.knobs.rounds_per_move
        return chosen_move, diagnostics

    def create_node(self, game_state, move=None, parent=None, is_root=False):
        state_tensor = self.encoder.encode(game_state)
        priors, value = self.model.predict(state_tensor, device=self.device)

        if is_root and self.knobs.dirichlet_epsilon > 0:
            noise = self.rng.dirichlet(
                [self.knobs.dirichlet_alpha] * len(priors))
            priors = (1 - self.knobs.dirichlet_epsilon) * priors + \
                self.knobs.dirichlet_epsilon * noise

        move_priors = {
            self.encoder.decode_move_index(idx): p
            for idx, p in enumerate(priors)
        }
        new_node = ZeroTreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def select_branch(self, node):
        if not node.branches:
            return None
        total_n = node.total_visit_count

        def score_branch(move):
            branch = node.branches[move]
            q = branch.mean_value
            p = branch.prior
            n = branch.visit_count
            return q + self.knobs.c * p * math.sqrt(total_n) / (n + 1)

        return max(node.branches.keys(), key=score_branch)

    def choose_from_root(self, root):
        """Style-knob move selection: rank the genuinely-viable
        candidates (near the best backed-up value, not just the raw
        argmax) by a complexity/safety score, then pick deterministically
        or sample by temperature."""
        visited = [m for m in root.moves() if root.visit_count(m) > 0]
        if not visited:
            return Move.pass_turn(), {'viable_count': 0, 'complexity': 0.0, 'variance': 0.0}

        best_value = max(root.expected_value(m) for m in visited)
        tol = self.knobs.equal_value_tolerance
        viable = [m for m in visited if root.expected_value(m) >= best_value - tol]

        def complexity_of(move):
            if root.has_child(move):
                child = root.get_child(move)
                max_entropy = math.log(max(2, len(child.branches)))
                return child.policy_entropy() / max_entropy if max_entropy > 0 else 0.0
            return 0.0

        scored = []
        for m in viable:
            branch = root.branches[m]
            complexity = complexity_of(m)
            score = (
                branch.mean_value
                + self.knobs.complexity_weight * complexity
                + self.knobs.safety_lambda * branch.variance
            )
            scored.append((m, score, complexity, branch.variance))

        if self.knobs.temperature > 0 and len(scored) > 1:
            scores = np.array([s[1] for s in scored])
            scores = scores / self.knobs.temperature
            scores -= scores.max()
            probs = np.exp(scores)
            probs /= probs.sum()
            idx = self.rng.choice(len(scored), p=probs)
        else:
            idx = int(np.argmax([s[1] for s in scored]))

        chosen_move, _, chosen_complexity, chosen_variance = scored[idx]
        diagnostics = {
            'viable_count': len(viable),
            'candidate_count': len(visited),
            'complexity': chosen_complexity,
            'variance': chosen_variance,
            'value': root.expected_value(chosen_move),
        }
        return chosen_move, diagnostics
