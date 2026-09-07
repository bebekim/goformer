"""Tests for selfplay.py's play_one_game --temperature-cutoff annealing.

Added after the repo owner's own qualitative play-testing found the
resulting checkpoints weak ("18k-level") -- every self-play game uses
temperature=1.0 the whole way through (needed for opening diversity),
which also makes the training data noisy for the entire game, not
just the opening, unlike standard AlphaZero-style training which
anneals to greedy after an opening window.
"""
import dataclasses

from selfplay import play_one_game
from engine import BoardSpec, TokenEncoder, TokenTransformerNet, ZeroAgent, StyleKnobs


def _make_agents(knobs_a, knobs_b, board_size=5):
    spec = BoardSpec(board_size=board_size, history_depth=1)
    encoder = TokenEncoder(spec)
    model = TokenTransformerNet(spec, d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
                                 pos_mode='both')
    model.eval()
    black_agent = ZeroAgent(model, encoder, knobs_a, seed=0)
    white_agent = ZeroAgent(model, encoder, knobs_b, seed=1)
    return black_agent, white_agent


def test_no_cutoff_leaves_knobs_untouched():
    knobs = StyleKnobs(rounds_per_move=4, temperature=1.0, dirichlet_epsilon=0.25)
    black_agent, white_agent = _make_agents(knobs, knobs)
    play_one_game(5, black_agent, white_agent, max_moves=6, temperature_cutoff=None)
    assert black_agent.knobs.temperature == 1.0
    assert white_agent.knobs.dirichlet_epsilon == 0.25


def test_cutoff_switches_to_greedy_after_n_plies():
    knobs = StyleKnobs(rounds_per_move=4, temperature=1.0, dirichlet_epsilon=0.25)
    black_agent, white_agent = _make_agents(knobs, knobs)
    play_one_game(5, black_agent, white_agent, max_moves=6, temperature_cutoff=2)
    assert black_agent.knobs.temperature == 0.0
    assert black_agent.knobs.dirichlet_epsilon == 0.0
    assert white_agent.knobs.temperature == 0.0
    assert white_agent.knobs.dirichlet_epsilon == 0.0


def test_cutoff_does_not_mutate_shared_original_knobs():
    # The real bug this test guards against: build_knobs() constructs
    # ONE StyleKnobs instance reused across every game in a run --
    # mutating agent.knobs in place would leak annealed settings into
    # every subsequent game, not just the one that hit the cutoff.
    shared_knobs = StyleKnobs(rounds_per_move=4, temperature=1.0, dirichlet_epsilon=0.25)
    black_agent, white_agent = _make_agents(shared_knobs, shared_knobs)
    play_one_game(5, black_agent, white_agent, max_moves=6, temperature_cutoff=1)
    assert shared_knobs.temperature == 1.0
    assert shared_knobs.dirichlet_epsilon == 0.25
    # confirm the agents got their OWN independent copies, not the shared object
    assert black_agent.knobs is not shared_knobs
    assert dataclasses.is_dataclass(black_agent.knobs)


def test_cutoff_beyond_game_length_never_triggers():
    knobs = StyleKnobs(rounds_per_move=4, temperature=1.0, dirichlet_epsilon=0.25)
    black_agent, white_agent = _make_agents(knobs, knobs)
    play_one_game(5, black_agent, white_agent, max_moves=3, temperature_cutoff=1000)
    assert black_agent.knobs.temperature == 1.0
