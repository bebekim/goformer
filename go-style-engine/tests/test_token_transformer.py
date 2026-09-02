"""Tests for engine/token_transformer.py -- see docs/board-specification.md.

Two things under test:
1. Shape/gradient correctness at multiple board sizes (the "readily
   changeable" claim, one level up from test_token_encoder.py).
2. That the net actually satisfies GoZeroNet's predict() contract well
   enough to drop into ZeroAgent (mcts.py) unmodified -- an end-to-end
   integration smoke test, not just isolated shape checks.
"""
import numpy as np
import pytest
import torch

from engine.experience import ZeroExperienceCollector
from engine.goboard import GameState
from engine.gotypes import Player
from engine.mcts import StyleKnobs, ZeroAgent
from engine.token_encoder import BoardSpec, TokenEncoder
from engine.token_transformer import RowColPositionalEmbedding, TokenTransformerNet

BOARD_SIZES = (9, 13, 19)


def tiny_net(board_size, history_depth=1):
    spec = BoardSpec(board_size=board_size, history_depth=history_depth)
    return spec, TokenTransformerNet(
        spec, d_model=16, nhead=2, num_layers=2, dim_feedforward=32,
    )


class TestRowColPositionalEmbedding:
    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_param_count_is_linear_in_board_size(self, n):
        d_model = 16
        pe = RowColPositionalEmbedding(n, d_model)
        num_params = sum(p.numel() for p in pe.parameters())
        assert num_params == 2 * n * d_model  # not n*n*d_model

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_distinct_tokens_get_distinct_bias(self, n):
        d_model = 8
        pe = RowColPositionalEmbedding(n, d_model)
        x = torch.zeros(1, n * n, d_model)
        out = pe(x)
        # token 0 = (row=1,col=1), token 1 = (row=1,col=2): different col
        # embedding -> different output, even from identical input.
        assert not torch.allclose(out[0, 0], out[0, 1])
        # token 0 vs token n = (row=2, col=1): different row embedding.
        assert not torch.allclose(out[0, 0], out[0, n])


class TestTokenTransformerNetShape:
    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_forward_shapes(self, n):
        spec, model = tiny_net(n)
        x = torch.randn(3, spec.num_tokens, spec.token_dim)
        policy_logits, value = model(x)
        assert policy_logits.shape == (3, spec.num_moves)
        assert value.shape == (3,)

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_gradients_flow_to_every_submodule(self, n):
        spec, model = tiny_net(n)
        x = torch.randn(2, spec.num_tokens, spec.token_dim)
        policy_logits, value = model(x)
        loss = policy_logits.sum() + value.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f'{name} got no gradient'
            assert torch.isfinite(p.grad).all(), f'{name} got a non-finite gradient'

    def test_changing_board_size_only_changes_shapes(self):
        # Same d_model/nhead/etc, only board_size differs -- confirms no
        # hidden N literal anywhere in the module.
        _, small = tiny_net(9)
        _, large = tiny_net(19)
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        # policy_head/pass_head/value head/input_proj/transformer body
        # are all board-size-independent; only pos_embed grows (linearly).
        assert large_params > small_params
        d_model = 16
        expected_pe_delta = 2 * (19 - 9) * d_model
        assert (large_params - small_params) == expected_pe_delta


class TestPredictContract:
    @pytest.mark.parametrize('n', (9, 13))
    def test_predict_returns_valid_distribution_and_value(self, n):
        spec, model = tiny_net(n)
        encoder = TokenEncoder(spec)
        game = GameState.new_game(n)
        state_tensor = encoder.encode(game)

        priors, value = model.predict(state_tensor)

        assert priors.shape == (spec.num_moves,)
        assert np.all(priors >= 0)
        assert priors.sum() == pytest.approx(1.0, abs=1e-5)
        assert -1.0 <= value <= 1.0


class TestZeroAgentIntegration:
    """End-to-end: TokenEncoder + TokenTransformerNet plugged into the
    existing ZeroAgent/MCTS harness, unmodified -- proves the predict()
    contract match isn't just superficial."""

    def test_select_move_runs_on_tiny_board(self):
        board_size = 5
        spec = BoardSpec(board_size=board_size, history_depth=1)
        model = TokenTransformerNet(
            spec, d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        )
        encoder = TokenEncoder(spec)
        agent = ZeroAgent(
            model, encoder,
            knobs=StyleKnobs(rounds_per_move=10),
            seed=0,
        )

        game = GameState.new_game(board_size)
        move, diag = agent.select_move(game)

        assert move is not None
        assert 'root_value' in diag
        assert -1.0 <= diag['root_value'] <= 1.0

    def test_select_move_with_collector_attached(self):
        # Regression test: engine/mcts.py's select_move only calls
        # self.encoder.num_moves() when a collector is attached (the
        # path selfplay.py always exercises, to save experience shards).
        # TokenEncoder was missing that method entirely until this test
        # was added -- the test above never caught it because it never
        # attaches a collector.
        board_size = 5
        spec = BoardSpec(board_size=board_size, history_depth=1)
        model = TokenTransformerNet(
            spec, d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        )
        encoder = TokenEncoder(spec)
        agent = ZeroAgent(model, encoder, StyleKnobs(rounds_per_move=10), seed=0)
        collector = ZeroExperienceCollector()
        agent.set_collector(collector)
        collector.begin_episode()

        game = GameState.new_game(board_size)
        move, diag = agent.select_move(game)
        collector.complete_episode(reward=0)

        assert move is not None
        assert len(collector.states) == 1
        assert collector.visit_counts[0].shape == (spec.num_moves,)

    def test_play_a_few_moves(self):
        board_size = 5
        spec = BoardSpec(board_size=board_size, history_depth=1)
        model = TokenTransformerNet(
            spec, d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        )
        encoder = TokenEncoder(spec)
        black = ZeroAgent(model, encoder, StyleKnobs(rounds_per_move=8), seed=1)
        white = ZeroAgent(model, encoder, StyleKnobs(rounds_per_move=8), seed=2)

        game = GameState.new_game(board_size)
        for _ in range(6):
            agent = black if game.next_player == Player.black else white
            move, _ = agent.select_move(game)
            game = game.apply_move(move)
        # Didn't crash; game state is still well-formed.
        assert game.board.num_rows == board_size
