"""Tests for GeometricAttentionBias (engine/token_transformer.py) --
stage 3 of docs/board-specification.md §6's positional-info rollout.

Mirrors tests/test_relative_position_bias.py's shape of coverage:
standalone module correctness first (including the property that
DEFINES it as "dynamic" -- different board content gives different
bias, unlike RelativePositionBias), then integration through
TokenTransformerNet('gab' and 'gab_absolute'), then a ZeroAgent smoke
test. Also re-checks the eval-mode-after-training NaN regression from
stage 2, since GAB injects a float attn_mask the same way.
"""
import torch
import torch.nn.functional as F
import pytest

from engine.goboard import GameState
from engine.mcts import StyleKnobs, ZeroAgent
from engine.token_encoder import BoardSpec, TokenEncoder
from engine.token_transformer import GeometricAttentionBias, TokenTransformerNet

BOARD_SIZES = (9, 13, 19)


class TestGeometricAttentionBiasShape:
    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_forward_shape(self, n):
        d_model, nheads = 16, 4
        gab = GeometricAttentionBias(n, d_model, nheads, gen_size=8, intermediate_dim=16)
        h = torch.randn(3, n * n, d_model)
        bias = gab(h)
        assert bias.shape == (3, nheads, n * n, n * n)

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_gab_weight_dominates_param_count_as_expected(self, n):
        # gab_weight: num_tokens^2 * gen_size -- the O(N^4) cost flagged
        # in the class docstring and docs/board-specification.md §6.
        d_model, nheads, gen_size = 16, 4, 8
        gab = GeometricAttentionBias(n, d_model, nheads, gen_size=gen_size, intermediate_dim=16)
        assert gab.gab_weight.numel() == (n * n) ** 2 * gen_size
        total = sum(p.numel() for p in gab.parameters())
        assert gab.gab_weight.numel() / total > 0.9  # generator is small by comparison


class TestGeometricAttentionBiasContent:
    def test_different_board_content_gives_different_bias(self):
        # THE defining property that makes this "dynamic": unlike
        # RelativePositionBias (pure function of geometry, identical
        # every forward pass), GAB must vary with what's actually on
        # the board.
        n = 9
        gab = GeometricAttentionBias(n, d_model=16, nheads=2, gen_size=8, intermediate_dim=16)
        torch.manual_seed(0)
        h_a = torch.randn(1, n * n, 16)
        h_b = torch.randn(1, n * n, 16)

        bias_a = gab(h_a)
        bias_b = gab(h_b)
        assert not torch.allclose(bias_a, bias_b)

    def test_same_content_gives_same_bias(self):
        # Deterministic given fixed weights/input (no dropout in GAB).
        n = 9
        gab = GeometricAttentionBias(n, d_model=16, nheads=2, gen_size=8, intermediate_dim=16)
        gab.eval()
        h = torch.randn(1, n * n, 16)
        with torch.no_grad():
            bias_1 = gab(h)
            bias_2 = gab(h)
        assert torch.allclose(bias_1, bias_2)

    def test_batch_examples_get_independent_bias(self):
        # Each example in a batch has its own board -> its own bias,
        # unlike RelativePositionBias which is identical across a batch.
        n = 9
        gab = GeometricAttentionBias(n, d_model=16, nheads=2, gen_size=8, intermediate_dim=16)
        gab.eval()
        torch.manual_seed(1)
        h = torch.randn(2, n * n, 16)
        with torch.no_grad():
            bias = gab(h)
        assert not torch.allclose(bias[0], bias[1])


class TestTokenTransformerNetGabModes:
    def tiny_net(self, n, pos_mode, **kwargs):
        spec = BoardSpec(board_size=n, history_depth=1)
        model = TokenTransformerNet(
            spec, d_model=16, nhead=2, num_layers=2, dim_feedforward=32,
            pos_mode=pos_mode, gab_gen_size=8, gab_intermediate_dim=16, **kwargs,
        )
        return spec, model

    @pytest.mark.parametrize('pos_mode', ('gab', 'gab_absolute'))
    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_forward_shapes(self, pos_mode, n):
        spec, model = self.tiny_net(n, pos_mode)
        x = torch.randn(3, spec.num_tokens, spec.token_dim)
        policy_logits, value = model(x)
        assert policy_logits.shape == (3, spec.num_moves)
        assert value.shape == (3,)

    @pytest.mark.parametrize('pos_mode', ('gab', 'gab_absolute'))
    def test_gradients_flow_including_gab(self, pos_mode):
        spec, model = self.tiny_net(9, pos_mode)
        x = torch.randn(2, spec.num_tokens, spec.token_dim)
        policy_logits, value = model(x)
        (policy_logits.sum() + value.sum()).backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f'{name} got no gradient'
            assert torch.isfinite(p.grad).all(), f'{name} got a non-finite gradient'
        assert model.gab.gab_weight.grad is not None
        if pos_mode == 'gab_absolute':
            assert model.pos_embed.row_embed.weight.grad is not None

    def test_gab_absolute_builds_both_submodules_gab_alone_does_not(self):
        _, both = self.tiny_net(9, 'gab_absolute')
        assert hasattr(both, 'pos_embed')
        assert hasattr(both, 'gab')

        _, gab_only = self.tiny_net(9, 'gab')
        assert not hasattr(gab_only, 'pos_embed')
        assert hasattr(gab_only, 'gab')

    def test_works_with_batch_size_one(self):
        spec, model = self.tiny_net(9, 'gab')
        x = torch.randn(1, spec.num_tokens, spec.token_dim)
        policy_logits, value = model(x)
        assert policy_logits.shape == (1, spec.num_moves)

    @pytest.mark.parametrize('pos_mode', ('gab', 'gab_absolute'))
    def test_eval_mode_after_training_step_produces_no_nan(self, pos_mode):
        # Same fastpath regression as stage 2 -- GAB injects a float
        # attn_mask the same way RelativePositionBias does.
        spec, model = self.tiny_net(9, pos_mode)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.randn(1, spec.num_tokens, spec.token_dim)
        policy_target = torch.full((1, spec.num_moves), 1.0 / spec.num_moves)
        value_target = torch.tensor([0.5])

        model.train()
        for _ in range(3):
            policy_logits, value = model(x)
            log_probs = F.log_softmax(policy_logits, dim=1)
            loss = (-(policy_target * log_probs).sum(dim=1).mean()
                    + F.mse_loss(value, value_target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            policy_logits, value = model(x)
        assert torch.isfinite(policy_logits).all()
        assert torch.isfinite(value).all()


class TestZeroAgentIntegrationGabModes:
    @pytest.mark.parametrize('pos_mode', ('gab', 'gab_absolute'))
    def test_select_move_runs_on_tiny_board(self, pos_mode):
        board_size = 5
        spec = BoardSpec(board_size=board_size, history_depth=1)
        model = TokenTransformerNet(
            spec, d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
            pos_mode=pos_mode, gab_gen_size=8, gab_intermediate_dim=16,
        )
        encoder = TokenEncoder(spec)
        agent = ZeroAgent(model, encoder, StyleKnobs(rounds_per_move=10), seed=0)

        game = GameState.new_game(board_size)
        move, diag = agent.select_move(game)

        assert move is not None
        assert -1.0 <= diag['root_value'] <= 1.0
