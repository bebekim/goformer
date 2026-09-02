"""Tests for RelativePositionBias (engine/token_transformer.py) -- stage
2 of docs/board-specification.md §6's positional-info rollout.

Mirrors tests/test_token_transformer.py's shape of coverage for stage
1: standalone module correctness first, then integration through
TokenTransformerNet(pos_mode='relative'), then a ZeroAgent smoke test.
"""
import torch
import pytest

from engine.goboard import GameState
from engine.mcts import StyleKnobs, ZeroAgent
from engine.token_encoder import BoardSpec, TokenEncoder
from engine.token_transformer import RelativePositionBias, TokenTransformerNet

BOARD_SIZES = (9, 13, 19)


class TestRelativePositionBiasShape:
    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_forward_shape(self, n):
        nheads = 4
        bias_module = RelativePositionBias(n, nheads)
        bias = bias_module()
        assert bias.shape == (nheads, n * n, n * n)

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_learned_param_count_is_quadratic_in_board_size_not_quartic(self, n):
        # gate: nheads * (2N-1)^2 -- NOT nheads * N^4, which is what a
        # dense per-pair table would cost (see the module docstring's
        # 71MB/714MB comparison).
        nheads = 8
        bias_module = RelativePositionBias(n, nheads)
        num_learned = sum(p.numel() for p in bias_module.parameters())
        assert num_learned == nheads * (2 * n - 1) ** 2

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_index_buffer_is_not_counted_as_a_learned_parameter(self, n):
        bias_module = RelativePositionBias(n, 4)
        assert 'bin_index' not in dict(bias_module.named_parameters())
        assert bias_module.bin_index.shape == (n * n, n * n)


class TestRelativePositionBiasContent:
    def test_same_offset_gives_same_bias_regardless_of_where_on_board(self):
        # The defining property of a *relative* (not absolute) bias:
        # two pairs of tokens with the same (Δrow, Δcol) get identical
        # bias, even though they're at different absolute positions.
        n = 9
        bias_module = RelativePositionBias(n, nheads=3)
        torch.manual_seed(0)
        bias_module.gate.data = torch.randn_like(bias_module.gate)
        bias = bias_module()  # (H, T, T)

        def idx(r, c):
            return n * (r - 1) + (c - 1)

        # (2,2)->(3,3) is Δ=(+1,+1); (5,5)->(6,6) is also Δ=(+1,+1).
        pair_a = bias[:, idx(2, 2), idx(3, 3)]
        pair_b = bias[:, idx(5, 5), idx(6, 6)]
        assert torch.allclose(pair_a, pair_b)

    def test_different_offsets_are_independently_learnable(self):
        n = 9
        bias_module = RelativePositionBias(n, nheads=1)
        assert bias_module.num_bins == (2 * n - 1) ** 2
        # A neighbor offset and a far offset must land in different
        # bins, so gradient updates to one don't move the other.
        def idx(r, c):
            return n * (r - 1) + (c - 1)
        neighbor_bin = bias_module.bin_index[idx(5, 5), idx(5, 6)].item()
        far_bin = bias_module.bin_index[idx(5, 5), idx(1, 1)].item()
        assert neighbor_bin != far_bin

    def test_self_pair_is_the_zero_offset_bin_for_every_token(self):
        n = 9
        bias_module = RelativePositionBias(n, nheads=1)
        diag = torch.diagonal(bias_module.bin_index)
        assert torch.all(diag == diag[0])  # Δ=(0,0) is one fixed bin


class TestTokenTransformerNetRelativeMode:
    def tiny_net(self, n, **kwargs):
        spec = BoardSpec(board_size=n, history_depth=1)
        model = TokenTransformerNet(
            spec, d_model=16, nhead=2, num_layers=2, dim_feedforward=32,
            pos_mode='relative', **kwargs,
        )
        return spec, model

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_forward_shapes(self, n):
        spec, model = self.tiny_net(n)
        x = torch.randn(3, spec.num_tokens, spec.token_dim)
        policy_logits, value = model(x)
        assert policy_logits.shape == (3, spec.num_moves)
        assert value.shape == (3,)

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_gradients_flow_including_relative_bias_gate(self, n):
        spec, model = self.tiny_net(n)
        x = torch.randn(2, spec.num_tokens, spec.token_dim)
        policy_logits, value = model(x)
        (policy_logits.sum() + value.sum()).backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f'{name} got no gradient'
            assert torch.isfinite(p.grad).all(), f'{name} got a non-finite gradient'
        assert model.relative_bias.gate.grad is not None

    def test_works_with_batch_size_one(self):
        # The batch expansion in forward() (unsqueeze/expand/reshape)
        # is the one place a batch-size-1 edge case could silently
        # break; check it explicitly rather than only at batch=2/3.
        spec, model = self.tiny_net(9)
        x = torch.randn(1, spec.num_tokens, spec.token_dim)
        policy_logits, value = model(x)
        assert policy_logits.shape == (1, spec.num_moves)

    def test_invalid_pos_mode_rejected(self):
        with pytest.raises(ValueError):
            TokenTransformerNet(BoardSpec(board_size=9), pos_mode='gab')


class TestZeroAgentIntegrationRelativeMode:
    def test_select_move_runs_on_tiny_board(self):
        board_size = 5
        spec = BoardSpec(board_size=board_size, history_depth=1)
        model = TokenTransformerNet(
            spec, d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
            pos_mode='relative',
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
        assert -1.0 <= diag['root_value'] <= 1.0
