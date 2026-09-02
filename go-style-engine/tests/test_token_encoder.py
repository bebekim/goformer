"""Tests for engine/token_encoder.py -- see docs/board-specification.md.

The central claim under test is "board_size is readily changeable": the
same encoder code must produce correctly-shaped, correctly-populated
output at 9x9, 13x13 and 19x19 with no code changes, only the BoardSpec
field.
"""
import numpy as np
import pytest

from engine.encoder import ZeroEncoder
from engine.goboard import GameState, Move
from engine.gotypes import Player, Point
from engine.token_encoder import BoardSpec, TokenEncoder

BOARD_SIZES = (9, 13, 19)


# ---------------------------------------------------------------------------
# BoardSpec is a pure function of its fields
# ---------------------------------------------------------------------------

class TestBoardSpec:
    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_num_tokens_scales_with_board_size(self, n):
        spec = BoardSpec(board_size=n)
        assert spec.num_tokens == n * n

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_num_moves_is_tokens_plus_pass(self, n):
        spec = BoardSpec(board_size=n)
        assert spec.num_moves == n * n + 1

    @pytest.mark.parametrize('depth', [0, 1, 7])
    def test_token_dim_scales_with_history_depth(self, depth):
        spec = BoardSpec(history_depth=depth)
        assert spec.token_dim == 4 * (depth + 1)

    def test_defaults_match_existing_engine_convention(self):
        # 13x13 default matches selfplay.py's --board-size default;
        # history_depth=7 matches Chessformer's n=7.
        spec = BoardSpec()
        assert spec.board_size == 13
        assert spec.history_depth == 7


# ---------------------------------------------------------------------------
# Encoding shape and content, parametrized over board size
# ---------------------------------------------------------------------------

class TestTokenEncoderShape:
    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_encode_shape(self, n):
        spec = BoardSpec(board_size=n, history_depth=2)
        game = GameState.new_game(n)
        encoder = TokenEncoder(spec)
        tokens = encoder.encode(game)
        assert tokens.shape == (n * n, spec.token_dim)
        assert tokens.dtype == np.float32

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_empty_board_all_empty_flags_set(self, n):
        spec = BoardSpec(board_size=n, history_depth=0)
        game = GameState.new_game(n)
        tokens = TokenEncoder(spec).encode(game)
        # frame 0 only (history_depth=0): [is_empty, is_self, is_opponent, is_ko]
        assert np.all(tokens[:, 0] == 1.0)
        assert np.all(tokens[:, 1] == 0.0)
        assert np.all(tokens[:, 2] == 0.0)

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_pre_game_history_frames_are_zero(self, n):
        # New game has no previous_state chain deep enough to fill a
        # history_depth=3 window -- those frames should be all-zero,
        # not "all empty" (is_empty unset too; see token_encoder.py).
        spec = BoardSpec(board_size=n, history_depth=3)
        game = GameState.new_game(n)
        tokens = TokenEncoder(spec).encode(game)
        frame1 = tokens[:, 4:8]
        assert np.all(frame1 == 0.0)


class TestTokenEncoderContent:
    def test_self_and_opponent_flip_with_perspective(self):
        spec = BoardSpec(board_size=9, history_depth=0)
        game = GameState.new_game(9)
        game = game.apply_move(Move.play(Point(3, 3)))  # black plays
        encoder = TokenEncoder(spec)

        i = 9 * (3 - 1) + (3 - 1)
        tokens = encoder.encode(game)
        # next_player is now white; black's stone at (3,3) is "opponent"
        assert tokens[i, 2] == 1.0  # is_opponent
        assert tokens[i, 1] == 0.0  # is_self

        game2 = game.apply_move(Move.play(Point(7, 7)))  # white plays
        tokens2 = encoder.encode(game2)
        j = 9 * (7 - 1) + (7 - 1)
        # next_player is black again; white's stone at (7,7) is "opponent",
        # black's own stone at (3,3) is "self"
        assert tokens2[i, 1] == 1.0
        assert tokens2[j, 2] == 1.0

    def test_history_frames_walk_backward_through_previous_states(self):
        spec = BoardSpec(board_size=9, history_depth=2)
        game = GameState.new_game(9)
        game = game.apply_move(Move.play(Point(1, 1)))
        game = game.apply_move(Move.play(Point(2, 2)))
        game = game.apply_move(Move.play(Point(3, 3)))
        tokens = TokenEncoder(spec).encode(game)

        idx = {p: 9 * (p.row - 1) + (p.col - 1)
               for p in (Point(1, 1), Point(2, 2), Point(3, 3))}

        # frame 0 = current position (all three stones present)
        assert tokens[idx[Point(1, 1)], 0:3].argmax() != 0
        assert tokens[idx[Point(2, 2)], 0:3].argmax() != 0
        assert tokens[idx[Point(3, 3)], 0:3].argmax() != 0

        # frame 1 = position before the last move ((3,3) not yet played)
        base = 4
        assert tokens[idx[Point(3, 3)], base + 0] == 1.0  # empty

        # frame 2 = position before that ((2,2) not yet played either)
        base = 8
        assert tokens[idx[Point(2, 2)], base + 0] == 1.0
        assert tokens[idx[Point(3, 3)], base + 0] == 1.0


# ---------------------------------------------------------------------------
# Move-index convention must match ZeroEncoder's, at the shared default size
# ---------------------------------------------------------------------------

class TestMoveIndexCompatibility:
    def test_play_move_indices_match_zero_encoder(self):
        board_size = 13
        token_encoder = TokenEncoder(BoardSpec(board_size=board_size))
        zero_encoder = ZeroEncoder(board_size)

        for r in (1, 5, 13):
            for c in (1, 7, 13):
                move = Move.play(Point(row=r, col=c))
                assert (token_encoder.encode_move(move)
                        == zero_encoder.encode_move(move))

    def test_pass_move_index_matches_zero_encoder(self):
        board_size = 13
        token_encoder = TokenEncoder(BoardSpec(board_size=board_size))
        zero_encoder = ZeroEncoder(board_size)
        assert (token_encoder.encode_move(Move.pass_turn())
                == zero_encoder.encode_move(Move.pass_turn()))

    @pytest.mark.parametrize('n', BOARD_SIZES)
    def test_decode_move_index_round_trips(self, n):
        encoder = TokenEncoder(BoardSpec(board_size=n))
        for index in (0, n * n // 2, n * n - 1, n * n):
            move = encoder.decode_move_index(index)
            assert encoder.encode_move(move) == index
