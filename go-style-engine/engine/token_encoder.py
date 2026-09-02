"""Intersection-token board encoder for the trans-go-former Transformer
body. See docs/board-specification.md for the design rationale; this is
the code half of that spec, not a replacement for ZeroEncoder (the CNN
encoder in encoder.py) -- both read the same GameState.

One token per intersection, occupancy alphabet {empty, self, opponent}
plus a per-token ko-illegal flag, times a short move-history window.
Everything is a function of BoardSpec, so changing board_size is a one
field edit -- see tests/test_token_encoder.py for the parametrized
proof at 9x9/13x13/19x19.
"""
from dataclasses import dataclass

import numpy as np

from .goboard import Move
from .gotypes import Player, Point

__all__ = ['BoardSpec', 'TokenEncoder']


@dataclass(frozen=True)
class BoardSpec:
    board_size: int = 13
    history_depth: int = 7

    @property
    def num_tokens(self) -> int:
        return self.board_size * self.board_size

    @property
    def token_dim(self) -> int:
        # [is_empty, is_self, is_opponent, is_ko_illegal] per history frame
        return 4 * (self.history_depth + 1)

    @property
    def num_moves(self) -> int:
        return self.num_tokens + 1  # + pass


class TokenEncoder:
    """Encodes a GameState into a (num_tokens, token_dim) float32 array.

    Token index i for Point(row=r, col=c) is N*(r-1) + (c-1), matching
    ZeroEncoder.encode_move / decode_move_index exactly (see
    docs/board-specification.md:2), so the move-index convention is
    drop-in compatible even though the two encoders share no code.
    """

    def __init__(self, spec: BoardSpec = None):
        self.spec = spec or BoardSpec()

    def encode(self, game_state) -> np.ndarray:
        spec = self.spec
        N = spec.board_size
        tokens = np.zeros((spec.num_tokens, spec.token_dim), dtype=np.float32)

        perspective = game_state.next_player
        state = game_state
        for frame in range(spec.history_depth + 1):
            base = frame * 4
            if state is not None:
                # is_ko_illegal only computed for frame 0 (the current
                # position): it answers "can the player to move play
                # here right now", which is only well-defined for the
                # live position, not a hypothetical replay against a
                # historical board (see docs/board-specification.md).
                check_ko = (frame == 0)
                self._fill_frame(tokens, state, perspective, base, N, check_ko)
                state = state.previous_state
            # else: leave this frame's slice as all-zero ("is_empty" not
            # even set) -- distinguishable from an actual empty board,
            # which does set is_empty=1. Only matters before move 0.

        return tokens

    def _fill_frame(self, tokens, state, perspective, base, N, check_ko):
        board = state.board
        for r in range(1, N + 1):
            for c in range(1, N + 1):
                p = Point(row=r, col=c)
                i = N * (r - 1) + (c - 1)
                occupant = board.get(p)
                if occupant is None:
                    tokens[i, base + 0] = 1.0  # is_empty
                    if check_ko and state.does_move_violate_ko(perspective, Move.play(p)):
                        tokens[i, base + 3] = 1.0  # is_ko_illegal
                elif occupant == perspective:
                    tokens[i, base + 1] = 1.0  # is_self
                else:
                    tokens[i, base + 2] = 1.0  # is_opponent

    def encode_move(self, move) -> int:
        N = self.spec.board_size
        if move.is_play:
            return N * (move.point.row - 1) + (move.point.col - 1)
        elif move.is_pass:
            return self.spec.num_tokens
        raise ValueError('Cannot encode resign move')

    def decode_move_index(self, index) -> Move:
        N = self.spec.board_size
        if index == self.spec.num_tokens:
            return Move.pass_turn()
        row = index // N
        col = index % N
        return Move.play(Point(row=row + 1, col=col + 1))

    def num_moves(self) -> int:
        """Method, not just BoardSpec.num_moves as a property, to match
        ZeroEncoder's API exactly -- engine/mcts.py's ZeroAgent calls
        self.encoder.num_moves() (only reached when a collector is
        attached, e.g. by selfplay.py -- that's why this gap survived
        the earlier ZeroAgent smoke tests, which never attach one)."""
        return self.spec.num_moves
