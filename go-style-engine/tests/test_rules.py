"""
Test suite for the Go rules layer (goboard.py, scoring.py, gotypes.py, zobrist.py).

This suite serves as an executable specification for a future Swift port.
Tests are organized by behavior and use plain construction of positions via
sequences of apply_move with Move.play / Move.pass_turn.

Behaviors marked with @pytest.mark.xfail(strict=True, ...) describe DESIRED
behavior that the current code does not yet implement. These tests will pass
(treated as expected failures) today and flip to failures when the fix lands.
"""

import pytest

from engine.gotypes import Player, Point
from engine.goboard import Board, GameState, Move, IllegalMoveError
from engine.scoring import evaluate_territory, compute_game_result, GameResult, DEFAULT_KOMI
from engine import zobrist


# ---------------------------------------------------------------------------
# Board bounds happy path (holds today)
# ---------------------------------------------------------------------------

class TestBoardBounds:
    """Verify that Board(13,13) and Board(19,19) construct and accept plays."""

    def test_board_13x13_constructs(self):
        board = Board(13, 13)
        assert board.num_rows == 13
        assert board.num_cols == 13

    def test_board_13x13_accepts_plays(self):
        board = Board(13, 13)
        board.place_stone(Player.black, Point(1, 1))
        board.place_stone(Player.white, Point(2, 2))
        assert board.get(Point(1, 1)) == Player.black
        assert board.get(Point(2, 2)) == Player.white

    def test_board_19x19_constructs(self):
        board = Board(19, 19)
        assert board.num_rows == 19
        assert board.num_cols == 19

    def test_board_19x19_accepts_plays(self):
        board = Board(19, 19)
        board.place_stone(Player.black, Point(1, 1))
        board.place_stone(Player.white, Point(19, 19))
        assert board.get(Point(1, 1)) == Player.black
        assert board.get(Point(19, 19)) == Player.white


# ---------------------------------------------------------------------------
# Capture (holds today)
# ---------------------------------------------------------------------------

class TestCapture:
    """Surrounding an opponent stone/group so it has zero liberties removes it."""

    def test_capture_single_stone_small_board(self):
        """Black captures a single white stone by surrounding it completely."""
        # Set up position directly on a board: white stone with 1 liberty
        board = Board(5, 5)
        board.place_stone(Player.white, Point(3, 3))
        board.place_stone(Player.black, Point(3, 2))
        board.place_stone(Player.black, Point(2, 3))
        board.place_stone(Player.black, Point(4, 3))
        # White at (3,3) has 1 liberty at (3,4). Now create a GameState
        # with black to move to test the capture.
        # Use a minimal GameState wrapping this board.
        game = GameState(board, Player.black, None, None)
        # Black plays at (3,4) to capture white
        assert game.is_valid_move(Move.play(Point(3, 4)))
        game = game.apply_move(Move.play(Point(3, 4)))
        # White stone should be removed
        assert game.board.get(Point(3, 3)) is None
        # Black stones should still be on the board
        assert game.board.get(Point(3, 2)) == Player.black
        assert game.board.get(Point(2, 3)) == Player.black
        assert game.board.get(Point(4, 3)) == Player.black
        assert game.board.get(Point(3, 4)) == Player.black

    def test_capture_removes_opponent_group(self):
        """When an opponent group has zero liberties after a play,
        the entire group is removed."""
        # Set up: white group of 2 stones with 1 liberty
        board = Board(5, 5)
        board.place_stone(Player.white, Point(3, 3))
        board.place_stone(Player.white, Point(3, 4))
        # Surround: black fills all liberties except one
        board.place_stone(Player.black, Point(2, 3))
        board.place_stone(Player.black, Point(2, 4))
        board.place_stone(Player.black, Point(4, 3))
        board.place_stone(Player.black, Point(4, 4))
        board.place_stone(Player.black, Point(3, 2))
        # White group liberties: (3,5) is the only one left
        # Create game with black to move
        game = GameState(board, Player.black, None, None)
        # Black plays at (3,5) - should capture both white stones
        assert game.is_valid_move(Move.play(Point(3, 5)))
        game = game.apply_move(Move.play(Point(3, 5)))
        assert game.board.get(Point(3, 3)) is None
        assert game.board.get(Point(3, 4)) is None


# ---------------------------------------------------------------------------
# Suicide forbidden (holds today)
# ---------------------------------------------------------------------------

class TestSuicideForbidden:
    """Self-capture plays are invalid. Capturing plays are legal."""

    def test_self_capture_invalid(self):
        """Playing into a point where your stone would have no liberties
        and capture nothing is invalid."""
        # Construct a solid black ring where each black stone has >1 liberty.
        # White playing in the center would have 0 liberties and capture nothing.
        board = Board(5, 5)
        # Build a black ring: all 8 points around (3,3) are black.
        # Each black stone has liberties beyond the center:
        # (2,2) has (1,2),(2,1); (2,3) has (1,3); (2,4) has (1,4),(2,5);
        # (3,2) has (3,1); (3,4) has (3,5); (4,2) has (5,2),(4,1);
        # (4,3) has (5,3); (4,4) has (5,4),(4,5).
        # The black group is connected with 12+ liberties total.
        for r, c in [(2, 2), (2, 3), (2, 4),
                     (3, 2), (3, 4),
                     (4, 2), (4, 3), (4, 4)]:
            board.place_stone(Player.black, Point(r, c))
        # Now white to move; center (3,3) is empty but surrounded.
        # White playing at (3,3) would be self-capture (no liberties,
        # no opponent stones with 1 liberty to capture).
        game = GameState(board, Player.white, None, None)
        white_self_capture = Move.play(Point(3, 3))
        assert not game.is_valid_move(white_self_capture)

    def test_capture_legal_even_into_eye_shape(self):
        """A play that captures an opponent stone is legal even when the
        played stone itself would have no liberties after placement."""
        # White has a single stone with 1 liberty. Black plays at that
        # liberty point, capturing white. This is legal because it captures.
        board = Board(5, 5)
        board.place_stone(Player.white, Point(3, 3))
        board.place_stone(Player.black, Point(3, 2))
        board.place_stone(Player.black, Point(2, 3))
        board.place_stone(Player.black, Point(4, 3))
        # White at (3,3) has 1 liberty at (3,4). Black to move.
        game = GameState(board, Player.black, None, None)
        capture_move = Move.play(Point(3, 4))
        assert game.is_valid_move(capture_move)
        game = game.apply_move(capture_move)
        assert game.board.get(Point(3, 3)) is None  # white captured
        assert game.board.get(Point(3, 4)) == Player.black


# ---------------------------------------------------------------------------
# Situational superko (holds today)
# ---------------------------------------------------------------------------

class TestSuperko:
    """After a capture-recapture cycle, a move recreating a previous
    (player, board-hash) situation is rejected."""

    @staticmethod
    def _game_at_ko_moment():
        """Build a genuine single-stone ko via apply_move.

        Final position before the ko capture (next player: black):
          white: (3,3), (2,4), (4,4), (3,5)
          black: (2,3), (4,3), (3,2)
        Black then captures white (3,3) by playing (3,4). The black
        stone at (3,4) is a one-stone group whose only liberty is (3,3),
        so white recapturing at (3,3) would capture it and restore the
        exact previous situation -- that recapture is the ko violation.
        Note the recapture IS a capturing move (not suicide): the only
        rule rejecting it is ko.
        """
        game = GameState.new_game((5, 5))
        game = game.apply_move(Move.play(Point(2, 3)))  # black
        game = game.apply_move(Move.play(Point(2, 4)))  # white
        game = game.apply_move(Move.play(Point(4, 3)))  # black
        game = game.apply_move(Move.play(Point(4, 4)))  # white
        game = game.apply_move(Move.play(Point(3, 2)))  # black
        game = game.apply_move(Move.play(Point(3, 5)))  # white
        game = game.apply_move(Move.pass_turn())        # black
        game = game.apply_move(Move.play(Point(3, 3)))  # white: ko bait
        game = game.apply_move(Move.play(Point(3, 4)))  # black captures
        return game

    def test_single_stone_ko_recapture_rejected(self):
        """Classic ko: after capture, the recapture that would restore
        the prior position is illegal."""
        game = self._game_at_ko_moment()
        recapture = Move.play(Point(3, 3))
        assert not game.is_valid_move(recapture)

    def test_ko_violation_via_does_move_violate_ko(self):
        """Direct test of does_move_violate_ko with a ko situation."""
        game = self._game_at_ko_moment()
        ko_move = Move.play(Point(3, 3))
        assert game.does_move_violate_ko(Player.white, ko_move)

    def test_ko_resolves_after_intervening_move(self):
        """After both sides play elsewhere, the ko point becomes legal
        again (the situation no longer repeats)."""
        game = self._game_at_ko_moment()
        game = game.apply_move(Move.play(Point(1, 1)))  # white elsewhere
        game = game.apply_move(Move.play(Point(5, 5)))  # black elsewhere
        assert game.is_valid_move(Move.play(Point(3, 3)))


# ---------------------------------------------------------------------------
# Two-pass termination (holds today)
# ---------------------------------------------------------------------------

class TestTwoPassTermination:
    """is_over() returns False initially and after one pass,
    True after two consecutive passes or resign."""

    def test_not_over_initially(self):
        game = GameState.new_game((5, 5))
        assert not game.is_over()

    def test_not_over_after_one_pass(self):
        game = GameState.new_game((5, 5))
        game = game.apply_move(Move.pass_turn())
        assert not game.is_over()

    def test_over_after_two_passes(self):
        game = GameState.new_game((5, 5))
        game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.pass_turn())
        assert game.is_over()

    def test_over_after_resign(self):
        game = GameState.new_game((5, 5))
        game = game.apply_move(Move.resign())
        assert game.is_over()


# ---------------------------------------------------------------------------
# Invalid moves (holds today)
# ---------------------------------------------------------------------------

class TestInvalidMoves:
    """Playing on an occupied point is invalid."""

    def test_play_on_occupied_point_invalid(self):
        game = GameState.new_game((5, 5))
        game = game.apply_move(Move.play(Point(3, 3)))
        # The point is occupied — playing there again is invalid
        invalid_move = Move.play(Point(3, 3))
        assert not game.is_valid_move(invalid_move)


# ---------------------------------------------------------------------------
# Area scoring semantics (holds today)
# ---------------------------------------------------------------------------

class TestAreaScoring:
    """Verify territory evaluation and game result computation on small boards."""

    def test_stones_count_for_owner(self):
        """Each stone counts toward its owner's score."""
        board = Board(5, 5)
        board.place_stone(Player.black, Point(3, 3))
        board.place_stone(Player.white, Point(3, 4))
        territory = evaluate_territory(board)
        assert territory.num_black_stones == 1
        assert territory.num_white_stones == 1

    def test_enclosed_empty_region_counts_for_border_color(self):
        """Empty region bordered exclusively by one color counts as
        territory for that color; regions bordered by both are neutral."""
        board = Board(9, 9)
        # Black ring around (4,4); white ring around (7,7). The rings
        # don't touch each other or the edge, so all remaining empty
        # points form one region bordering BOTH colors -> dame.
        for r, c in [(3, 3), (3, 4), (3, 5),
                     (4, 3), (4, 5),
                     (5, 3), (5, 4), (5, 5)]:
            board.place_stone(Player.black, Point(r, c))
        for r, c in [(6, 6), (6, 7), (6, 8),
                     (7, 6), (7, 8),
                     (8, 6), (8, 7), (8, 8)]:
            board.place_stone(Player.white, Point(r, c))
        territory = evaluate_territory(board)
        assert territory.num_black_stones == 8
        assert territory.num_white_stones == 8
        # Interiors (4,4) and (7,7) are bordered by a single color each.
        assert territory.num_black_territory == 1
        assert territory.num_white_territory == 1
        # Everything else (81 - 16 stones - 2 territory) is dame.
        assert territory.num_dame == 63

    def test_single_color_board_counts_all_empty_as_territory(self):
        """Document the simple algorithm's behavior: with only one color
        on the board, every empty region borders just that color, so the
        whole exterior counts as territory (no dead-group detection)."""
        board = Board(9, 9)
        for r, c in [(4, 4), (4, 5), (4, 6),
                     (5, 4), (5, 6),
                     (6, 4), (6, 5), (6, 6)]:
            board.place_stone(Player.black, Point(r, c))
        territory = evaluate_territory(board)
        assert territory.num_black_stones == 8
        assert territory.num_black_territory == 73  # interior + exterior

    def test_mixed_border_empty_region_is_dame(self):
        """Empty region bordered by both colors is neutral (dame)."""
        board = Board(9, 9)
        # Black at (5, 4), white at (5, 6)
        # Empty point (5, 5) between them
        board.place_stone(Player.black, Point(5, 4))
        board.place_stone(Player.white, Point(5, 6))
        territory = evaluate_territory(board)
        # (5,5) is bordered by both black and white → dame
        assert Point(5, 5) in territory.dame_points
        # Also verify that (5,4) and (5,6) are stones, not territory
        assert territory.num_black_stones == 1
        assert territory.num_white_stones == 1

    def test_compute_game_result_margin(self):
        """compute_game_result returns correct margin on a finished position."""
        game = GameState.new_game((5, 5))
        # Simple position: one black stone, one white stone, then two passes
        game = game.apply_move(Move.play(Point(3, 3)))  # black
        game = game.apply_move(Move.play(Point(3, 4)))  # white
        game = game.apply_move(Move.pass_turn())  # black passes
        game = game.apply_move(Move.pass_turn())  # white passes
        result = compute_game_result(game, komi=0)
        # Each has 1 stone. On 5x5, the empty space is all connected to
        # the edge. With stones of both colors on the board, the empty
        # regions are bordered by both colors → dame (mostly).
        # But the simple algorithm may count some as territory.
        # Key assertion: both players have equal stones (1 each).
        assert result.b == result.w
        assert result.winning_margin == 0

    def test_compute_game_result_with_komi(self):
        """compute_game_result applies komi to white's score."""
        game = GameState.new_game((5, 5))
        game = game.apply_move(Move.play(Point(3, 3)))  # black
        game = game.apply_move(Move.pass_turn())  # white passes
        game = game.apply_move(Move.pass_turn())  # black passes
        result = compute_game_result(game, komi=7.5)
        # The result includes komi in the comparison
        assert result.komi == 7.5
        # With one black stone and vast black "territory" (due to algorithm
        # limitation), black wins by a lot. But the key is komi is applied.
        assert result.b > result.w + result.komi  # black wins


# ---------------------------------------------------------------------------
# Desired behaviors NOT implemented today (xfail, strict=True)
# ---------------------------------------------------------------------------

class TestDesiredNotImplemented:
    """These describe desired behavior that the current code does not yet
    implement. Tests are marked xfail(strict=True) so the suite is green
    today and will fail loudly when the fix lands."""

    @pytest.mark.xfail(strict=True, reason="is_valid_move raises KeyError for off-grid points; should return False")
    def test_off_grid_move_invalid(self):
        """is_valid_move(Move.play(Point(0, 0))) should return False,
        not raise KeyError."""
        game = GameState.new_game((5, 5))
        off_grid = Move.play(Point(0, 0))
        assert not game.is_valid_move(off_grid)

    @pytest.mark.xfail(strict=True, reason="legal_moves returns [pass, resign] after game over; should return []")
    def test_terminal_legal_moves_empty(self):
        """After is_over(), legal_moves() should return []."""
        game = GameState.new_game((5, 5))
        game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.pass_turn())
        assert game.is_over()
        assert game.legal_moves() == []

    @pytest.mark.xfail(strict=True, reason="Move XOR assert accepts three-true (point+is_pass+is_resign); should reject")
    def test_move_constructor_rejects_three_kinds(self):
        """Constructing Move with all three kind fields set should raise.
        (The two-true case already raises correctly via XOR; the bug is
        specifically three-true which XOR evaluates to True.)"""
        with pytest.raises(AssertionError):
            Move(point=Point(1, 1), is_pass=True, is_resign=True)

    @pytest.mark.xfail(strict=True, reason="GameState has no komi attribute; should carry komi preserved through apply_move")
    def test_komi_on_game_state(self):
        """GameState should carry a komi attribute preserved through
        apply_move, and compute_game_result should use it."""
        game = GameState.new_game((5, 5))
        assert hasattr(game, 'komi')
        assert game.komi == DEFAULT_KOMI
        game = game.apply_move(Move.play(Point(1, 1)))
        assert game.komi == DEFAULT_KOMI

    @pytest.mark.xfail(strict=True, reason="GameResult.winner awards White on tie; should be None for equal scores with integer komi")
    def test_tie_handling_with_integer_komi(self):
        """With integer komi and equal scores, GameResult.winner should be
        None."""
        game = GameState.new_game((5, 5))
        game = game.apply_move(Move.play(Point(3, 3)))  # black
        game = game.apply_move(Move.play(Point(3, 4)))  # white
        game = game.apply_move(Move.pass_turn())
        game = game.apply_move(Move.pass_turn())
        result = compute_game_result(game, komi=0)
        assert result.b == result.w
        assert result.winner is None

    @pytest.mark.xfail(strict=True, reason="Board(20,20) constructs without error; should raise because zobrist table only supports up to 19x19")
    def test_board_size_validation_20x20(self):
        """Constructing Board(20, 20) should raise immediately because
        the zobrist table only supports up to 19x19."""
        with pytest.raises(Exception):
            Board(20, 20)

    @pytest.mark.xfail(strict=True, reason="compute_game_result scores non-terminal positions; should raise")
    def test_early_scoring_rejection(self):
        """compute_game_result on a non-terminal GameState should raise."""
        game = GameState.new_game((5, 5))
        assert not game.is_over()
        with pytest.raises(Exception):
            compute_game_result(game)
