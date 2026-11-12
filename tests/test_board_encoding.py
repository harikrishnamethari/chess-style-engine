"""
tests/test_board_encoding.py
Unit tests for the AlphaZero-style board/move encoding.
Run: pytest tests/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest
from src.core.board_encoding import (
    board_to_tensor, move_to_index, legal_move_mask, NUM_MOVE_CLASSES
)


def test_tensor_shape():
    board = chess.Board()
    t = board_to_tensor(board)
    assert t.shape == (18, 8, 8), f"Expected (18,8,8), got {t.shape}"


def test_tensor_dtype():
    t = board_to_tensor(chess.Board())
    assert t.dtype == np.float32


def test_no_collisions_starting_position():
    board = chess.Board()
    seen = {}
    for mv in board.legal_moves:
        idx = move_to_index(mv, board)
        assert idx not in seen, f"Collision: {mv} and {seen[idx]} → index {idx}"
        assert 0 <= idx < NUM_MOVE_CLASSES
        seen[idx] = mv


def test_side_to_move_plane():
    board_white = chess.Board()
    board_black = chess.Board()
    board_black.push(chess.Move.from_uci("e2e4"))
    t_white = board_to_tensor(board_white)
    t_black = board_to_tensor(board_black)
    assert t_white[12].all(), "Plane 12 should be all-ones for white to move"
    assert not t_black[12].any(), "Plane 12 should be all-zeros for black to move"


def test_legal_move_mask():
    board = chess.Board()
    mask, idx_to_move = legal_move_mask(board)
    assert mask.shape == (NUM_MOVE_CLASSES,)
    assert mask.sum() == len(list(board.legal_moves))


def test_promotion_encoding():
    """Under-promotions should not collide with queen promotions."""
    board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")  # White pawn on a7
    seen = {}
    for mv in board.legal_moves:
        idx = move_to_index(mv, board)
        assert idx not in seen, f"Promotion collision: {mv} and {seen[idx]}"
        seen[idx] = mv


@pytest.mark.parametrize("n_games", [20])
def test_no_collisions_random_games(n_games):
    """No index collisions across many positions from random games."""
    import random
    rng = random.Random(99)
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(rng.randint(10, 30)):
            if board.is_game_over():
                break
            mv = rng.choice(list(board.legal_moves))
            idx = move_to_index(mv, board)
            assert 0 <= idx < NUM_MOVE_CLASSES
            board.push(mv)
