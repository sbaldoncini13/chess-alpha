import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from legal_moves import VOCAB_SIZE, expand_legal_mask, legal_action_ids  # noqa: E402
from tokenizer import move_to_action_id  # noqa: E402


def test_legal_move_mask_matches_python_chess_start_position():
    board = chess.Board()
    ids = legal_action_ids(board)
    assert len(ids) == 20  # 20 movimientos legales en la posicion inicial
    expected = {move_to_action_id(m) for m in board.legal_moves}
    assert set(ids) == expected


def test_legal_move_mask_never_out_of_range():
    board = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]:
        board.push_san(san)
        ids = legal_action_ids(board)
        assert all(0 <= i <= VOCAB_SIZE - 1 for i in ids)


def test_expand_legal_mask_matches_ids():
    board = chess.Board()
    ids = legal_action_ids(board)
    mask = expand_legal_mask(ids)
    assert mask.dtype == np.bool_
    assert mask.shape == (VOCAB_SIZE,)
    assert mask.sum() == len(ids)
    for i in ids:
        assert mask[i]


def test_terminal_position_has_empty_mask():
    # jaque mate: no hay movimientos legales
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        board.push_san(san)
    ids = legal_action_ids(board)
    assert ids == []
    mask = expand_legal_mask(ids)
    assert mask.sum() == 0


def test_stalemate_position_has_empty_mask():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    ids = legal_action_ids(board)
    assert ids == []


def test_human_move_always_in_its_own_legal_mask():
    board = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5"]:
        ids_before = legal_action_ids(board)
        move = board.parse_san(san)
        assert move_to_action_id(move) in ids_before
        board.push(move)


def test_promotion_all_variants_present_in_mask():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    ids = set(legal_action_ids(board))
    for promo_san in ["a8=Q", "a8=R", "a8=B", "a8=N"]:
        move = board.parse_san(promo_san)
        assert move_to_action_id(move) in ids
