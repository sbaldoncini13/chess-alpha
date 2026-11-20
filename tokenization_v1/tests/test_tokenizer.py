import os
import sys

import chess
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from tokenizer import (
    ACTION_ID_MAX,
    ACTION_ID_MIN,
    action_id_to_move,
    move_to_action_id,
    promo_code_to_piece,
)


def roundtrip(move: chess.Move) -> chess.Move:
    action_id = move_to_action_id(move)
    return action_id_to_move(action_id)


def test_normal_move_e2e4():
    board = chess.Board()
    move = board.parse_san("e4")
    decoded = roundtrip(move)
    assert decoded == move
    assert decoded.from_square == chess.E2
    assert decoded.to_square == chess.E4
    assert decoded.promotion is None


def test_normal_move_g1f3():
    board = chess.Board()
    move = board.parse_san("Nf3")
    decoded = roundtrip(move)
    assert decoded == move


def test_capture_normal():
    board = chess.Board()
    for san in ["e4", "d5"]:
        board.push_san(san)
    move = board.parse_san("exd5")
    decoded = roundtrip(move)
    assert decoded == move
    board.push(decoded)
    assert board.piece_at(chess.D5).piece_type == chess.PAWN
    assert board.piece_at(chess.D5).color == chess.WHITE


CASTLE_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"


def test_white_kingside_castle():
    board = chess.Board(CASTLE_FEN)
    move = board.parse_san("O-O")
    decoded = roundtrip(move)
    assert decoded == move
    board.push(decoded)
    assert board.piece_at(chess.G1).piece_type == chess.KING
    assert board.piece_at(chess.F1).piece_type == chess.ROOK


def test_white_queenside_castle():
    board = chess.Board(CASTLE_FEN)
    move = board.parse_san("O-O-O")
    decoded = roundtrip(move)
    assert decoded == move
    board.push(decoded)
    assert board.piece_at(chess.C1).piece_type == chess.KING
    assert board.piece_at(chess.D1).piece_type == chess.ROOK


def test_black_kingside_castle():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
    board = chess.Board(fen)
    move = board.parse_san("O-O")
    decoded = roundtrip(move)
    assert decoded == move
    board.push(decoded)
    assert board.piece_at(chess.G8).piece_type == chess.KING
    assert board.piece_at(chess.F8).piece_type == chess.ROOK


def test_black_queenside_castle():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
    board = chess.Board(fen)
    move = board.parse_san("O-O-O")
    decoded = roundtrip(move)
    assert decoded == move
    board.push(decoded)
    assert board.piece_at(chess.C8).piece_type == chess.KING
    assert board.piece_at(chess.D8).piece_type == chess.ROOK


def test_en_passant():
    board = chess.Board()
    for san in ["e4", "a6", "e5", "d5"]:
        board.push_san(san)
    move = board.parse_san("exd6")
    assert board.is_en_passant(move)
    decoded = roundtrip(move)
    assert decoded == move
    board.push(decoded)
    assert board.piece_at(chess.D6).piece_type == chess.PAWN
    assert board.piece_at(chess.D5) is None  # peon negro capturado al paso


@pytest.mark.parametrize("promo_san,piece_type", [
    ("a8=Q", chess.QUEEN),
    ("a8=R", chess.ROOK),
    ("a8=B", chess.BISHOP),
    ("a8=N", chess.KNIGHT),
])
def test_promotions(promo_san, piece_type):
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    move = board.parse_san(promo_san)
    decoded = roundtrip(move)
    assert decoded == move
    board.push(decoded)
    assert board.piece_at(chess.A8).piece_type == piece_type
    assert board.piece_at(chess.A8).color == chess.WHITE


@pytest.mark.parametrize("promo_san,piece_type", [
    ("axb8=Q", chess.QUEEN),
    ("axb8=R", chess.ROOK),
    ("axb8=B", chess.BISHOP),
    ("axb8=N", chess.KNIGHT),
])
def test_capture_with_promotion(promo_san, piece_type):
    board = chess.Board("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    move = board.parse_san(promo_san)
    decoded = roundtrip(move)
    assert decoded == move
    board.push(decoded)
    assert board.piece_at(chess.B8).piece_type == piece_type
    assert board.piece_at(chess.B8).color == chess.WHITE


def test_reject_promo_code_out_of_range():
    with pytest.raises(ValueError):
        promo_code_to_piece(5)
    with pytest.raises(ValueError):
        promo_code_to_piece(-1)


def test_reject_action_id_out_of_range():
    assert ACTION_ID_MIN == 0
    assert ACTION_ID_MAX == 20479
    with pytest.raises(ValueError):
        action_id_to_move(-1)
    with pytest.raises(ValueError):
        action_id_to_move(20480)
    # promo_code=5 codificado a mano cae justo 1 por encima del maximo valido
    with pytest.raises(ValueError):
        action_id_to_move((5 << 12) | 0)


def test_full_game_reconstruction():
    sample_moves_san = [
        "e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7",
        "Re1", "b5", "Bb3", "d6", "c3", "O-O",
    ]
    board = chess.Board()
    action_ids = []
    for san in sample_moves_san:
        move = board.parse_san(san)
        action_ids.append(move_to_action_id(move))
        board.push(move)
    expected_fen = board.fen()

    replay_board = chess.Board()
    for action_id in action_ids:
        decoded_move = action_id_to_move(action_id)
        assert decoded_move in replay_board.legal_moves
        replay_board.push(decoded_move)

    assert replay_board.fen() == expected_fen
