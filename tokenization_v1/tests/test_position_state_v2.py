import os
import sys

import chess
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import position_state as ps  # noqa: E402


def test_castling_rights_all_available_start():
    board = chess.Board()
    rights = ps.castling_rights(board)
    assert rights == {
        "white_kingside": True, "white_queenside": True,
        "black_kingside": True, "black_queenside": True,
    }


def test_castling_rights_lost_after_king_move():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push_san("Kd1")
    rights = ps.castling_rights(board)
    assert rights["white_kingside"] is False
    assert rights["white_queenside"] is False
    assert rights["black_kingside"] is True
    assert rights["black_queenside"] is True


def test_castling_move_type_flag():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    move = board.parse_san("O-O")
    flags = ps.move_type_flags(board, move)
    assert flags["castling"] is True
    assert flags["capture"] is False
    assert flags["quiet"] is False


def test_en_passant_square_available():
    board = chess.Board()
    for san in ["e4", "a6", "e5", "d5"]:
        board.push_san(san)
    # blancas pueden capturar al paso en d6
    assert ps.en_passant_square(board) == chess.D6


def test_en_passant_square_not_available_start():
    board = chess.Board()
    assert ps.en_passant_square(board) == -1


def test_en_passant_move_type_flag():
    board = chess.Board()
    for san in ["e4", "a6", "e5", "d5"]:
        board.push_san(san)
    move = board.parse_san("exd6")
    flags = ps.move_type_flags(board, move)
    assert flags["en_passant"] is True
    assert flags["capture"] is True


@pytest.mark.parametrize("promo_san,piece_type", [
    ("a8=Q", chess.QUEEN), ("a8=R", chess.ROOK), ("a8=B", chess.BISHOP), ("a8=N", chess.KNIGHT),
])
def test_promotion_move_type_flag(promo_san, piece_type):
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    move = board.parse_san(promo_san)
    flags = ps.move_type_flags(board, move)
    assert flags["promotion"] is True


def test_multiple_promotions_legal_simultaneously():
    """PASO: posicion con varias promociones legales (a8=Q/R/B/N todas legales
    desde la misma posicion de partida)."""
    from legal_moves import legal_action_ids
    from tokenizer import move_to_action_id

    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    legal_ids = set(legal_action_ids(board))
    for promo_san in ["a8=Q", "a8=R", "a8=B", "a8=N"]:
        move = board.parse_san(promo_san)
        assert move_to_action_id(move) in legal_ids


def test_is_check_flag():
    # torre negra en e8 da jaque al rey blanco en e1 a lo largo de la columna e
    board = chess.Board("4r3/8/8/8/8/8/8/4K3 w - - 0 1")
    flags = ps.position_flags(board)
    assert flags["is_check"] is True
    assert flags["is_checkmate"] is False


FOOLS_MATE_MOVES = ["f3", "e5", "g4", "Qh4#"]


def test_checkmate_detected_fools_mate():
    board = chess.Board()
    for san in FOOLS_MATE_MOVES:
        board.push_san(san)
    flags = ps.position_flags(board)
    assert flags["is_checkmate"] is True
    assert flags["is_check"] is True
    assert flags["is_stalemate"] is False


def test_checkmate_move_type_flag():
    board = chess.Board()
    for san in FOOLS_MATE_MOVES[:-1]:
        board.push_san(san)
    move = board.parse_san("Qh4#")
    flags = ps.move_type_flags(board, move)
    assert flags["checkmate"] is True
    assert flags["check"] is True


def test_stalemate_detected():
    # posicion de ahogado clasica: negras al turno, sin jaque, sin movimientos legales
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    flags = ps.position_flags(board)
    assert flags["is_stalemate"] is True
    assert flags["is_check"] is False
    assert flags["is_checkmate"] is False


def test_halfmove_clock_increments_and_resets():
    board = chess.Board()
    assert board.halfmove_clock == 0
    board.push_san("Nf3")  # movimiento de pieza, sin captura ni peon: incrementa
    assert board.halfmove_clock == 1
    board.push_san("Nf6")
    assert board.halfmove_clock == 2
    board.push_san("e4")  # movimiento de peon: resetea a 0
    assert board.halfmove_clock == 0


def test_fifty_move_rule_can_claim_draw():
    # posicion cercana a 50 movimientos sin captura ni peon (halfmove_clock=99)
    board = chess.Board("7R/8/8/4k3/8/4K3/8/8 w - - 99 60")
    assert board.halfmove_clock == 99
    move = board.parse_san("Ra8")
    board.push(move)
    assert board.halfmove_clock == 100
    assert board.can_claim_fifty_moves() is True


def test_board_to_tensor_v2_shape_dtype():
    board = chess.Board()
    t = ps.board_to_tensor_v2(board)
    assert t.shape == (23, 8, 8)
    assert t.dtype == np.float32


def test_board_to_tensor_v2_orientation_absolute():
    """Plano 0 (peones blancos) debe reflejar la posicion REAL de las
    piezas blancas -- orientacion absoluta, no volteada para negras."""
    board = chess.Board()
    t = ps.board_to_tensor_v2(board)
    # peones blancos en fila 2 (indice de fila 1, ya que row=divmod(square,8)[0])
    row, col = divmod(chess.E2, 8)
    assert t[0, row, col] == 1.0
    # peones negros (plano 6) en fila 7
    row_b, col_b = divmod(chess.E7, 8)
    assert t[6, row_b, col_b] == 1.0


def test_board_to_tensor_v2_side_to_move_plane():
    board = chess.Board()
    t = ps.board_to_tensor_v2(board)
    assert (t[12] == 1.0).all()
    board.push_san("e4")
    t2 = ps.board_to_tensor_v2(board)
    assert (t2[12] == 0.0).all()


def test_board_to_tensor_v2_consistency_across_positions():
    """Consistencia: mismo shape/dtype en posiciones muy distintas (apertura,
    medio juego, final con pocas piezas)."""
    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/8/8/4k3/8/4K3/8/7R w - - 0 1",
    ]
    shapes = set()
    dtypes = []
    for fen in fens:
        t = ps.board_to_tensor_v2(chess.Board(fen))
        shapes.add(t.shape)
        dtypes.append(t.dtype)
    assert shapes == {(23, 8, 8)}
    assert all(d == np.dtype(np.float32) for d in dtypes)


def test_material_counts_and_score_start_position():
    board = chess.Board()
    wc = ps.material_counts(board, chess.WHITE)
    assert wc == {"pawns": 8, "knights": 2, "bishops": 2, "rooks": 2, "queens": 1}
    assert ps.material_score(board, chess.WHITE) == 8 * 1 + 2 * 3 + 2 * 3 + 2 * 5 + 1 * 9
    assert ps.material_score(board, chess.WHITE) == ps.material_score(board, chess.BLACK)


def test_game_phase_opening_middlegame_endgame():
    assert ps.game_phase(chess.Board()) == "opening"
    assert ps.game_phase(chess.Board("8/8/8/4k3/8/4K3/8/7R w - - 0 1")) == "endgame"
