import os
import sys

import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from position_dedup import GlobalOccurrenceCounter, position_id  # noqa: E402
from repetition import InGameRepetitionTracker  # noqa: E402


def test_position_id_deterministic():
    b1 = chess.Board()
    b2 = chess.Board()
    assert position_id(b1) == position_id(b2)


def test_position_id_differs_on_different_position():
    b1 = chess.Board()
    b2 = chess.Board()
    b2.push_san("e4")
    assert position_id(b1) != position_id(b2)


def test_position_id_sensitive_to_castling_rights():
    """Misma disposicion de piezas pero distintos derechos de enroque debe
    dar distinto position_id (identidad FUNCIONAL, no solo visual)."""
    fen_full_rights = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    fen_no_rights = "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"
    assert position_id(chess.Board(fen_full_rights)) != position_id(chess.Board(fen_no_rights))


def test_position_id_sensitive_to_en_passant():
    board_with_ep = chess.Board()
    for san in ["e4", "a6", "e5", "d5"]:
        board_with_ep.push_san(san)
    # misma disposicion de piezas, pero SIN la posibilidad de en passant activa
    board_no_ep = chess.Board(board_with_ep.fen())
    board_no_ep.ep_square = None
    assert position_id(board_with_ep) != position_id(board_no_ep)


def test_global_occurrence_counter_counts_across_games():
    counter = GlobalOccurrenceCounter()
    start_pid = position_id(chess.Board())
    for _ in range(5):
        counter.add(start_pid)
    assert counter.count_of(start_pid) == 5
    assert counter.count_of(999999) == 0


def test_in_game_repetition_tracker_counts_first_second_third():
    board = chess.Board()
    tracker = InGameRepetitionTracker()
    assert tracker.observe(board) == 1
    board.push_san("Nf3")
    board.push_san("Nf6")
    board.push_san("Ng1")
    board.push_san("Ng8")
    # de vuelta a la posicion inicial: 2da aparicion
    assert tracker.observe(board) == 2
    board.push_san("Nf3")
    board.push_san("Nf6")
    board.push_san("Ng1")
    board.push_san("Ng8")
    # 3ra aparicion -> repeticion triple
    assert tracker.observe(board) == 3


def test_in_game_repetition_tracker_independent_per_game():
    board = chess.Board()
    tracker_a = InGameRepetitionTracker()
    tracker_b = InGameRepetitionTracker()
    assert tracker_a.observe(board) == 1
    assert tracker_a.observe(board) == 2
    # un tracker nuevo (nueva partida) no hereda el conteo del anterior
    assert tracker_b.observe(board) == 1
