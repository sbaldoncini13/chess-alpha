"""#14 -- amenazas inmediatas, conservador, sin motor de busqueda nuevo:
solo mira jugadas LEGALES desde la posicion actual (profundidad 1) o del
rival si se le pasara el turno (profundidad 1 tambien, via el mismo truco
de flip-turn que mobility.py, con ep_square limpiado igual que alli)."""
import chess

from see import PIECE_VALUES, static_exchange_eval


def _mate_in_1_available(board: chess.Board) -> bool:
    for move in board.legal_moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return True
    return False


def _major_piece_hanging(board: chess.Board, color: bool, attackers_count_white, attackers_count_black) -> bool:
    enemy_attackers = attackers_count_black if color == chess.WHITE else attackers_count_white
    own_attackers = attackers_count_white if color == chess.WHITE else attackers_count_black
    for pt in (chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            if enemy_attackers[sq] > 0 and own_attackers[sq] == 0:
                return True
    return False


def _promotion_threat(board: chess.Board) -> bool:
    return any(move.promotion is not None for move in board.legal_moves)


def _forcing_check_available(board: chess.Board) -> bool:
    return any(board.gives_check(move) for move in board.legal_moves)


def _forcing_capture_available(board: chess.Board) -> bool:
    return any(board.is_capture(move) for move in board.legal_moves)


def _opponent_best_see_if_their_turn(board: chess.Board, threshold: int) -> bool:
    board.turn = not board.turn
    saved_ep = board.ep_square
    board.ep_square = None
    try:
        best = -999
        for move in board.legal_moves:
            if board.is_capture(move):
                best = max(best, static_exchange_eval(board, move))
        return best >= threshold
    finally:
        board.turn = not board.turn
        board.ep_square = saved_ep


def compute_threats(board: chess.Board, attackers_count_white, attackers_count_black, config: dict) -> dict:
    color = board.turn
    threshold = config["tactical_thresholds"]["immediate_material_threat_min_value"]
    return {
        "mate_in_1_available": _mate_in_1_available(board),
        "major_piece_hanging": _major_piece_hanging(board, color, attackers_count_white, attackers_count_black),
        "promotion_threat": _promotion_threat(board),
        "forcing_check_available": _forcing_check_available(board),
        "forcing_capture_available": _forcing_capture_available(board),
        "immediate_material_threat_by_opponent": _opponent_best_see_if_their_turn(board, threshold),
    }
