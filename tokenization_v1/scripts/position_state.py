"""Reconstruccion de estado por posicion + tensorizacion + metadata derivada.

Modulo independiente de tokenization_v1/: no importa nada de src/chess_alpha
ni de search_v1. La semantica de board_to_tensor_v2 replica EXACTAMENTE la
definicion canonica ya existente en src/chess_alpha/encoding.py (leida en
modo solo lectura, nunca modificada ni importada -- ver SCHEMA_V2.md).

Todas las funciones toman un chess.Board ya posicionado y son puras (no
mutan el tablero salvo push/pop balanceado explicito y documentado).
"""
import chess
import numpy as np

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
GAME_PHASE_TO_NUM = {"opening": 0.0, "middlegame": 0.5, "endgame": 1.0}
NUM_PLANES = 23  # debe coincidir con src/chess_alpha/encoding.py:NUM_PLANES

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_counts(board: chess.Board, color: bool) -> dict:
    """Cuenta de piezas por tipo (PASO 15), sin el rey."""
    return {
        "pawns": len(board.pieces(chess.PAWN, color)),
        "knights": len(board.pieces(chess.KNIGHT, color)),
        "bishops": len(board.pieces(chess.BISHOP, color)),
        "rooks": len(board.pieces(chess.ROOK, color)),
        "queens": len(board.pieces(chess.QUEEN, color)),
    }


def material_score(board: chess.Board, color: bool) -> int:
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
    return score


def game_phase(board: chess.Board) -> str:
    """Heuristica identica a src/chess_alpha/board_features.game_phase
    (solo lectura, reimplementada aqui para no crear una dependencia de
    import entre tokenization_v1/ y src/chess_alpha/)."""
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    total_material = material_score(board, chess.WHITE) + material_score(board, chess.BLACK)
    if total_material <= 20 or queens == 0:
        return "endgame"
    elif total_material <= 45:
        return "middlegame"
    else:
        return "opening"


def castling_rights(board: chess.Board) -> dict:
    """PASO 8: derechos reales de la posicion (board.has_*_castling_rights
    ya deriva esto del campo de castling del FEN/estado interno, no de la
    ubicacion actual de rey/torres -- python-chess lo trackea como estado
    propio actualizado en cada push/pop)."""
    return {
        "white_kingside": board.has_kingside_castling_rights(chess.WHITE),
        "white_queenside": board.has_queenside_castling_rights(chess.WHITE),
        "black_kingside": board.has_kingside_castling_rights(chess.BLACK),
        "black_queenside": board.has_queenside_castling_rights(chess.BLACK),
    }


def en_passant_square(board: chess.Board) -> int:
    """PASO 9: -1 si no hay casilla de captura al paso disponible en esta
    posicion. board.ep_square ya es None salvo que la captura al paso sea
    realmente posible segun el estado del juego."""
    return board.ep_square if board.ep_square is not None else -1


def board_to_tensor_v2(board: chess.Board) -> np.ndarray:
    """PASO 6. Replica bit-a-bit src/chess_alpha/encoding.py:board_to_tensor.
    Orientacion absoluta: plano 0 son SIEMPRE peones blancos en su casilla
    real (no se voltea el tablero para negras) -- esa es la convencion
    canonica del proyecto y se preserva sin modificarla."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for color_idx, color in enumerate((chess.WHITE, chess.BLACK)):
        for piece_idx, piece_type in enumerate(PIECE_ORDER):
            plane = color_idx * 6 + piece_idx
            for square in board.pieces(piece_type, color):
                row, col = divmod(square, 8)
                planes[plane, row, col] = 1.0

    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    for square in chess.SQUARES:
        row, col = divmod(square, 8)
        if board.is_attacked_by(chess.WHITE, square):
            planes[17, row, col] = 1.0
        if board.is_attacked_by(chess.BLACK, square):
            planes[18, row, col] = 1.0

    white_material = material_score(board, chess.WHITE)
    black_material = material_score(board, chess.BLACK)
    material_balance = (white_material - black_material) / 39.0
    mobility = min(board.legal_moves.count(), 60) / 60.0
    phase_num = GAME_PHASE_TO_NUM[game_phase(board)]

    planes[19, :, :] = material_balance
    planes[20, :, :] = mobility
    planes[21, :, :] = 1.0 if board.is_check() else 0.0
    planes[22, :, :] = phase_num

    return planes


def move_type_flags(board: chess.Board, move: chess.Move) -> dict:
    """PASO 18. `board` debe estar en el estado ANTES de jugar `move`.
    Varios flags pueden ser True simultaneamente (p. ej. una captura que
    ademas da jaque)."""
    is_capture = board.is_capture(move)
    is_en_passant = board.is_en_passant(move)
    is_castling = board.is_castling(move)
    is_promotion = move.promotion is not None

    board.push(move)
    try:
        is_check = board.is_check()
        is_checkmate = board.is_checkmate()
    finally:
        board.pop()

    is_quiet = not (is_capture or is_en_passant or is_castling or is_promotion)

    return {
        "capture": is_capture,
        "check": is_check,
        "checkmate": is_checkmate,
        "promotion": is_promotion,
        "castling": is_castling,
        "en_passant": is_en_passant,
        "quiet": is_quiet,
    }


def position_flags(board: chess.Board) -> dict:
    """PASO 17. Estado de ESTA posicion (antes de cualquier movimiento)."""
    return {
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
    }


NO_MOVE_TYPE_FLAGS = {
    "capture": False, "check": False, "checkmate": False, "promotion": False,
    "castling": False, "en_passant": False, "quiet": False,
}
