import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_score(board: chess.Board, color: bool) -> int:
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
    return score


def game_phase(board: chess.Board) -> str:
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    total_material = material_score(board, chess.WHITE) + material_score(board, chess.BLACK)

    if total_material <= 20 or queens == 0:
        return "endgame"
    elif total_material <= 45:
        return "middlegame"
    else:
        return "opening"


def position_hash(board: chess.Board) -> str:
    """Hash canónico de la posición (tablero + turno), independiente del
    camino de jugadas que llevó a ella. No incluye enroques/al paso a
    propósito: se prioriza agrupar posiciones "equivalentes" para el
    reporte de transposiciones."""
    return board.board_fen() + " " + ("w" if board.turn == chess.WHITE else "b")


def canonical_fen(board: chess.Board) -> str:
    """FEN completo (incluye enroques y al paso), para reproducibilidad exacta."""
    return board.fen()


def basic_features(board: chess.Board) -> dict:
    white_material = material_score(board, chess.WHITE)
    black_material = material_score(board, chess.BLACK)

    return {
        "white_material": white_material,
        "black_material": black_material,
        "material_balance": white_material - black_material,
        "legal_moves_count": board.legal_moves.count(),
        "is_check": board.is_check(),
        "game_phase": game_phase(board),
        "position_hash": position_hash(board),
    }
