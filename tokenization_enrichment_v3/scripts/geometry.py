"""Utilidad geometrica compartida por pins_skewers.py, overloaded.py,
threats.py, tactical_motifs.py: escaneo de rayos (direcciones de torre/
alfil) desde una casilla, devolviendo las piezas encontradas EN ORDEN a lo
largo del rayo hasta el borde del tablero o la primera pieza que lo
bloquea totalmente.

Deliberadamente simple y auditable -- si un patron no puede determinarse
con este escaneo directo, no se afirma (ver principio "ser conservador"
del encargo). No usa magic bitboards ni tablas precalculadas: para 818k
posiciones un escaneo directo en Python ya es suficientemente rapido (ver
benchmark en el piloto).
"""
import chess

ROOK_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
BISHOP_DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
QUEEN_DIRECTIONS = ROOK_DIRECTIONS + BISHOP_DIRECTIONS

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def slider_directions(piece_type: int):
    if piece_type == chess.ROOK:
        return ROOK_DIRECTIONS
    if piece_type == chess.BISHOP:
        return BISHOP_DIRECTIONS
    if piece_type == chess.QUEEN:
        return QUEEN_DIRECTIONS
    return []


def ray_pieces(board: chess.Board, from_square: int, direction: tuple, max_pieces: int = 3) -> list:
    """Devuelve hasta `max_pieces` (square, piece) encontrados a lo largo
    del rayo desde from_square (SIN incluirla), en orden de distancia
    creciente, deteniendose antes de la 3ra pieza (rara vez hace falta
    mirar mas alla para pin/skewer de 2 piezas)."""
    dr, df = direction
    r, f = divmod(from_square, 8)
    found = []
    r, f = r + dr, f + df
    while 0 <= r <= 7 and 0 <= f <= 7 and len(found) < max_pieces:
        sq = r * 8 + f
        piece = board.piece_at(sq)
        if piece is not None:
            found.append((sq, piece))
        r, f = r + dr, f + df
    return found


def sliders_of(board: chess.Board, color: bool):
    """(square, piece_type) de todas las piezas deslizantes de `color`."""
    out = []
    for pt in (chess.ROOK, chess.BISHOP, chess.QUEEN):
        for sq in board.pieces(pt, color):
            out.append((sq, pt))
    return out
