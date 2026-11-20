"""#4 -- movilidad global y por tipo de pieza. Movilidad = cantidad de
movimientos LEGALES (no pseudo-legales) del lado AL TURNO -- python-chess
board.legal_moves ya filtra jaques propios. Esta familia es metadata
adicional, NO sustituye la legal_move_mask ya persistida en el Golden
(esa sigue siendo la fuente de verdad para el vocabulario de acciones).

Solo se puede calcular movilidad completa del lado que NO esta al turno
contando sus pseudo-legales (python-chess no expone "legal moves si fuera
su turno" directamente sin mutar el tablero) -- se resuelve pasando el
turno con board.turn = not board.turn temporalmente y restaurandolo,
documentado explicitamente aca para que no sea una sorpresa.
"""
import chess

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_NAMES = {chess.PAWN: "pawns", chess.KNIGHT: "knights", chess.BISHOP: "bishops",
               chess.ROOK: "rooks", chess.QUEEN: "queens", chess.KING: "king"}


def _mobility_by_type_for_side_to_move(board: chess.Board) -> dict:
    counts = {name: 0 for name in PIECE_NAMES.values()}
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        counts[PIECE_NAMES[piece.piece_type]] += 1
    return counts


def compute_mobility(board: chess.Board) -> dict:
    stm_white = board.turn == chess.WHITE
    stm_counts = _mobility_by_type_for_side_to_move(board)
    total_stm = sum(stm_counts.values())

    # movilidad del lado que NO esta al turno: se calcula pasandole el
    # turno temporalmente (sin mutar la posicion real, solo el flag de
    # turno) -- valido siempre que la posicion no este en jaque para ese
    # lado de forma que lo deje "en jaque perpetuo ajeno" (edge case raro,
    # documentado: si el lado que no mueve estuviera en jaque en una
    # posicion legal real esto no puede pasar, python-chess ya lo garantiza).
    board.turn = not board.turn
    saved_ep = board.ep_square
    board.ep_square = None  # el derecho de al paso no es aplicable "si fuera el turno del otro lado"
    try:
        other_counts = _mobility_by_type_for_side_to_move(board)
    finally:
        board.turn = not board.turn
        board.ep_square = saved_ep
    total_other = sum(other_counts.values())

    white_counts = stm_counts if stm_white else other_counts
    black_counts = other_counts if stm_white else stm_counts
    mobility_white = total_stm if stm_white else total_other
    mobility_black = total_other if stm_white else total_stm

    result = {
        "total_legal_moves_side_to_move": total_stm,
        "mobility_white": mobility_white, "mobility_black": mobility_black,
        "mobility_diff": mobility_white - mobility_black,
    }
    for pt in PIECE_TYPES:
        name = PIECE_NAMES[pt]
        result[f"mobility_white_{name}"] = white_counts[name]
        result[f"mobility_black_{name}"] = black_counts[name]
    return result
