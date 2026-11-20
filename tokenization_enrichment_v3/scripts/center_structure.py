"""#18 -- estructura del centro. Heuristica CONSERVADORA basada
exclusivamente en los peones de los files d/e (mismo espiritu que el resto
de V3: no se pretende un modelo posicional perfecto).

  central_pawns(color) := peones de `color` en d4/d5/e4/e5 (las 4 casillas
                           centrales clasicas).
  blocked(pawn)         := la casilla justo delante esta ocupada (no puede
                           avanzar un paso).

Clasificacion (mutuamente excluyente, en este orden de prioridad):
  "open"      -- ningun peon en los files d/e (ninguno de los colores).
  "locked"    -- ambos colores tienen peones en d/e Y al menos un par
                 esta bloqueado mutuamente (peon propio bloqueado por peon
                 enemigo directamente enfrente).
  "closed"    -- ambos colores tienen peones en d/e pero sin bloqueo
                 mutuo directo detectado (mas cerrado que "mobile" porque
                 hay tension, menos estricto que "locked").
  "semi_open" -- solo un color tiene peones en d/e.
  "mobile"    -- ninguno de los casos anteriores aplica de forma clara
                 (fallback conservador).
"""
import chess

D_E_FILES = {3, 4}
CENTRAL_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}


def compute_center_structure(board: chess.Board) -> dict:
    white_de_pawns = [sq for sq in board.pieces(chess.PAWN, chess.WHITE) if (sq % 8) in D_E_FILES]
    black_de_pawns = [sq for sq in board.pieces(chess.PAWN, chess.BLACK) if (sq % 8) in D_E_FILES]

    central_white = sum(1 for sq in white_de_pawns if sq in CENTRAL_SQUARES)
    central_black = sum(1 for sq in black_de_pawns if sq in CENTRAL_SQUARES)

    def is_blocked(sq, color):
        direction = 1 if color == chess.WHITE else -1
        front = sq + 8 * direction
        return 0 <= front <= 63 and board.piece_at(front) is not None

    mutually_locked = False
    for wsq in white_de_pawns:
        front = wsq + 8
        if front in black_de_pawns and is_blocked(wsq, chess.WHITE):
            mutually_locked = True
            break

    if not white_de_pawns and not black_de_pawns:
        classification = "open"
    elif white_de_pawns and black_de_pawns:
        classification = "locked" if mutually_locked else "closed"
    elif white_de_pawns or black_de_pawns:
        classification = "semi_open"
    else:
        classification = "mobile"

    return {
        "center_classification": classification,
        "central_pawns_white": central_white, "central_pawns_black": central_black,
        "d_e_file_pawns_white": len(white_de_pawns), "d_e_file_pawns_black": len(black_de_pawns),
        "center_mutually_locked": mutually_locked,
    }
