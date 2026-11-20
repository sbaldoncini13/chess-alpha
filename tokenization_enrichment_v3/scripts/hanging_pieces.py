"""#3 -- piezas atacadas/defendidas/colgadas. Definicion CONSERVADORA y
explicita (documentada tambien en docs/FEATURE_DEFINITIONS.md):

  attacked(piece en casilla S, color C)  := attackers_count[opponent(C)][S] > 0
  defended(piece en casilla S, color C)  := attackers_count[C][S] > 0
  hanging(piece)                          := attacked AND NOT defended

Esta es la definicion MAS SIMPLE y menos ambigua posible: "colgada" =
atacada y con CERO defensores propios. Deliberadamente NO intenta capturar
el caso "defendida pero perdiendo el cambio" (una torre defendida una sola
vez pero atacada por una dama y un peon, p.ej.) -- ese analisis de
intercambios pertenece a SEE (#13, see.py), no a esta familia. Mezclar
ambos conceptos bajo el mismo nombre "hanging" seria exactamente el tipo
de definicion ambigua que el encargo pide evitar.

Usa las mismas piece values que tokenization_v1/scripts/position_state.py
(P=1,N=3,B=3,R=5,Q=9), solo lectura, no importado.
"""
import chess
import numpy as np

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def compute_hanging_pieces(board: chess.Board, attackers_count_white: np.ndarray, attackers_count_black: np.ndarray) -> dict:
    counts = {
        "pieces_attacked_white": 0, "pieces_attacked_black": 0,
        "pieces_defended_white": 0, "pieces_defended_black": 0,
        "pieces_undefended_white": 0, "pieces_undefended_black": 0,
        "hanging_pieces_white": 0, "hanging_pieces_black": 0,
        "hanging_material_white": 0, "hanging_material_black": 0,
    }
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        is_white = piece.color == chess.WHITE
        attacked_by = attackers_count_black[square] if is_white else attackers_count_white[square]
        defended_by = attackers_count_white[square] if is_white else attackers_count_black[square]
        side = "white" if is_white else "black"

        if attacked_by > 0:
            counts[f"pieces_attacked_{side}"] += 1
        if defended_by > 0:
            counts[f"pieces_defended_{side}"] += 1
        else:
            counts[f"pieces_undefended_{side}"] += 1
        if attacked_by > 0 and defended_by == 0:
            counts[f"hanging_pieces_{side}"] += 1
            counts[f"hanging_material_{side}"] += PIECE_VALUES[piece.piece_type]

    return counts
