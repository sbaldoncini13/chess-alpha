"""Evaluacion estatica MINIMA de search_v1 - deliberadamente simple.

Alcance de Fase I (explicito en el pedido): solo balance material,
determinista, sin redes neuronales, sin tablas pieza-casilla, sin
movilidad/estructura de peones/seguridad del rey, sin tuning. El objetivo
es tener un score utilizable para probar el BUSCADOR mas adelante (Fase
II en adelante), no una evaluacion fuerte.

Convencion de perspectiva (documentada explicitamente, item obligatorio
del pedido): `evaluate_static_white_pov()` devuelve el balance en
centipawns desde la perspectiva de BLANCAS SIEMPRE (positivo = mejor para
blancas), independiente de quien mueve. `evaluate_static_relative()`
devuelve el mismo valor desde la perspectiva del lado que mueve (side to
move) - convencion que necesitara Negamax en una fase futura (Negamax
exige evaluacion relativa al que mueve). Las dos conviven aca porque el
pedido de Fase I exige "perspectiva documentada" + "resultado simetrico al
cambiar la perspectiva", y la forma mas directa de probar simetria es
comparar white_pov de una posicion contra white_pov de su espejo de
color.

Terminales NO se manejan aca (ver terminal.py) - evaluate_static_white_pov
calcula material igual en una posicion de jaque mate que en cualquier
otra; quien llama debe consultar detect_terminal() primero si terminales
importan para su caso de uso."""

from __future__ import annotations

import chess

from search_v1.position import Position

PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def evaluate_static_white_pov(position: Position) -> int:
    """Balance material en centipawns, positivo = ventaja de blancas.
    Determinista: misma posicion -> mismo valor, siempre."""
    board = position.board
    total = 0
    for piece_type, value in PIECE_VALUES_CP.items():
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total -= value * len(board.pieces(piece_type, chess.BLACK))
    return total


def evaluate_static_relative(position: Position) -> int:
    """Mismo balance, pero desde la perspectiva del lado que mueve
    (positivo = ventaja para side_to_move) - convencion Negamax, no usada
    todavia por ningun buscador en esta fase, expuesta para Fase II."""
    white_pov = evaluate_static_white_pov(position)
    return white_pov if position.side_to_move == "white" else -white_pov
