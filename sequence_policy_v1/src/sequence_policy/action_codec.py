"""Codec de action_id <-> chess.Move. Identico en definicion al esquema de
tokenization_v1 (from_square<<6 | to_square, con promo_code<<12) -- ver
seccion 1/4 del pedido. Se duplica aca (en vez de importar entre proyectos)
a proposito, para que sequence_policy_v1 no dependa en tiempo de ejecucion
de tokenization_v1 y quede completamente aislado.
"""
import chess

from sequence_policy.constants import ACTION_ID_MAX, ACTION_ID_MIN

PROMO_TO_CODE = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
CODE_TO_PROMO = {v: k for k, v in PROMO_TO_CODE.items()}

CASTLING_UCI = ("e1g1", "e1c1", "e8g8", "e8c8")


def move_to_action_id(move: chess.Move) -> int:
    promo_code = PROMO_TO_CODE[move.promotion]
    return (promo_code << 12) | (move.from_square << 6) | move.to_square


def action_id_to_move(action_id: int) -> chess.Move:
    if not (ACTION_ID_MIN <= action_id <= ACTION_ID_MAX):
        raise ValueError(f"action_id fuera de rango [{ACTION_ID_MIN}, {ACTION_ID_MAX}]: {action_id}")
    promo_code = action_id >> 12
    rem = action_id & 0xFFF
    from_square = rem >> 6
    to_square = rem & 0x3F
    if promo_code not in CODE_TO_PROMO:
        raise ValueError(f"promo_code fuera de rango [0, 4]: {promo_code}")
    return chess.Move(from_square=from_square, to_square=to_square, promotion=CODE_TO_PROMO[promo_code])


def castling_action_ids() -> set:
    """Los 4 action_id que representan un enroque (rey moviendose 2 casillas).
    Se calculan a mano una sola vez con chess.Move.from_uci, no dependen del
    estado del tablero -- from/to identifican el enroque de forma unica."""
    return {move_to_action_id(chess.Move.from_uci(u)) for u in CASTLING_UCI}


def promo_code_of(action_id: int) -> int:
    return action_id >> 12
