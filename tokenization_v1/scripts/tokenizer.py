"""Esquema compacto de identidad de movimiento: from_square + to_square + promotion.

action_id = (promo_code << 12) | (from_square << 6) | to_square

- from_square, to_square: 0-63 (6 bits cada uno)
- promo_code: 0=sin promocion, 1=caballo, 2=alfil, 3=torre, 4=dama (3 bits)
- Espacio total: 5 * 64 * 64 = 20480 valores -> cabe en uint16.

No representa material ni valores de pieza. El resto de la informacion
(captura, jaque, tipo de pieza movida) se deriva del tablero, no de este id.
"""
import chess

PROMO_TO_CODE = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
CODE_TO_PROMO = {v: k for k, v in PROMO_TO_CODE.items()}

ACTION_ID_MIN = 0
ACTION_ID_MAX = 5 * 64 * 64 - 1  # 20479


def move_to_action_id(move: chess.Move) -> int:
    promo_code = PROMO_TO_CODE[move.promotion]
    return (promo_code << 12) | (move.from_square << 6) | move.to_square


def action_id_to_move(action_id: int) -> chess.Move:
    if not (ACTION_ID_MIN <= action_id <= ACTION_ID_MAX):
        raise ValueError(
            f"action_id fuera de rango [{ACTION_ID_MIN}, {ACTION_ID_MAX}]: {action_id}"
        )
    promo_code = action_id >> 12
    rem = action_id & 0xFFF
    from_square = rem >> 6
    to_square = rem & 0x3F
    if promo_code not in CODE_TO_PROMO:
        raise ValueError(f"promo_code fuera de rango [0, 4]: {promo_code}")
    promotion = CODE_TO_PROMO[promo_code]
    return chess.Move(from_square=from_square, to_square=to_square, promotion=promotion)


def promo_code_to_piece(promo_code: int):
    if promo_code not in CODE_TO_PROMO:
        raise ValueError(f"promo_code fuera de rango [0, 4]: {promo_code}")
    return CODE_TO_PROMO[promo_code]
