"""#13 -- Static Exchange Evaluation. Implementacion del algoritmo estandar
"swap list" (Letouzey / chessprogrammingwiki "SEE - The Swap Algorithm"),
en unidades de material (P=1,N=3,B=3,R=5,Q=9), NO en centipawns de motor
-- estas unidades son deterministas/geometricas, no vienen de Stockfish.

Simplificacion documentada: en una captura CON promocion se usa el valor
de la pieza que efectivamente esta en from_square antes del movimiento
(el peon), no el valor de la pieza promovida -- subestima el SEE real de
capturas-con-promocion a dama. Conservador (nunca sobreestima).

No modifica el tablero real: siempre trabaja sobre board.copy(stack=False).
"""
import chess

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def static_exchange_eval(board: chess.Board, move: chess.Move) -> int:
    to_sq = move.to_square
    from_sq = move.from_square
    working = board.copy(stack=False)

    moving_piece = working.piece_at(from_sq)
    is_ep = working.is_en_passant(move)
    if is_ep:
        target_value = PIECE_VALUES[chess.PAWN]
    else:
        target_piece = working.piece_at(to_sq)
        target_value = PIECE_VALUES[target_piece.piece_type] if target_piece is not None else 0

    gain = [target_value]
    current_attacker_value = PIECE_VALUES[moving_piece.piece_type]

    if is_ep:
        ep_sq = to_sq + (-8 if moving_piece.color == chess.WHITE else 8)
        working.remove_piece_at(ep_sq)
    working.remove_piece_at(from_sq)
    working.set_piece_at(to_sq, moving_piece)

    side = not moving_piece.color
    d = 0
    while True:
        attackers = working.attackers(side, to_sq)
        if not attackers:
            break
        d += 1
        gain.append(current_attacker_value - gain[d - 1])
        least_sq = min(attackers, key=lambda s: PIECE_VALUES[working.piece_at(s).piece_type])
        piece = working.piece_at(least_sq)
        current_attacker_value = PIECE_VALUES[piece.piece_type]
        working.remove_piece_at(to_sq)
        working.remove_piece_at(least_sq)
        working.set_piece_at(to_sq, piece)
        side = not side

    while d > 0:
        gain[d - 1] = -max(-gain[d - 1], gain[d])
        d -= 1

    return gain[0]


def compute_see_summary(board: chess.Board) -> dict:
    """Sobre TODAS las capturas legales desde `board` (side to move)."""
    see_scores = []
    for move in board.legal_moves:
        if board.is_capture(move):
            see_scores.append((move, static_exchange_eval(board, move)))

    if not see_scores:
        return {
            "best_see": None, "worst_legal_capture_see": None,
            "positive_see_capture_count": 0, "negative_see_capture_count": 0,
            "zero_see_capture_count": 0, "legal_capture_count": 0,
        }

    values = [v for _, v in see_scores]
    return {
        "best_see": max(values), "worst_legal_capture_see": min(values),
        "positive_see_capture_count": sum(1 for v in values if v > 0),
        "negative_see_capture_count": sum(1 for v in values if v < 0),
        "zero_see_capture_count": sum(1 for v in values if v == 0),
        "legal_capture_count": len(values),
    }


def compute_see_moves(board: chess.Board, move_to_action_id) -> list:
    """Filas para see_moves.parquet: (move_id, see_material_units) por cada
    captura legal. `move_to_action_id` se inyecta (tokenizer.py de
    tokenization_v1, solo lectura) para no duplicar el vocabulario."""
    rows = []
    for move in board.legal_moves:
        if board.is_capture(move):
            rows.append({"move_id": move_to_action_id(move), "see_material_units": static_exchange_eval(board, move)})
    return rows
