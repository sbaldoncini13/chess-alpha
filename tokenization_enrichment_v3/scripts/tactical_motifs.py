"""#15 -- motivos tacticos con evidencia CLARA y geometrica (no se generan
labels ruidosos solo para cubrir la lista). Consolida pins/skewers/
overloaded (calculados en sus propios modulos, se reciben como parametro,
NUNCA se recalculan aca) + fork/discovered/double_check/back_rank
(computados localmente, definiciones abajo).

  fork(piece)            := la pieza ataca >=2 piezas enemigas (cualquier
                             tipo, incluye rey) SIMULTANEAMENTE. Puramente
                             geometrico -- no evalua si el fork "funciona"
                             (eso requeriria profundidad >1, fuera de
                             alcance de esta familia).
  double_attack           := igual que fork pero sin distinguir tipo de
                             pieza atacante (fork suele reservarse para
                             caballo; double_attack es el concepto general).
  discovered_attack_avail := una pieza propia bloquea la linea entre un
                             deslizante propio y una pieza enemiga, Y esa
                             pieza bloqueadora tiene al menos una jugada
                             legal (puede quitarse de la linea).
  discovered_check_avail  := igual, pero la pieza detras del bloqueador es
                             el REY enemigo.
  double_check             := board.checkers() tiene 2+ elementos AHORA
                             MISMO (ya ocurrio, no es "disponible").
  back_rank_weakness       := rey en su fila inicial (0 blancas / 7 negras)
                             Y TODAS las casillas delanteras existentes
                             (hasta 3) estan bloqueadas por piezas PROPIAS
                             (sin escape hacia adelante).
"""
import chess

from geometry import ray_pieces, slider_directions, sliders_of


def _attacks_count_multi(board: chess.Board, color: bool, piece_types) -> int:
    count = 0
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece.color != color or piece.piece_type not in piece_types:
            continue
        targets = board.attacks(sq)
        enemy_targets = [t for t in targets if board.piece_at(t) is not None and board.piece_at(t).color != color]
        if len(enemy_targets) >= 2:
            count += 1
    return count


def _discovered_availability(board: chess.Board, color: bool) -> dict:
    discovered_attack = 0
    discovered_check = 0
    enemy_king = board.king(not color)
    for sq, ptype in sliders_of(board, color):
        for d in slider_directions(ptype):
            hits = ray_pieces(board, sq, d, max_pieces=2)
            if len(hits) < 2:
                continue
            (sq1, p1), (sq2, p2) = hits
            if p1.color != color or p2.color == color:
                continue  # bloqueador debe ser propio, objetivo debe ser enemigo
            if not board.piece_at(sq1):
                continue
            has_legal_move = any(m.from_square == sq1 for m in board.legal_moves)
            if has_legal_move:
                discovered_attack += 1
                if sq2 == enemy_king:
                    discovered_check += 1
    return {"discovered_attack_available": discovered_attack, "discovered_check_available": discovered_check}


def _back_rank_weakness(board: chess.Board, color: bool) -> bool:
    king_sq = board.king(color)
    if king_sq is None:
        return False
    r, f = divmod(king_sq, 8)
    home_rank = 0 if color == chess.WHITE else 7
    if r != home_rank:
        return False
    front_rank = r + (1 if color == chess.WHITE else -1)
    if not (0 <= front_rank <= 7):
        return False
    front_files = [g for g in (f - 1, f, f + 1) if 0 <= g <= 7]
    blocked = 0
    for g in front_files:
        piece = board.piece_at(front_rank * 8 + g)
        if piece is not None and piece.color == color:
            blocked += 1
    return blocked == len(front_files)


def compute_tactical_motifs_side(board: chess.Board, color: bool, pins_skewers_row: dict, overloaded_row: dict) -> dict:
    name = "white" if color == chess.WHITE else "black"
    fork_count = _attacks_count_multi(board, color, {chess.KNIGHT})
    double_attack_count = _attacks_count_multi(board, color, {chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING})
    disc = _discovered_availability(board, color)

    # pins/skewers/overloaded ya vienen calculados por color = quien los
    # SUFRE (ver pins_skewers.py/overloaded.py) -- se leen directo con el
    # mismo `name`, sin invertir el color.
    return {
        "fork_count": fork_count,
        "double_attack_count": double_attack_count,
        "discovered_attack_available": disc["discovered_attack_available"],
        "discovered_check_available": disc["discovered_check_available"],
        "pin_present": pins_skewers_row.get(f"{name}_absolute_pins", 0) > 0,
        "skewer_present": pins_skewers_row.get(f"{name}_skewers_against", 0) > 0,
        "overloaded_defender_present": overloaded_row.get(f"{name}_overloaded_piece_count", 0) > 0,
        "back_rank_weakness": _back_rank_weakness(board, color),
    }


def compute_tactical_motifs_both(board: chess.Board, pins_skewers_row: dict, overloaded_row: dict) -> dict:
    out = {"double_check": len(board.checkers()) >= 2}
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        side = compute_tactical_motifs_side(board, color, pins_skewers_row, overloaded_row)
        for k, v in side.items():
            out[f"{name}_{k}"] = v
    return out
