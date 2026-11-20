"""#5 -- PRIORIDAD ALTA. Seguridad del rey. Formula documentada y
versionada aca mismo (ver tambien docs/FEATURE_DEFINITIONS.md); todos los
pesos viven en config (king_safety.exposure_score_weights), nunca
hardcodeados en silencio.

Definiciones exactas:
  king_ring        := las (hasta 8) casillas a distancia Chebyshev == 1 del
                       rey (NO incluye la casilla del rey -- eso ya es
                       is_check, familia separada).
  pawn_shield       := peones PROPIOS en las hasta 3 casillas directamente
                       delante del rey (mismo file-1..file+1, una fila hacia
                       el bando enemigo).
  open_file(f)      := ningun peon de ningun color en el file f.
  semi_open_file(f) := sin peon PROPIO en el file f, pero con al menos un
                       peon enemigo.
  queen/rook_proximity := distancia Chebyshev minima entre el rey y la
                       dama/torre enemiga mas cercana (None si no hay).
  king_exposure_score := combinacion lineal en [0,1] de los componentes de
                       arriba, normalizados; pesos en config.
"""
import chess
import numpy as np

FRONT_DIRECTION = {chess.WHITE: 1, chess.BLACK: -1}


def _chebyshev(sq_a: int, sq_b: int) -> int:
    ra, fa = divmod(sq_a, 8)
    rb, fb = divmod(sq_b, 8)
    return max(abs(ra - rb), abs(fa - fb))


def _file_has_pawn(board: chess.Board, file_idx: int, color: bool) -> bool:
    for r in range(8):
        piece = board.piece_at(r * 8 + file_idx)
        if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
            return True
    return False


def _king_ring(king_square: int, radius: int) -> list:
    kr, kf = divmod(king_square, 8)
    ring = []
    for r in range(max(0, kr - radius), min(7, kr + radius) + 1):
        for f in range(max(0, kf - radius), min(7, kf + radius) + 1):
            sq = r * 8 + f
            if sq != king_square:
                ring.append(sq)
    return ring


def compute_king_safety(board: chess.Board, color: bool, attackers_count_white: np.ndarray,
                         attackers_count_black: np.ndarray, config: dict) -> dict:
    king_square = board.king(color)
    if king_square is None:
        return None  # no deberia pasar en una posicion legal real; se documenta el caso defensivamente

    enemy = not color
    radius = config["king_ring_radius"]
    ring = _king_ring(king_square, radius)

    own_attackers = attackers_count_white if color == chess.WHITE else attackers_count_black
    enemy_attackers = attackers_count_black if color == chess.WHITE else attackers_count_white

    king_ring_attacked = sum(1 for sq in ring if enemy_attackers[sq] > 0)
    king_ring_defended = sum(1 for sq in ring if own_attackers[sq] > 0)

    kr, kf = divmod(king_square, 8)
    front_rank = kr + FRONT_DIRECTION[color]
    shield_files = [f for f in (kf - 1, kf, kf + 1) if 0 <= f <= 7]
    pawn_shield = 0
    if 0 <= front_rank <= 7:
        for f in shield_files:
            piece = board.piece_at(front_rank * 8 + f)
            if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                pawn_shield += 1

    open_files = 0
    semi_open_files = 0
    for f in shield_files:
        own_pawn = _file_has_pawn(board, f, color)
        enemy_pawn = _file_has_pawn(board, f, enemy)
        if not own_pawn and not enemy_pawn:
            open_files += 1
        elif not own_pawn and enemy_pawn:
            semi_open_files += 1

    enemy_attacker_squares = set()
    for sq in ring:
        enemy_attacker_squares |= set(board.attackers(enemy, sq))
    enemy_attackers_near_king = len(enemy_attacker_squares)

    enemy_queens = list(board.pieces(chess.QUEEN, enemy))
    enemy_rooks = list(board.pieces(chess.ROOK, enemy))
    queen_proximity = min((_chebyshev(king_square, sq) for sq in enemy_queens), default=None)
    rook_proximity = min((_chebyshev(king_square, sq) for sq in enemy_rooks), default=None)
    queen_near = queen_proximity is not None and queen_proximity <= config["queen_proximity_max_distance"]
    rook_near = rook_proximity is not None and rook_proximity <= config["rook_proximity_max_distance"]

    diagonal_pressure = 0
    ring_and_king = set(ring) | {king_square}
    for sq in list(board.pieces(chess.BISHOP, enemy)) + list(board.pieces(chess.QUEEN, enemy)):
        if ring_and_king & set(board.attacks(sq)):
            diagonal_pressure += 1

    w = config["exposure_score_weights"]
    n_ring = max(1, len(ring))
    n_shield_files = max(1, len(shield_files))
    score = (
        w["w_king_ring_attacked"] * (king_ring_attacked / n_ring)
        + w["w_pawn_shield_missing"] * (1.0 - pawn_shield / n_shield_files)
        + w["w_open_files"] * (open_files / n_shield_files)
        + w["w_semi_open_files"] * (semi_open_files / n_shield_files)
        + w["w_enemy_attackers_near"] * min(1.0, enemy_attackers_near_king / 4.0)
        + w["w_queen_near"] * float(queen_near)
        + w["w_rook_near"] * float(rook_near)
    )
    score = max(0.0, min(1.0, score))

    return {
        "king_ring_attacked_squares": king_ring_attacked, "king_ring_defended_squares": king_ring_defended,
        "pawn_shield_count": pawn_shield, "open_files_toward_king": open_files,
        "semi_open_files_toward_king": semi_open_files, "enemy_attackers_near_king": enemy_attackers_near_king,
        "queen_proximity": queen_proximity if queen_proximity is not None else -1,
        "rook_proximity": rook_proximity if rook_proximity is not None else -1,
        "queen_near_king": queen_near, "rook_near_king": rook_near,
        "bishop_queen_diagonal_pressure": diagonal_pressure,
        "king_exposure_score": score,
    }


def compute_king_safety_both(board: chess.Board, attackers_count_white, attackers_count_black, config: dict) -> dict:
    w = compute_king_safety(board, chess.WHITE, attackers_count_white, attackers_count_black, config)
    b = compute_king_safety(board, chess.BLACK, attackers_count_white, attackers_count_black, config)
    out = {}
    for k, v in (w or {}).items():
        out[f"white_{k}"] = v
    for k, v in (b or {}).items():
        out[f"black_{k}"] = v
    if w is not None and b is not None:
        out["king_safety_diff"] = b["king_exposure_score"] - w["king_exposure_score"]  # positivo = blancas mas seguras
    return out
