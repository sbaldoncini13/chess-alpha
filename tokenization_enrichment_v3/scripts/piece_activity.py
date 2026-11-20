"""#17 -- PRIORIDAD ALTA. Actividad y calidad de piezas (torres, alfiles,
caballos, dama). Reutiliza attackers_count ya calculado por attack_maps.py
(no vuelve a llamar board.attackers para las mismas 64 casillas). La
movilidad AGREGADA por tipo de pieza ya vive en mobility.py -- esta familia
agrega heuristicas de CALIDAD posicional especificas por pieza, no
redefine movilidad.

Definiciones (ver docs/FEATURE_DEFINITIONS.md):
  rook_open_file / rook_semi_open_file: igual definicion de file que
    pawn_structure.py, evaluada en el file de la torre.
  rook_on_7th: torre en la fila relativa 7 (0-indexada rank==6 para
    blancas, rank==1 para negras -- la "7a fila" clasica).
  connected_rooks: las dos torres propias en la misma fila/columna sin
    ninguna pieza entre medio (se defienden mutuamente).
  bishop_pair: exactamente 2 alfiles propios en el tablero.
  bad_bishop: alfil con >= config threshold peones PROPIOS en casillas del
    mismo color que el alfil (heuristica clasica, conservadora).
  knight_centralized: caballo en el centro extendido (files/ranks 2..5).
  knight_on_rim: caballo en file a o h.
"""
import chess

CONFIG_DEFAULTS = {"bad_bishop_min_same_color_pawns": 4}


def _file_has_pawn(board, file_idx, color):
    for r in range(8):
        p = board.piece_at(r * 8 + file_idx)
        if p is not None and p.piece_type == chess.PAWN and p.color == color:
            return True
    return False


def _square_color_is_light(square: int) -> bool:
    r, f = divmod(square, 8)
    return (r + f) % 2 == 1  # convencion python-chess: a1 (sq=0) es oscura


def _rooks_activity(board: chess.Board, color: bool) -> dict:
    enemy = not color
    open_file = 0
    semi_open = 0
    seventh = 0
    rooks = list(board.pieces(chess.ROOK, color))
    for sq in rooks:
        r, f = divmod(sq, 8)
        own_pawn = _file_has_pawn(board, f, color)
        enemy_pawn = _file_has_pawn(board, f, enemy)
        if not own_pawn and not enemy_pawn:
            open_file += 1
        elif not own_pawn and enemy_pawn:
            semi_open += 1
        relative_rank = r if color == chess.WHITE else 7 - r
        if relative_rank == 6:
            seventh += 1

    connected = False
    if len(rooks) == 2:
        a, b = rooks
        ra, fa = divmod(a, 8)
        rb, fb = divmod(b, 8)
        if ra == rb or fa == fb:
            between = chess.SquareSet(chess.between(a, b))
            connected = not any(board.piece_at(s) is not None for s in between)

    return {
        "rook_open_file_count": open_file, "rook_semi_open_file_count": semi_open,
        "rook_on_7th_count": seventh, "connected_rooks": connected,
    }


def _bishops_activity(board: chess.Board, color: bool, config: dict) -> dict:
    bishops = list(board.pieces(chess.BISHOP, color))
    own_pawns = list(board.pieces(chess.PAWN, color))
    threshold = config.get("bad_bishop_min_same_color_pawns", CONFIG_DEFAULTS["bad_bishop_min_same_color_pawns"])

    bad_bishops = 0
    total_diag_mobility = 0
    for sq in bishops:
        is_light = _square_color_is_light(sq)
        same_color_pawns = sum(1 for p in own_pawns if _square_color_is_light(p) == is_light)
        if same_color_pawns >= threshold:
            bad_bishops += 1
        total_diag_mobility += len(board.attacks(sq))

    return {
        "bishop_pair": len(bishops) == 2, "bad_bishop_count": bad_bishops,
        "bishop_diagonal_mobility_total": total_diag_mobility,
    }


def _knights_activity(board: chess.Board, color: bool) -> dict:
    knights = list(board.pieces(chess.KNIGHT, color))
    centralized = 0
    on_rim = 0
    for sq in knights:
        r, f = divmod(sq, 8)
        if 2 <= r <= 5 and 2 <= f <= 5:
            centralized += 1
        if f == 0 or f == 7:
            on_rim += 1
    return {"knight_centralized_count": centralized, "knight_on_rim_count": on_rim}


def _queen_activity(board: chess.Board, color: bool, attackers_count_white, attackers_count_black) -> dict:
    queens = list(board.pieces(chess.QUEEN, color))
    if not queens:
        return {"queen_activity_mobility": 0, "queen_attacked": False}
    sq = queens[0]
    enemy_attackers = attackers_count_black if color == chess.WHITE else attackers_count_white
    return {"queen_activity_mobility": len(board.attacks(sq)), "queen_attacked": bool(enemy_attackers[sq] > 0)}


def compute_piece_activity_side(board: chess.Board, color: bool, attackers_count_white, attackers_count_black,
                                 config: dict) -> dict:
    out = {}
    out.update(_rooks_activity(board, color))
    out.update(_bishops_activity(board, color, config))
    out.update(_knights_activity(board, color))
    out.update(_queen_activity(board, color, attackers_count_white, attackers_count_black))
    return out


def compute_piece_activity_both(board: chess.Board, attackers_count_white, attackers_count_black, config: dict) -> dict:
    w = compute_piece_activity_side(board, chess.WHITE, attackers_count_white, attackers_count_black, config)
    b = compute_piece_activity_side(board, chess.BLACK, attackers_count_white, attackers_count_black, config)
    out = {}
    for k, v in w.items():
        out[f"white_{k}"] = v
    for k, v in b.items():
        out[f"black_{k}"] = v
    return out
