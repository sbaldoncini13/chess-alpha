"""#9 -- casillas debiles, holes, outposts. Heuristica EXPLICITA y
conservadora (no afirma conocimiento perfecto -- ver encargo):

  weak_for(square, color) := ningun peon de `color` en un file adyacente
    esta en una fila igual-o-detras (segun la direccion de avance de
    `color`) de `square` -- es decir, `color` ya no puede volver a
    disputar esa casilla con un peon, sin importar como continue la
    partida desde aca.

  hole(square, color)     := weak_for(square, color) Y la casilla esta en
    el propio campo de `color` (ranks 0-3 para blancas, 4-7 para negras) Y
    esta vacia.

  knight_outpost(square, occupier) := weak_for(square, enemigo(occupier))
    Y defendida ahora mismo por un peon propio de `occupier` Y en
    territorio enemigo relativo (rank relativo >= 4).

  occupied_outpost  := outpost ocupado ahora mismo por un caballo propio.
  protected_outpost := occupied_outpost (por construccion ya esta
    defendido por un peon -- se reporta igual como campo separado por
    pedido explicito del encargo, documentado que es la misma condicion).
"""
import chess

_DIRECTION = {chess.WHITE: 1, chess.BLACK: -1}


def _pawn_files(board, color):
    files = {f: [] for f in range(8)}
    for sq in board.pieces(chess.PAWN, color):
        files[sq % 8].append(sq // 8)
    return files


def weak_squares_for(board: chess.Board, color: bool) -> set:
    direction = _DIRECTION[color]
    files = _pawn_files(board, color)
    weak = set()
    for sq in chess.SQUARES:
        r, f = divmod(sq, 8)
        adjacent_files = [g for g in (f - 1, f + 1) if 0 <= g <= 7]
        can_be_defended = any(
            (pr * direction) <= (r * direction) for g in adjacent_files for pr in files[g]
        )
        if not can_be_defended:
            weak.add(sq)
    return weak


def compute_outposts_side(board: chess.Board, occupier: bool, weak_for_enemy: set) -> dict:
    enemy = not occupier
    own_camp_ranks = {0, 1, 2, 3} if occupier == chess.WHITE else {4, 5, 6, 7}

    holes = 0
    weak_for_own = weak_squares_for(board, occupier)
    for sq in weak_for_own:
        r = sq // 8
        if r in own_camp_ranks and board.piece_at(sq) is None:
            holes += 1

    outpost_squares = []
    for sq in weak_for_enemy:
        r, f = divmod(sq, 8)
        relative_rank = r if occupier == chess.WHITE else 7 - r
        if relative_rank < 4:
            continue
        defenders = board.attackers(occupier, sq)
        defended_by_pawn = any(
            board.piece_at(d) is not None and board.piece_at(d).piece_type == chess.PAWN for d in defenders
        )
        if defended_by_pawn:
            outpost_squares.append(sq)

    occupied_outposts = sum(
        1 for sq in outpost_squares
        if board.piece_at(sq) is not None and board.piece_at(sq).color == occupier
        and board.piece_at(sq).piece_type == chess.KNIGHT
    )

    return {
        "holes": holes, "outpost_squares_available": len(outpost_squares),
        "occupied_knight_outposts": occupied_outposts, "protected_knight_outposts": occupied_outposts,
    }


def compute_outposts_both(board: chess.Board) -> dict:
    weak_for_black = weak_squares_for(board, chess.BLACK)  # outposts DISPONIBLES para blancas
    weak_for_white = weak_squares_for(board, chess.WHITE)  # outposts DISPONIBLES para negras
    w = compute_outposts_side(board, chess.WHITE, weak_for_black)
    b = compute_outposts_side(board, chess.BLACK, weak_for_white)
    out = {}
    for k, v in w.items():
        out[f"white_{k}"] = v
    for k, v in b.items():
        out[f"black_{k}"] = v
    return out
