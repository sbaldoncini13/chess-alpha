"""#6, #7, #8 -- PRIORIDAD ALTA. Estructura de peones, peones pasados,
mayorias. Definiciones exactas documentadas aca (ver tambien
docs/FEATURE_DEFINITIONS.md):

  isolated(pawn)  := sin peon propio en los files adyacentes (cualquier fila).
  doubled_files   := cantidad de files con EXACTAMENTE 2 peones propios.
  tripled_files   := cantidad de files con 3+ peones propios.
  backward(pawn)  := (a) sin peon propio en file adyacente en la misma fila
                     o mas atras (no puede recibir apoyo lateral), Y
                     (b) la casilla de avance esta atacada por un peon
                     enemigo. Definicion estandar (chessprogrammingwiki),
                     deliberadamente conservadora.
  connected(pawn) := defendido por OTRO peon propio (ataque diagonal de peon).
                     "protected_pawns" es EL MISMO concepto -- se reportan
                     ambos nombres por pedido explicito pero es una sola
                     computacion, documentado para no fingir una distincion
                     que no existe en la teoria de ajedrez estandar.
  blocked(pawn)   := la casilla directamente delante esta ocupada (por
                     cualquier color) -- no puede avanzar un paso (si podria
                     capturar en diagonal).
  pawn_islands    := cantidad de grupos de files consecutivos con al menos
                     un peon propio, separados por >=1 file totalmente vacio
                     de peones propios.

  passed(pawn)    := ningun peon ENEMIGO en el mismo file o files adyacentes,
                     en ninguna fila por delante de este peon (hacia la
                     promocion).
  connected_passed:= pasado Y con un peon propio pasado en file adyacente a
                     distancia de fila <=1.
  protected_passed:= pasado Y defendido por otro peon propio.
  relative_rank   := 0..7, 7 = a un paso de promocionar (ya normalizado por
                     color: para negras se invierte la fila).

  majority(side)  := cantidad de peones propios en ese sector > cantidad de
                     peones enemigos en el mismo sector (estrictamente mayor).
                     kingside = files e-h (4-7), queenside = files a-d (0-3),
                     center = files c-f (2-5).
"""
import chess

KINGSIDE_FILES = {4, 5, 6, 7}
QUEENSIDE_FILES = {0, 1, 2, 3}
CENTER_FILES = {2, 3, 4, 5}


def _pawn_files(board: chess.Board, color: bool) -> dict:
    files = {f: [] for f in range(8)}
    for sq in board.pieces(chess.PAWN, color):
        _, f = divmod(sq, 8)
        files[f].append(sq)
    return files


def compute_pawn_structure_side(board: chess.Board, color: bool) -> dict:
    enemy = not color
    own_files = _pawn_files(board, color)
    enemy_files = _pawn_files(board, enemy)
    direction = 1 if color == chess.WHITE else -1

    isolated = 0
    doubled_files = sum(1 for f in range(8) if len(own_files[f]) == 2)
    tripled_files = sum(1 for f in range(8) if len(own_files[f]) >= 3)
    backward = 0
    connected = 0
    blocked = 0

    for f in range(8):
        for sq in own_files[f]:
            rank = sq // 8
            adjacent_files = [g for g in (f - 1, f + 1) if 0 <= g <= 7]

            if not any(own_files[g] for g in adjacent_files):
                isolated += 1

            has_lateral_support = any(
                any((s // 8) * direction <= rank * direction for s in own_files[g]) for g in adjacent_files
            )
            stop_square = sq + 8 * direction
            stop_attacked_by_enemy_pawn = False
            if 0 <= stop_square <= 63:
                stop_attacked_by_enemy_pawn = any(
                    board.piece_at(a) is not None and board.piece_at(a).piece_type == chess.PAWN
                    and board.piece_at(a).color == enemy
                    for a in board.attackers(enemy, stop_square)
                )
            if not has_lateral_support and stop_attacked_by_enemy_pawn:
                backward += 1

            defenders = board.attackers(color, sq)
            if any(board.piece_at(d) is not None and board.piece_at(d).piece_type == chess.PAWN for d in defenders):
                connected += 1

            if 0 <= stop_square <= 63 and board.piece_at(stop_square) is not None:
                blocked += 1

    occupied_files = sorted(f for f in range(8) if own_files[f])
    islands = 0
    prev = None
    for f in occupied_files:
        if prev is None or f - prev > 1:
            islands += 1
        prev = f

    # peones pasados
    passed_squares = []
    max_relative_rank = -1
    for f in range(8):
        for sq in own_files[f]:
            rank = sq // 8
            adjacent_and_own = [g for g in (f - 1, f, f + 1) if 0 <= g <= 7]
            ahead_enemy = any(
                (r // 8 - rank) * direction > 0
                for g in adjacent_and_own for r in enemy_files[g]
            )
            if not ahead_enemy:
                passed_squares.append(sq)
                relative_rank = rank if color == chess.WHITE else 7 - rank
                max_relative_rank = max(max_relative_rank, relative_rank)

    connected_passed = 0
    protected_passed = 0
    passed_set = set(passed_squares)
    for sq in passed_squares:
        rank = sq // 8
        f = sq % 8
        adjacent_files = [g for g in (f - 1, f + 1) if 0 <= g <= 7]
        if any(abs((p // 8) - rank) <= 1 and (p % 8) in adjacent_files for p in passed_set):
            connected_passed += 1
        defenders = board.attackers(color, sq)
        if any(board.piece_at(d) is not None and board.piece_at(d).piece_type == chess.PAWN for d in defenders):
            protected_passed += 1

    # mayorias
    def sector_majority(files_set):
        own_n = sum(len(own_files[f]) for f in files_set)
        enemy_n = sum(len(enemy_files[f]) for f in files_set)
        return own_n > enemy_n

    return {
        "isolated_pawns": isolated, "doubled_pawn_files": doubled_files, "tripled_pawn_files": tripled_files,
        "backward_pawns": backward, "connected_pawns": connected, "protected_pawns": connected,
        "blocked_pawns": blocked, "pawn_islands": islands,
        "passed_pawns": len(passed_squares), "connected_passed_pawns": connected_passed,
        "protected_passed_pawns": protected_passed,
        "max_passed_pawn_relative_rank": max_relative_rank,
        "kingside_pawn_majority": sector_majority(KINGSIDE_FILES),
        "queenside_pawn_majority": sector_majority(QUEENSIDE_FILES),
        "center_pawn_majority": sector_majority(CENTER_FILES),
    }


def compute_pawn_structure_both(board: chess.Board) -> dict:
    w = compute_pawn_structure_side(board, chess.WHITE)
    b = compute_pawn_structure_side(board, chess.BLACK)
    out = {}
    for k, v in w.items():
        out[f"white_{k}"] = v
    for k, v in b.items():
        out[f"black_{k}"] = v
    return out
