"""Módulo heurístico (no entrenado) que arma la explicación en español pedida
originalmente: qué pasa en el tablero, plan probable del rival, amenazas,
jugadas candidatas, recomendación y motivo.

Es deliberadamente basado en reglas sobre python-chess, no en un modelo: no
existe un dataset etiquetado de "planes estratégicos" (ni Stockfish ni las
partidas humanas lo dan gratis), así que esto es una primera aproximación
razonable, pensada para iterarse con más heurísticas o, más adelante, un
clasificador entrenado sobre partidas anotadas.
"""

import chess

from chess_alpha.board_features import material_score, game_phase
from chess_alpha.search import best_move, material_eval

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
SIDE_NAME = {chess.WHITE: "blancas", chess.BLACK: "negras"}


def _center_control(board: chess.Board) -> dict:
    control = {chess.WHITE: 0, chess.BLACK: 0}
    for square in CENTER_SQUARES:
        piece = board.piece_at(square)
        if piece:
            control[piece.color] += 2
        control[chess.WHITE] += len(board.attackers(chess.WHITE, square))
        control[chess.BLACK] += len(board.attackers(chess.BLACK, square))
    return control


def _file_has_pawn(board: chess.Board, file_index: int, color: bool) -> bool:
    return any(
        board.piece_at(chess.square(file_index, rank)) == chess.Piece(chess.PAWN, color)
        for rank in range(8)
    )


def _open_and_half_open_files(board: chess.Board):
    """open: sin peones de ningún bando. half_open[color]: abierta para ese
    color porque no tiene peones propios ahí (aunque el rival sí los tenga)."""
    open_files, half_open = [], {chess.WHITE: [], chess.BLACK: []}
    for file_index in range(8):
        white_pawn = _file_has_pawn(board, file_index, chess.WHITE)
        black_pawn = _file_has_pawn(board, file_index, chess.BLACK)
        if not white_pawn and not black_pawn:
            open_files.append(file_index)
        else:
            if not white_pawn:
                half_open[chess.WHITE].append(file_index)
            if not black_pawn:
                half_open[chess.BLACK].append(file_index)
    return open_files, half_open


def _king_safety(board: chess.Board, color: bool) -> dict:
    king_sq = board.king(color)
    if king_sq is None:
        return {"shield_pawns": 0, "attackers_near_king": 0}

    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    shield_rank = king_rank + (1 if color == chess.WHITE else -1)

    shield_pawns = 0
    if 0 <= shield_rank <= 7:
        for f in (king_file - 1, king_file, king_file + 1):
            if 0 <= f <= 7:
                piece = board.piece_at(chess.square(f, shield_rank))
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    shield_pawns += 1

    rival = not color
    attackers_near = sum(
        len(board.attackers(rival, sq))
        for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
    )
    return {"shield_pawns": shield_pawns, "attackers_near_king": attackers_near}


def _pawn_storm_towards(board: chess.Board, attacker_color: bool, defender_king_square) -> bool:
    if defender_king_square is None:
        return False
    king_file = chess.square_file(defender_king_square)
    storm_files = {f for f in (king_file - 1, king_file, king_file + 1) if 0 <= f <= 7}
    for sq in board.pieces(chess.PAWN, attacker_color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if f in storm_files:
            advance = r if attacker_color == chess.WHITE else 7 - r
            if advance >= 4:
                return True
    return False


def describe_position(board: chess.Board) -> list:
    bullets = []
    control = _center_control(board)
    if control[chess.WHITE] - control[chess.BLACK] >= 2:
        bullets.append("Blancas tienen mayor control central.")
    elif control[chess.BLACK] - control[chess.WHITE] >= 2:
        bullets.append("Negras tienen mayor control central.")
    else:
        bullets.append("El control central está equilibrado.")

    white_mat = material_score(board, chess.WHITE)
    black_mat = material_score(board, chess.BLACK)
    if white_mat != black_mat:
        leader = "Blancas" if white_mat > black_mat else "Negras"
        bullets.append(f"{leader} tienen ventaja material ({abs(white_mat - black_mat)} puntos).")

    if _pawn_storm_towards(board, chess.BLACK, board.king(chess.WHITE)):
        bullets.append("Negras presionan el flanco de rey blanco con avance de peones.")
    if _pawn_storm_towards(board, chess.WHITE, board.king(chess.BLACK)):
        bullets.append("Blancas presionan el flanco de rey negro con avance de peones.")

    bullets.append(f"La posición está en fase de {game_phase(board)}.")
    return bullets


def infer_rival_plan(board: chess.Board) -> tuple:
    """Devuelve (rival_color, tag, bullets) con el plan probable del rival
    (el que no tiene el turno, ya que acaba de mover o va a responder)."""
    rival = not board.turn
    rival_king = board.king(board.turn)  # rey de quien tiene el turno = objetivo del rival
    open_files, half_open = _open_and_half_open_files(board)

    if game_phase(board) == "endgame":
        return rival, "transicion_final", [
            "Buscar simplificar y llevar la partida a un final favorable.",
            "Activar el rey y las piezas de largo alcance.",
        ]

    if _pawn_storm_towards(board, rival, rival_king):
        king_file = chess.square_file(rival_king) if rival_king is not None else 4
        target_file_letter = chess.FILE_NAMES[min(king_file + 2, 7)] if rival == chess.WHITE else chess.FILE_NAMES[min(king_file + 2, 7)]
        return rival, "ataque_flanco_rey", [
            f"Seguir avanzando peones sobre el flanco de rey (columna {target_file_letter} u otras cercanas).",
            "Abrir una columna para meter la torre y atacar al rey.",
        ]

    rival_half_open = half_open[rival]
    if rival_half_open:
        file_letters = ", ".join(chess.FILE_NAMES[f] for f in rival_half_open[:2])
        return rival, "presion_columna_abierta", [
            f"Doblar torres/dama en la columna {file_letters}, semiabierta para su bando.",
            "Presionar la debilidad estructural asociada a esa columna.",
        ]

    control = _center_control(board)
    if abs(control[chess.WHITE] - control[chess.BLACK]) <= 1:
        return rival, "ruptura_central", [
            "Preparar una ruptura de peones en el centro para abrir líneas.",
            "Mejorar la coordinación de piezas antes de la ruptura.",
        ]

    return rival, "mejora_posicional", [
        "Reubicar piezas hacia casillas más activas.",
        "Mantener tensión y esperar una imprecisión propia antes de comprometerse.",
    ]


def list_threats(board: chess.Board) -> list:
    threats = []
    rival = not board.turn
    my_king_safety = _king_safety(board, board.turn)

    if my_king_safety["attackers_near_king"] > 0:
        threats.append("Hay piezas rivales atacando casillas junto a nuestro rey.")
    if my_king_safety["shield_pawns"] <= 1:
        threats.append("El escudo de peones frente a nuestro rey está debilitado.")

    open_files, half_open = _open_and_half_open_files(board)
    for f in half_open[board.turn]:
        if any(board.piece_type_at(sq) == chess.ROOK and board.color_at(sq) == rival
               for sq in chess.SquareSet(chess.BB_FILES[f])):
            threats.append(f"Columna {chess.FILE_NAMES[f]} semiabierta con torre rival ya apuntando.")

    for move in board.legal_moves:
        if board.gives_check(move):
            threats.append("El rival tiene jaques disponibles a considerar.")
            break

    return threats


def candidate_moves(board: chess.Board, evaluator=None, top_n: int = 4):
    if evaluator is not None:
        return evaluator.top_k_moves(board, k=top_n)

    scored = []
    for move in board.legal_moves:
        board.push(move)
        score = material_eval(board)
        board.pop()
        scored.append((move, score if board.turn == chess.BLACK else -score))
    scored.sort(key=lambda pair: pair[1], reverse=(board.turn == chess.WHITE))
    return [move for move, _ in scored[:top_n]]


def _move_reasons(board: chess.Board, move: chess.Move, plan_tag: str) -> list:
    reasons = []
    piece = board.piece_at(move.from_square)
    back_rank = 0 if board.turn == chess.WHITE else 7

    if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(move.from_square) == back_rank:
        reasons.append("desarrolla una pieza que estaba sin jugar")

    board_after = board.copy()
    board_after.push(move)

    if board_after.is_check():
        reasons.append("da jaque al rey rival")
    if move.to_square in CENTER_SQUARES:
        reasons.append("ocupa o refuerza una casilla central")
    if piece and piece.piece_type == chess.KING and board.is_kingside_castling(move):
        reasons.append("pone al rey a salvo y conecta las torres")
    if plan_tag == "ataque_flanco_rey":
        reasons.append("ayuda a defender el flanco de rey frente al plan rival")
    if plan_tag == "presion_columna_abierta" and piece and piece.piece_type == chess.ROOK:
        reasons.append("disputa la columna abierta")

    if not reasons:
        reasons.append("mejora la actividad general de las piezas")
    return reasons


def explain_position(board: chess.Board, evaluator=None, depth: int = 3, top_n: int = 4) -> str:
    rival, plan_tag, plan_bullets = infer_rival_plan(board)
    threats = list_threats(board)
    candidates = candidate_moves(board, evaluator=evaluator, top_n=top_n)

    if evaluator is not None:
        from chess_alpha.search import neural_guided_best_move
        recommended = neural_guided_best_move(board, evaluator, depth=depth, top_k=max(top_n, 8))
    else:
        recommended = best_move(board, depth=depth)

    reasons = _move_reasons(board, recommended, plan_tag)

    lines = ["Posición actual:"]
    lines += [f"- {b}" for b in describe_position(board)]
    lines.append("")
    lines.append(f"Plan probable de {SIDE_NAME[rival]}:")
    lines += [f"- {b}" for b in plan_bullets]

    if threats:
        lines.append("")
        lines.append("Amenazas:")
        lines += [f"- {t}" for t in threats]

    lines.append("")
    lines.append("Jugadas candidatas:")
    lines += [f"- {board.san(m)}" for m in candidates]

    lines.append("")
    lines.append("Jugada recomendada:")
    lines.append(board.san(recommended))

    lines.append("")
    lines.append("Motivo:")
    lines += [f"- {r}" for r in reasons]

    return "\n".join(lines)
