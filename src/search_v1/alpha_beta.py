"""Alpha-Beta de search_v1 (Fase II, extendido en Fase III) - misma
semantica minimax que negamax.py (debe producir SIEMPRE el mismo score;
el best_move puede diferir unicamente en empates exactos, por el orden de
exploracion distinto).

FASE II (sin cambios de comportamiento): `tt=None, ordering_state=None`
(los valores por defecto) reproducen EXACTAMENTE el algoritmo validado en
Fase II - mismo orden de movimientos (`_order_moves`, tier fijo
mate>captura>promocion>jaque>resto + UCI), sin Transposition Table, sin
killers/history/countermove. Los 130 tests de Fase II siguen pasando sin
modificacion porque llaman `alpha_beta(position, depth, alpha, beta,
context)` con exactamente esos 5 argumentos posicionales.

FASE III (opt-in via kwargs):
    tt: TranspositionTable | None - si se provee, se prueba (`probe`) al
        entrar a cada nodo (score normalizado con score_from_tt) y se
        guarda (`store`) al salir (score normalizado con score_to_tt).
        Un hit con profundidad suficiente puede cortar la busqueda sin
        recursion (EXACT siempre; LOWER_BOUND si stored>=beta; UPPER_BOUND
        si stored<=alpha). El movimiento guardado (`best_move`) SIEMPRE se
        revalida como legal antes de usarse para ordenar (nunca se confia
        ciegamente en una entrada vieja).
    ordering_state: OrderingState | None - si se provee, reemplaza
        `_order_moves` por `move_ordering.order_moves` (hash move / mate
        inmediato / promociones / capturas MVV-LVA / killers / countermove
        / history). Los cutoffs sobre jugadas NO capturantes actualizan
        killers/history/countermove (ver seccion final de la funcion).
    prev_move_uci: str | None - UCI del movimiento que llevo a la posicion
        actual (para countermove) - se hilvana automaticamente en la
        recursion, no hace falta pasarlo desde fuera salvo en el nodo raiz."""

from __future__ import annotations

from search_v1.mate_scores import score_from_tt, score_to_tt, terminal_score
from search_v1.move_generator import generate_legal_moves
from search_v1.move_ordering import order_moves as _order_moves_advanced
from search_v1.position import Position
from search_v1.quiescence import quiescence
from search_v1.search_context import SearchContext
from search_v1.static_evaluation import evaluate_static_relative
from search_v1.terminal import TerminalStatus, detect_terminal, is_fifty_move_rule
from search_v1.transposition_table import Bound

_TIER_MATE = 0
_TIER_CAPTURE = 1
_TIER_PROMOTION = 2
_TIER_CHECK = 3
_TIER_REST = 4


def _order_moves(position: Position, moves) -> list:
    """Ordenamiento de Fase II (sin TT/killers/history/countermove) -
    preservado tal cual para que tt=None, ordering_state=None reproduzca
    exactamente el comportamiento ya validado."""
    keyed = []
    for lm in moves:
        if lm.gives_check:
            position.apply(lm.move)
            is_mate = detect_terminal(position).status == TerminalStatus.CHECKMATE
            position.undo()
            tier = _TIER_MATE if is_mate else _TIER_CHECK
        elif lm.is_capture:
            tier = _TIER_CAPTURE
        elif lm.is_promotion:
            tier = _TIER_PROMOTION
        else:
            tier = _TIER_REST
        keyed.append((tier, lm.uci, lm))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [lm for _, _, lm in keyed]


def alpha_beta(
    position: Position,
    depth: int,
    alpha: int,
    beta: int,
    context: SearchContext,
    tt=None,
    ordering_state=None,
    prev_move_uci: str | None = None,
) -> tuple[int, list[str]]:
    context.visit_node()

    terminal = detect_terminal(position)
    if terminal.is_terminal:
        context.terminal_nodes += 1
        return terminal_score(terminal, context.ply_from_root), []

    if is_fifty_move_rule(position):
        context.terminal_nodes += 1
        return 0, []

    if context.repetition.is_threefold_repetition(position):
        context.terminal_nodes += 1
        return 0, []

    if depth <= 0:
        if context.quiescence_limits is not None:
            # Fase IV: el horizonte normal entrega el control a quiescence
            # (capturas/promociones/jaques tacticos/evasiones) en vez de
            # evaluar estaticamente en seco - mitiga el efecto horizonte.
            return quiescence(
                position, alpha, beta, context, context.quiescence_limits,
                context.quiescence_limits.max_qdepth, context.quiescence_limits.max_check_extensions,
                tt=tt, ordering_state=ordering_state, prev_move_uci=prev_move_uci,
            )
        context.leaf_nodes += 1
        return evaluate_static_relative(position), []

    original_alpha = alpha
    key = position.zobrist_key if tt is not None else None
    tt_entry = None
    hash_move_uci = None

    if tt is not None:
        tt_entry = tt.probe(key)
        if tt_entry is not None:
            hash_move_uci = tt_entry.best_move
            if tt_entry.depth >= depth:
                stored = score_from_tt(tt_entry.score, context.ply_from_root)
                if tt_entry.bound == Bound.EXACT:
                    context.tt_cutoffs += 1
                    return stored, [hash_move_uci] if hash_move_uci else []
                if tt_entry.bound == Bound.LOWER_BOUND and stored >= beta:
                    context.tt_cutoffs += 1
                    return stored, [hash_move_uci] if hash_move_uci else []
                if tt_entry.bound == Bound.UPPER_BOUND and stored <= alpha:
                    context.tt_cutoffs += 1
                    return stored, [hash_move_uci] if hash_move_uci else []

    legal_moves = generate_legal_moves(position)
    if ordering_state is not None:
        moves = _order_moves_advanced(position, legal_moves, hash_move_uci, context.ply_from_root, ordering_state, prev_move_uci)
    else:
        moves = _order_moves(position, legal_moves)

    best_score = None
    best_pv: list[str] = []
    best_move_uci = None

    for lm in moves:
        position.apply(lm.move)
        context.repetition.push(position)
        context.ply_from_root += 1
        try:
            child_score, child_pv = alpha_beta(
                position, depth - 1, -beta, -alpha, context,
                tt=tt, ordering_state=ordering_state, prev_move_uci=lm.uci,
            )
        finally:
            context.ply_from_root -= 1
            context.repetition.pop()
            position.undo()

        score = -child_score
        if best_score is None or score > best_score:
            best_score = score
            best_pv = [lm.uci] + child_pv
            best_move_uci = lm.uci

        if score > alpha:
            alpha = score
        if alpha >= beta:
            context.alpha_beta_cutoffs += 1
            if ordering_state is not None and not lm.is_capture and not lm.is_promotion:
                if lm.uci in ordering_state._ply_killers(context.ply_from_root):
                    ordering_state.killer_cutoffs += 1
                ordering_state.update_killer(context.ply_from_root, lm.uci)
                ordering_state.update_history(lm.move.from_square, lm.move.to_square, depth)
                if prev_move_uci is not None:
                    ordering_state.update_countermove(prev_move_uci, lm.uci)
            break

    if tt is not None:
        if best_score <= original_alpha:
            bound = Bound.UPPER_BOUND
        elif best_score >= beta:
            bound = Bound.LOWER_BOUND
        else:
            bound = Bound.EXACT
        tt.store(key, depth, score_to_tt(best_score, context.ply_from_root), bound, best_move_uci)

    return best_score, best_pv
