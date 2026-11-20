"""Quiescence search de search_v1 (Fase IV) - extiende la busqueda mas
alla del horizonte normal SOLO por lineas tacticas (capturas/promociones/
jaques tacticos opcionales/evasiones), para mitigar el efecto horizonte de
una evaluacion estatica puramente material (Fase I/II).

SEMANTICA (documentada exactamente segun el pedido):
    - Posicion NO en jaque: stand-pat (evaluate_static_relative) como
      cota inferior de lo que el lado a mover puede lograr (siempre tiene
      la opcion de "no hacer nada tactico"), sujeto a cutoff normal de
      alpha-beta. Si stand-pat >= beta, corta. Si stand-pat > alpha, se
      convierte en la nueva cota. Se buscan despues las jugadas tacticas
      para ver si alguna mejora sobre el stand-pat.
    - Posicion EN jaque: NUNCA stand-pat (criterio de FAIL explicito si se
      hiciera) - se generan TODAS las evasiones legales (no solo
      tacticas, ver tactical_moves.py) y se buscan todas.

LIMITES DE SEGURIDAD (previenen explosiones/ciclos sin poda silenciosa):
    - max_qdepth: plies de busqueda tactica normal (medidos desde donde
      arranca ESTA llamada a quiescence, no desde la raiz de la busqueda
      principal). Al agotarse SIN estar en jaque, se devuelve stand-pat.
    - max_check_extensions: presupuesto SEPARADO que solo se consume
      cuando la posicion sigue en jaque despues de agotar max_qdepth
      (nunca se hace stand-pat en jaque). Al agotarse, se lanza
      SearchAborted("max_check_extensions") - el mismo mecanismo limpio
      ya usado para max_nodes/max_time en Fase II/III, que hace que
      iterative_deepening descarte la iteracion completa en vez de
      devolver un resultado incorrecto o corrupto.
    - max_qnodes: techo GLOBAL de nodos de quiescence para TODA la
      llamada a iterative_deepening_search (no por sub-arbol individual) -
      ver SearchContext.visit_qnode().
    - Repeticion/regla de 50/terminales: se revisan en CADA nodo de
      quiescence exactamente igual que en alpha_beta.py - un ciclo de
      jaques real siempre termina repitiendo una posicion, que estos
      chequeos detectan antes de necesitar ninguna de las salvaguardas
      anteriores.

TRANSPOSITION TABLE: las entradas de quiescence se guardan con
profundidad NEGATIVA via `quiescence_depth_key()`, garantizado disjunto
de las profundidades de busqueda normal (siempre >= 1) incluso durante
extensiones de jaque - ver TranspositionTable.probe_for_quiescence() y
el docstring del modulo transposition_table.py (seccion "BUG REAL") para
la logica de aceptacion asimetrica (una entrada de busqueda normal
completa SIEMPRE es valida para quiescence; lo inverso nunca es cierto) y
el bug de colision de profundidad ya corregido."""

from __future__ import annotations

from search_v1.mate_scores import score_from_tt, score_to_tt, terminal_score
from search_v1.move_ordering import order_tactical_moves
from search_v1.position import Position
from search_v1.quiescence_limits import QuiescenceLimits
from search_v1.search_context import SearchAborted, SearchContext
from search_v1.static_evaluation import evaluate_static_relative
from search_v1.tactical_moves import generate_tactical_moves
from search_v1.terminal import is_fifty_move_rule, detect_terminal
from search_v1.transposition_table import Bound, quiescence_depth_key


def quiescence(
    position: Position,
    alpha: int,
    beta: int,
    context: SearchContext,
    qlimits: QuiescenceLimits,
    qdepth_remaining: int,
    check_extensions_remaining: int,
    tt=None,
    ordering_state=None,
    prev_move_uci: str | None = None,
) -> tuple[int, list[str]]:
    """Wrapper FINO y ADITIVO (PASO 0, Search Tree Observatory) alrededor
    de `_quiescence_impl` - mismo patron que pvs.py/pvs(). Con
    `context.observer is None` (siempre, salvo tracing explicito), delega
    de inmediato sin ninguna otra diferencia de comportamiento."""
    if context.observer is None:
        return _quiescence_impl(
            position, alpha, beta, context, qlimits, qdepth_remaining, check_extensions_remaining,
            tt=tt, ordering_state=ordering_state, prev_move_uci=prev_move_uci,
        )

    node_id = context.observer.enter(
        node_type="quiescence", position=position, depth_remaining=qdepth_remaining,
        alpha=alpha, beta=beta, ply=context.ply_from_root, move_from_parent=prev_move_uci,
    )
    score, pv = _quiescence_impl(
        position, alpha, beta, context, qlimits, qdepth_remaining, check_extensions_remaining,
        tt=tt, ordering_state=ordering_state, prev_move_uci=prev_move_uci,
    )
    context.observer.exit(node_id, score, pv)
    return score, pv


def _quiescence_impl(
    position: Position,
    alpha: int,
    beta: int,
    context: SearchContext,
    qlimits: QuiescenceLimits,
    qdepth_remaining: int,
    check_extensions_remaining: int,
    tt=None,
    ordering_state=None,
    prev_move_uci: str | None = None,
) -> tuple[int, list[str]]:
    context.visit_qnode()

    terminal = detect_terminal(position)
    if terminal.is_terminal:
        context.q_terminal_nodes += 1
        return terminal_score(terminal, context.ply_from_root), []

    if is_fifty_move_rule(position):
        context.q_terminal_nodes += 1
        return 0, []

    if context.repetition.is_threefold_repetition(position):
        context.q_terminal_nodes += 1
        return 0, []

    in_check = position.is_check
    qdepth_used = qlimits.max_qdepth - qdepth_remaining
    if qdepth_used > context.q_max_depth_reached:
        context.q_max_depth_reached = qdepth_used

    if in_check and qdepth_remaining <= 0:
        # extension de jaque: presupuesto SEPARADO de max_qdepth, nunca
        # se hace stand-pat en jaque - ver docstring del modulo.
        if check_extensions_remaining <= 0:
            raise SearchAborted("max_check_extensions")
        check_extensions_remaining -= 1

    original_alpha = alpha
    key = position.zobrist_key if tt is not None else None
    hash_move_uci = None

    if tt is not None:
        context.q_tt_probes += 1
        tt_entry = tt.probe_for_quiescence(key, qdepth_remaining, qlimits)
        if tt_entry is not None:
            context.q_tt_hits += 1
            hash_move_uci = tt_entry.best_move
            stored = score_from_tt(tt_entry.score, context.ply_from_root)
            if tt_entry.bound == Bound.EXACT:
                if context.observer is not None:
                    context.observer.record_tt_status("hit_exact")
                return stored, [hash_move_uci] if hash_move_uci else []
            if tt_entry.bound == Bound.LOWER_BOUND and stored >= beta:
                if context.observer is not None:
                    context.observer.record_tt_status("hit_lower_bound")
                return stored, [hash_move_uci] if hash_move_uci else []
            if tt_entry.bound == Bound.UPPER_BOUND and stored <= alpha:
                if context.observer is not None:
                    context.observer.record_tt_status("hit_upper_bound")
                return stored, [hash_move_uci] if hash_move_uci else []
            if context.observer is not None:
                context.observer.record_tt_status("hit_no_cutoff")
        elif context.observer is not None:
            context.observer.record_tt_status("miss")

    if not in_check:
        stand_pat = evaluate_static_relative(position)
        context.q_stand_pat_calls += 1
        if stand_pat >= beta:
            context.q_stand_pat_cutoffs += 1
            context.q_beta_cutoffs += 1
            context.q_leaf_nodes += 1
            if tt is not None:
                tt.store(key, quiescence_depth_key(qdepth_remaining), score_to_tt(stand_pat, context.ply_from_root),
                         Bound.LOWER_BOUND, None)
            return stand_pat, []

        if stand_pat > alpha:
            alpha = stand_pat
        best_score = stand_pat
        best_pv: list[str] = []
        best_move_uci = None

        if qdepth_remaining <= 0:
            context.q_depth_limit_stops += 1
            context.q_leaf_nodes += 1
            if tt is not None:
                tt.store(key, quiescence_depth_key(qdepth_remaining), score_to_tt(stand_pat, context.ply_from_root),
                         Bound.EXACT, None)
            return best_score, best_pv

        allow_checks = qlimits.allow_tactical_checks and qdepth_used < qlimits.tactical_checks_max_qdepth
        moves = generate_tactical_moves(position, allow_tactical_checks=allow_checks)
        if not moves:
            context.q_leaf_nodes += 1  # posicion quieta, sin jugadas tacticas disponibles
    else:
        best_score = None
        best_pv = []
        best_move_uci = None
        moves = generate_tactical_moves(position)  # todas las evasiones, siempre

    ordered = order_tactical_moves(position, moves, hash_move_uci, ordering_state)
    if context.observer is not None:
        context.observer.record_move_list([lm.uci for lm in ordered], hash_move_uci)

    for idx, lm in enumerate(ordered):
        if lm.is_en_passant:
            context.q_en_passant_searched += 1
        elif lm.is_capture:
            context.q_captures_searched += 1
        if lm.is_promotion:
            context.q_promotions_searched += 1
        if in_check:
            context.q_evasions_searched += 1
        elif not lm.is_capture and not lm.is_promotion and lm.gives_check:
            context.q_checks_searched += 1

        position.apply(lm.move)
        context.repetition.push(position)
        context.ply_from_root += 1
        try:
            child_score, child_pv = quiescence(
                position, -beta, -alpha, context, qlimits,
                qdepth_remaining - 1, check_extensions_remaining,
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
            context.q_beta_cutoffs += 1
            if context.observer is not None:
                context.observer.record_cutoff(lm.uci, idx)
            break

    if tt is not None:
        if best_score <= original_alpha:
            bound = Bound.UPPER_BOUND
        elif best_score >= beta:
            bound = Bound.LOWER_BOUND
        else:
            bound = Bound.EXACT
        tt.store(key, quiescence_depth_key(qdepth_remaining), score_to_tt(best_score, context.ply_from_root), bound, best_move_uci)

    return best_score, best_pv
