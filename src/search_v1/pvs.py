"""Principal Variation Search de search_v1 (Fase V) - variante de
Alpha-Beta que busca la PRIMERA jugada de cada nodo con ventana completa
(-beta, -alpha) y el RESTO con ventana nula (-alpha-1, -alpha), asumiendo
que el ordenamiento de jugadas (move_ordering.py, ya validado en Fase III)
suele poner la mejor jugada primero - una ventana nula prueba mucho mas
rapido "esta jugada NO mejora alpha" sin gastar el esfuerzo de acotar su
valor exacto. Si una jugada tardia SI supera alpha (la ventana nula "falla
alto"), se re-busca con la ventana completa para obtener su valor exacto -
sin esto, PVS devolveria bounds en vez de scores exactos y rompería la
igualdad de resultado exigida contra Alpha-Beta.

GARANTIA DE CORRECCION (Gate V-A): con `pruning_config=None` (o todas las
tecnicas desactivadas), `pvs()` debe producir SIEMPRE el mismo score,
mate_distance y PV legal que `alpha_beta()` de Fase III/IV a la misma
profundidad, con las mismas configuraciones de tt/ordering_state/
quiescence_limits - la ventana nula + re-search NUNCA cambia el resultado
final de un alpha-beta correcto, solo el ORDEN/CANTIDAD de trabajo
(menos nodos si el ordenamiento es bueno). Verificado por
tests/search_v1/test_pvs.py::TestPVSMatchesAlphaBeta + benchmark A/B.

SEMANTICA FAIL-SOFT: igual que alpha_beta.py, el score devuelto NO se
clampea a [alpha, beta] - un fail-high devuelve el score real de la jugada
que corto (>= beta), un fail-low devuelve el mejor score real encontrado
(<= alpha original). La logica de re-search de PVS depende de esta
convencion (ver comentarios inline).

PODAS EXPERIMENTALES (opt-in via `pruning_config`, ver pruning_config.py):
    - Null move pruning: se intenta ANTES de generar/ordenar jugadas, una
      vez por nodo (no por jugada). Guardas: nunca en jaque, nunca en la
      raiz, nunca si el lado a mover no tiene material distinto de peones
      (evita el falso cutoff clasico de zugzwang en finales de peones),
      nunca si alpha/beta ya estan en rango de mate (evitar interferir con
      lineas de mate). Ver `_has_non_pawn_material`.
    - Late Move Reductions: solo sobre jugadas "tardias" (indice >=
      lmr_min_move_index), no capturas/promociones/jaques, con el nodo
      fuera de jaque - se busca primero con ventana nula Y profundidad
      reducida; si supera alpha, se re-busca con ventana nula a
      profundidad COMPLETA (telemetria lmr_researches, separada de
      pvs_researches) antes de decidir si hace falta el re-search de PVS
      de ventana completa.
    - Razoring: solo a profundidad <= razoring_max_depth, fuera de jaque y
      lejos de scores de mate - si la evaluacion estatica + margen no
      alcanza alpha, se asume que ninguna jugada tranquila va a levantar
      el resultado y se salta DIRECTO a quiescence (o a la evaluacion
      estatica si no hay quiescence configurada) en vez de generar/buscar
      jugadas normales.
    - Futility pruning: solo a profundidad <= futility_max_depth, fuera de
      jaque y lejos de scores de mate - si la evaluacion estatica + margen
      no alcanza alpha, se SALTAN (no se buscan) las jugadas tranquilas
      (ni captura ni promocion ni jaque) DESPUES de que ya exista al menos
      un best_score valido (nunca se salta la primera jugada considerada
      en el nodo, para garantizar que siempre quede una PV real).
      IMPORTANTE: ambas quedan disenadas e implementadas pero
      `enabled=False` por defecto y NO pasan por la validacion completa de
      categorias criticas de Fase V (ver PHASE_V_GATE.md, estado
      DESIGNED_NOT_ENABLED) - "la evaluacion estatica actual (material
      puro) limita la seguridad de estas tecnicas", tal como exige el
      pedido explicitamente ("no forzar su aprobacion").

PROOF_MODE (Fase VI): `proof_mode=True` DESACTIVA INCONDICIONALMENTE null
move/LMR/futility/razoring, sin importar lo que diga `pruning_config` (se
ignora por completo, no solo sus flags individuales - "una prueba exacta
no puede depender de una poda heuristica sin verificacion", item explicito
del pedido). El orden de jugadas (hash move/jaques/capturas/promociones/
killers/history/countermove) SIGUE permitido en proof_mode - cambia el
COSTO, nunca el conjunto de jugadas legales exploradas.

GATE PVS+DFPN (Fase VI, opt-in via `dfpn_gate`, ver dfpn_gate.py):
desactivado por defecto - "DFPN no debe ejecutarse indiscriminadamente en
cada nodo". Cuando esta habilitado y se cumplen sus condiciones (jaque +
profundidad restante acotada), se invoca `search_dfpn` con presupuesto
PROPIO y acotado. Si prueba PROVEN_MATE, el resultado se ELEVA a mate
exacto (score/PV) sin seguir con la busqueda normal de este nodo. Si
prueba no-mate, solo se registra en telemetria (no cambia el flujo). Si
devuelve UNKNOWN, se continua con PVS normal - UNKNOWN NUNCA se convierte
en un score exacto."""

from __future__ import annotations

import chess

from search_v1.alpha_beta import _order_moves
from search_v1.candidate_fusion import reorder_with_fusion
from search_v1.dfpn import search_dfpn
from search_v1.dfpn_limits import DFPNLimits
from search_v1.dfpn_table import DFPNTable
from search_v1.epoch4_provider import evaluate_epoch4, reorder_with_epoch4
from search_v1.mate_scores import MATE_SCORE, MATE_THRESHOLD, score_from_tt, score_to_tt, terminal_score
from search_v1.move_generator import generate_legal_moves
from search_v1.move_ordering import order_moves as _order_moves_advanced
from search_v1.position import Position
from search_v1.proof_status import ProofStatus
from search_v1.quiescence import quiescence
from search_v1.search_context import SearchContext
from search_v1.sequence_policy_provider import evaluate_sequence_policy, reorder_with_sequence_policy
from search_v1.static_evaluation import evaluate_static_relative
from search_v1.terminal import detect_terminal, is_fifty_move_rule
from search_v1.transposition_table import Bound


def _has_non_pawn_material(position: Position) -> bool:
    """Guarda de null move pruning: el lado a mover tiene al menos una
    pieza mayor/menor (no solo peones y rey) - evita el falso cutoff
    clasico de null move en finales de peones con zugzwang, donde "pasar"
    JAMAS es una opcion real."""
    board = position.board
    color = board.turn
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if board.pieces(piece_type, color):
            return True
    return False


def pvs(
    position: Position,
    depth: int,
    alpha: int,
    beta: int,
    context: SearchContext,
    tt=None,
    ordering_state=None,
    prev_move_uci: str | None = None,
    pruning_config=None,
    proof_mode: bool = False,
    dfpn_gate=None,
    dfpn_table: DFPNTable | None = None,
    epoch4_evaluator=None,
    epoch4_config=None,
    epoch4_cache=None,
    sequence_policy_evaluator=None,
    sequence_policy_config=None,
    sequence_policy_cache=None,
    root_move_history_ucis=None,
    fusion_config=None,
) -> tuple[int, list[str]]:
    """Wrapper FINO y ADITIVO (PASO 0, Search Tree Observatory) alrededor
    de `_pvs_impl` (el `pvs()` original, logica de busqueda intacta, byte
    a byte identica salvo el nombre de la funcion). Con
    `context.observer is None` (SIEMPRE salvo tracing explicito) el unico
    costo es un `if` mas por nodo - se delega de inmediato a
    `_pvs_impl` sin ninguna otra diferencia. Ver
    tests/search_v1/test_observer.py para la prueba de que esto NUNCA
    cambia score/best_move/nodes."""
    if context.observer is None:
        return _pvs_impl(
            position, depth, alpha, beta, context, tt=tt, ordering_state=ordering_state,
            prev_move_uci=prev_move_uci, pruning_config=pruning_config, proof_mode=proof_mode,
            dfpn_gate=dfpn_gate, dfpn_table=dfpn_table,
            epoch4_evaluator=epoch4_evaluator, epoch4_config=epoch4_config, epoch4_cache=epoch4_cache,
            sequence_policy_evaluator=sequence_policy_evaluator, sequence_policy_config=sequence_policy_config,
            sequence_policy_cache=sequence_policy_cache, root_move_history_ucis=root_move_history_ucis,
            fusion_config=fusion_config,
        )

    node_id = context.observer.enter(
        node_type="search", position=position, depth_remaining=depth,
        alpha=alpha, beta=beta, ply=context.ply_from_root, move_from_parent=prev_move_uci,
    )
    score, pv = _pvs_impl(
        position, depth, alpha, beta, context, tt=tt, ordering_state=ordering_state,
        prev_move_uci=prev_move_uci, pruning_config=pruning_config, proof_mode=proof_mode,
        dfpn_gate=dfpn_gate, dfpn_table=dfpn_table,
        epoch4_evaluator=epoch4_evaluator, epoch4_config=epoch4_config, epoch4_cache=epoch4_cache,
        sequence_policy_evaluator=sequence_policy_evaluator, sequence_policy_config=sequence_policy_config,
        sequence_policy_cache=sequence_policy_cache, root_move_history_ucis=root_move_history_ucis,
        fusion_config=fusion_config,
    )
    context.observer.exit(node_id, score, pv)
    return score, pv


def _pvs_impl(
    position: Position,
    depth: int,
    alpha: int,
    beta: int,
    context: SearchContext,
    tt=None,
    ordering_state=None,
    prev_move_uci: str | None = None,
    pruning_config=None,
    proof_mode: bool = False,
    dfpn_gate=None,
    dfpn_table: DFPNTable | None = None,
    epoch4_evaluator=None,
    epoch4_config=None,
    epoch4_cache=None,
    sequence_policy_evaluator=None,
    sequence_policy_config=None,
    sequence_policy_cache=None,
    root_move_history_ucis=None,
    fusion_config=None,
) -> tuple[int, list[str]]:
    if proof_mode:
        # ignora pruning_config POR COMPLETO, no solo sus flags - ver
        # docstring del modulo, seccion PROOF_MODE.
        pruning_config = None

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
            return quiescence(
                position, alpha, beta, context, context.quiescence_limits,
                context.quiescence_limits.max_qdepth, context.quiescence_limits.max_check_extensions,
                tt=tt, ordering_state=ordering_state, prev_move_uci=prev_move_uci,
            )
        context.leaf_nodes += 1
        return evaluate_static_relative(position), []

    original_alpha = alpha
    key = position.zobrist_key if tt is not None else None
    hash_move_uci = None

    if tt is not None:
        tt_entry = tt.probe(key)
        if tt_entry is not None:
            hash_move_uci = tt_entry.best_move
            if tt_entry.depth >= depth:
                stored = score_from_tt(tt_entry.score, context.ply_from_root)
                if tt_entry.bound == Bound.EXACT:
                    context.tt_cutoffs += 1
                    if context.observer is not None:
                        context.observer.record_tt_status("hit_exact")
                    return stored, [hash_move_uci] if hash_move_uci else []
                if tt_entry.bound == Bound.LOWER_BOUND and stored >= beta:
                    context.tt_cutoffs += 1
                    if context.observer is not None:
                        context.observer.record_tt_status("hit_lower_bound")
                    return stored, [hash_move_uci] if hash_move_uci else []
                if tt_entry.bound == Bound.UPPER_BOUND and stored <= alpha:
                    context.tt_cutoffs += 1
                    if context.observer is not None:
                        context.observer.record_tt_status("hit_upper_bound")
                    return stored, [hash_move_uci] if hash_move_uci else []
            if context.observer is not None:
                context.observer.record_tt_status("hit_no_cutoff")
        elif context.observer is not None:
            context.observer.record_tt_status("miss")

    # --- Gate PVS+DFPN (opt-in, ver dfpn_gate.py) ---
    if (
        dfpn_gate is not None and dfpn_gate.enabled
        and depth <= dfpn_gate.max_pvs_depth_to_trigger
        and (not dfpn_gate.trigger_on_check or position.is_check)
    ):
        context.dfpn_gate_triggers += 1
        gate_table = dfpn_table if dfpn_table is not None else DFPNTable()
        dfpn_limits = DFPNLimits(max_nodes=dfpn_gate.dfpn_max_nodes, max_depth_plies=dfpn_gate.dfpn_max_depth_plies)
        dfpn_res = search_dfpn(position, limits=dfpn_limits, table=gate_table, game_repetition=context.repetition)
        context.dfpn_gate_nodes_spent += dfpn_res.nodes
        if dfpn_res.proof_status == ProofStatus.PROVEN_MATE:
            context.dfpn_gate_mate_elevations += 1
            score = MATE_SCORE - (context.ply_from_root + dfpn_res.mate_distance_plies)
            pv = list(dfpn_res.principal_variation)
            if tt is not None:
                tt.store(key, depth, score_to_tt(score, context.ply_from_root), Bound.EXACT, pv[0] if pv else None)
            return score, pv
        elif dfpn_res.proof_status in (ProofStatus.PROVEN_NO_MATE, ProofStatus.PROVEN_NO_MATE_WITHIN_BOUND):
            context.dfpn_gate_no_mate_confirmations += 1
            # solo se registra - NO se usa para podar ni para fijar un
            # score (probar "no hay mate" no dice nada sobre el valor
            # posicional real, solo que no es un mate forzado)
        else:
            context.dfpn_gate_unknown += 1
            # UNKNOWN nunca se convierte en un resultado exacto - se sigue
            # con PVS normal, sin excepcion.

    # --- Null move pruning (opt-in, experimental) ---
    if (
        pruning_config is not None and pruning_config.null_move_enabled
        and context.ply_from_root > 0
        and depth >= pruning_config.null_move_min_depth
        and not position.is_check
        and beta < MATE_THRESHOLD
        and alpha > -MATE_THRESHOLD
        and _has_non_pawn_material(position)
    ):
        context.null_move_attempts += 1
        reduction = pruning_config.null_move_reduction
        position.apply_null()
        context.repetition.push(position)
        context.ply_from_root += 1
        try:
            null_score, _ = pvs(
                position, depth - 1 - reduction, -beta, -beta + 1, context,
                tt=tt, ordering_state=ordering_state, prev_move_uci=None,
                pruning_config=pruning_config, proof_mode=proof_mode,
                dfpn_gate=dfpn_gate, dfpn_table=dfpn_table,
                epoch4_evaluator=epoch4_evaluator, epoch4_config=epoch4_config, epoch4_cache=epoch4_cache,
            )
        finally:
            context.ply_from_root -= 1
            context.repetition.pop()
            position.undo()
        null_score = -null_score
        if null_score >= beta:
            context.null_move_cutoffs += 1
            return null_score, []

    node_in_check = position.is_check
    near_mate_window = beta >= MATE_THRESHOLD or alpha <= -MATE_THRESHOLD

    # --- Razoring (opt-in, DISENADA pero no validada - ver docstring del
    # modulo, DESIGNED_NOT_ENABLED en el gate) ---
    if (
        pruning_config is not None and pruning_config.razoring_enabled
        and depth <= pruning_config.razoring_max_depth
        and context.ply_from_root > 0
        and not node_in_check
        and not near_mate_window
    ):
        static_eval = evaluate_static_relative(position)
        if static_eval + pruning_config.razoring_margin_cp <= alpha:
            context.razoring_prunes += 1
            if context.quiescence_limits is not None:
                return quiescence(
                    position, alpha, beta, context, context.quiescence_limits,
                    context.quiescence_limits.max_qdepth, context.quiescence_limits.max_check_extensions,
                    tt=tt, ordering_state=ordering_state, prev_move_uci=prev_move_uci,
                )
            return static_eval, []

    # --- Futility pruning: flag calculado una vez por nodo (opt-in,
    # DISENADA pero no validada - ver nota de razoring arriba) ---
    futility_prune_active = False
    if (
        pruning_config is not None and pruning_config.futility_enabled
        and depth <= pruning_config.futility_max_depth
        and context.ply_from_root > 0
        and not node_in_check
        and not near_mate_window
    ):
        static_eval = evaluate_static_relative(position)
        futility_prune_active = static_eval + pruning_config.futility_margin_cp <= alpha

    legal_moves = generate_legal_moves(position)
    if ordering_state is not None:
        moves = _order_moves_advanced(position, legal_moves, hash_move_uci, context.ply_from_root, ordering_state, prev_move_uci)
    else:
        moves = _order_moves(position, legal_moves)

    # --- Epoch4 / Sequence Policy / Fusion, proveedores de ordenamiento
    # opt-in (Fase VII/VIII) - SOLO reordenan, nunca cambian el conjunto
    # de jugadas exploradas (garantizado por construccion en cada
    # reorder_with_*, verificado empiricamente en el gate de Fase VII).
    #
    # CADENA DE FALLBACK (item explicito del pedido de Fase VIII): una
    # excepcion de CUALQUIERA de los dos modelos (OOM, historia
    # inconsistente -> SequencePolicyHistoryMismatchError, etc.) NUNCA
    # tumba la busqueda - se captura, se cuenta en telemetria dedicada
    # (nunca fingir exito en silencio, mismo espiritu que
    # dfpn_gate_unknown de Fase VI) y se degrada al proveedor restante o
    # al orden clasico. Sequence Policy SOLO se evalua en la raiz
    # (context.ply_from_root == 0, decision de diseno explicita de
    # Fase VIII - ver sequence_policy_provider.py, scope=root_only). ---
    epoch4_probs = None
    if (
        epoch4_config is not None and epoch4_config.enabled
        and (epoch4_config.scope == "all_nodes" or context.ply_from_root == 0)
    ):
        try:
            epoch4_probs = evaluate_epoch4(position, epoch4_evaluator, epoch4_cache, context)
        except Exception:
            context.epoch4_fallback_triggered += 1
            epoch4_probs = None

    sequence_probs = None
    if (
        sequence_policy_config is not None and sequence_policy_config.enabled
        and context.ply_from_root == 0
    ):
        try:
            sequence_probs = evaluate_sequence_policy(
                position, sequence_policy_evaluator, sequence_policy_cache,
                root_move_history_ucis, context,
            )
        except Exception:
            context.sequence_fallback_triggered += 1
            sequence_probs = None

    if epoch4_probs is not None and sequence_probs is not None and fusion_config is not None and fusion_config.enabled:
        moves = reorder_with_fusion(moves, hash_move_uci, epoch4_probs, sequence_probs, fusion_config)
        context.fusion_positions_reordered += 1
    elif epoch4_probs is not None:
        moves = reorder_with_epoch4(moves, hash_move_uci, epoch4_probs)
        context.epoch4_positions_reordered += 1
        n_covered = sum(1 for lm in moves if lm.uci in epoch4_probs)
        context.epoch4_moves_covered_total += n_covered
        context.epoch4_moves_uncovered_total += len(moves) - n_covered
    elif sequence_probs is not None:
        moves = reorder_with_sequence_policy(moves, hash_move_uci, sequence_probs)
        context.sequence_positions_reordered += 1
        n_covered = sum(1 for lm in moves if lm.uci in sequence_probs)
        context.sequence_moves_covered_total += n_covered
        context.sequence_moves_uncovered_total += len(moves) - n_covered

    if context.observer is not None:
        context.observer.record_move_list([lm.uci for lm in moves], hash_move_uci)

    best_score = None
    best_pv: list[str] = []
    best_move_uci = None

    for i, lm in enumerate(moves):
        # Futility: salta jugadas tranquilas ya con un best_score valido
        # en mano (nunca se salta la primera jugada del nodo - siempre
        # queda al menos una PV real).
        if (
            futility_prune_active and best_score is not None
            and not lm.is_capture and not lm.is_promotion and not lm.gives_check
        ):
            context.futility_prunes += 1
            continue

        # LMR: candidata a reduccion SOLO si es una jugada tardia, tranquila
        # (no captura/promocion/jaque), el nodo actual no esta en jaque, y
        # no es la jugada de la TT (siempre se busca a profundidad completa).
        lmr_reduction = 0
        if (
            pruning_config is not None and pruning_config.lmr_enabled
            and i >= pruning_config.lmr_min_move_index
            and depth >= pruning_config.lmr_min_depth
            and not node_in_check
            and not lm.is_capture and not lm.is_promotion and not lm.gives_check
            and lm.uci != hash_move_uci
        ):
            lmr_reduction = pruning_config.lmr_reduction

        position.apply(lm.move)
        context.repetition.push(position)
        context.ply_from_root += 1
        try:
            if i == 0:
                child_score, child_pv = pvs(
                    position, depth - 1, -beta, -alpha, context,
                    tt=tt, ordering_state=ordering_state, prev_move_uci=lm.uci,
                    pruning_config=pruning_config, proof_mode=proof_mode,
                    dfpn_gate=dfpn_gate, dfpn_table=dfpn_table,
                    epoch4_evaluator=epoch4_evaluator, epoch4_config=epoch4_config, epoch4_cache=epoch4_cache,
                )
                context.pvs_full_window_searches += 1
            else:
                search_depth = depth - 1 - lmr_reduction
                child_score, child_pv = pvs(
                    position, search_depth, -alpha - 1, -alpha, context,
                    tt=tt, ordering_state=ordering_state, prev_move_uci=lm.uci,
                    pruning_config=pruning_config, proof_mode=proof_mode,
                    dfpn_gate=dfpn_gate, dfpn_table=dfpn_table,
                    epoch4_evaluator=epoch4_evaluator, epoch4_config=epoch4_config, epoch4_cache=epoch4_cache,
                )
                if lmr_reduction > 0 and -child_score > alpha:
                    # la reduccion "fallo alto" - re-busca ventana nula a
                    # profundidad COMPLETA (telemetria propia de LMR,
                    # separada de la contabilidad de PVS de mas abajo).
                    context.lmr_reductions_applied += 1
                    context.lmr_researches += 1
                    child_score, child_pv = pvs(
                        position, depth - 1, -alpha - 1, -alpha, context,
                        tt=tt, ordering_state=ordering_state, prev_move_uci=lm.uci,
                        pruning_config=pruning_config, proof_mode=proof_mode,
                        dfpn_gate=dfpn_gate, dfpn_table=dfpn_table,
                        epoch4_evaluator=epoch4_evaluator, epoch4_config=epoch4_config, epoch4_cache=epoch4_cache,
                    )
                elif lmr_reduction > 0:
                    context.lmr_reductions_applied += 1

                context.pvs_null_window_searches += 1
                score = -child_score
                if score >= beta:
                    context.pvs_fail_highs += 1
                elif score <= alpha:
                    context.pvs_fail_lows += 1
                else:
                    # la ventana nula "fallo alto" respecto de alpha sin
                    # llegar a probar beta - hace falta el valor EXACTO via
                    # ventana completa (unica forma de saber si de verdad
                    # mejora la PV, ver garantia de correccion arriba).
                    context.pvs_researches += 1
                    child_score, child_pv = pvs(
                        position, depth - 1, -beta, -alpha, context,
                        tt=tt, ordering_state=ordering_state, prev_move_uci=lm.uci,
                        pruning_config=pruning_config, proof_mode=proof_mode,
                        dfpn_gate=dfpn_gate, dfpn_table=dfpn_table,
                        epoch4_evaluator=epoch4_evaluator, epoch4_config=epoch4_config, epoch4_cache=epoch4_cache,
                    )
                    context.pvs_full_window_searches += 1
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
            if i == 0:
                context.first_move_cutoffs += 1
            if context.observer is not None:
                context.observer.record_cutoff(lm.uci, i)
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
