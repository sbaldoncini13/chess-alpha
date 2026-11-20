"""Iterative deepening de search_v1 (Fase II) - orquesta negamax.py o
alpha_beta.py en profundidad creciente (1, 2, ..., max_depth), conservando
SIEMPRE el resultado de la ultima iteracion COMPLETA. Si una iteracion se
aborta a mitad de camino (por limite de nodos o de tiempo), su resultado
parcial se DESCARTA por completo - nunca se devuelve como definitivo.

`position` se recibe por REFERENCIA (no se copia) y se muta/restaura
internamente via apply()/undo() en cada nodo visitado (mismo patron
"make/unmake" de Position ya validado en Fase I) - al terminar esta
funcion, `position` queda EXACTAMENTE en el mismo estado (mismo FEN) en el
que estaba al ser invocada, sin importar por que motivo termino la
busqueda (profundidad completa, limite de nodos, limite de tiempo, o
posicion raiz ya terminal)."""

from __future__ import annotations

from search_v1.alpha_beta import alpha_beta
from search_v1.aspiration import search_with_aspiration
from search_v1.epoch4_provider import Epoch4OrderingCache
from search_v1.mate_scores import INFINITY, is_mate_score, mate_distance_plies_from_score, terminal_score
from search_v1.move_generator import generate_legal_moves
from search_v1.negamax import negamax
from search_v1.position import Position
from search_v1.pvs import pvs
from search_v1.quiescence_result import quiescence_telemetry
from search_v1.repetition import RepetitionTracker
from search_v1.search_context import SearchAborted, SearchContext
from search_v1.search_limits import SearchLimits
from search_v1.search_result import ScoreType, SearchResult, StopReason
from search_v1.sequence_policy_provider import SequencePolicyCache
from search_v1.telemetry import nodes_per_second as _nps
from search_v1.terminal import detect_terminal

_STOP_REASON_BY_ABORT = {
    "max_nodes": StopReason.MAX_NODES,
    "max_time": StopReason.MAX_TIME,
    "max_qnodes": StopReason.MAX_QNODES,
    "max_check_extensions": StopReason.MAX_CHECK_EXTENSIONS,
}


def _telemetry_metadata(context, tt, ordering_state, quiescence_limits, use_pvs=False, pruning_config=None, aspiration=None, dfpn_gate=None, epoch4_config=None, sequence_policy_config=None, fusion_config=None) -> dict:
    metadata = {"algorithm": context.metadata_algorithm, "limits": context.limits.__dict__,
                "tt_cutoffs": context.tt_cutoffs}
    if quiescence_limits is not None:
        metadata["quiescence_limits"] = quiescence_limits.__dict__
        metadata.update(quiescence_telemetry(context))
    if use_pvs:
        metadata.update({
            "pvs_null_window_searches": context.pvs_null_window_searches,
            "pvs_researches": context.pvs_researches,
            "pvs_fail_highs": context.pvs_fail_highs,
            "pvs_fail_lows": context.pvs_fail_lows,
            "pvs_full_window_searches": context.pvs_full_window_searches,
            "first_move_cutoffs": context.first_move_cutoffs,
        })
    if aspiration is not None:
        metadata["aspiration_config"] = aspiration.__dict__
        metadata.update({
            "aspiration_fail_lows": context.aspiration_fail_lows,
            "aspiration_fail_highs": context.aspiration_fail_highs,
            "aspiration_researches": context.aspiration_researches,
            "aspiration_fallbacks": context.aspiration_fallbacks,
            "aspiration_successful_windows": context.aspiration_successful_windows,
        })
    if pruning_config is not None:
        metadata["pruning_config"] = pruning_config.__dict__
        metadata.update({
            "null_move_attempts": context.null_move_attempts,
            "null_move_cutoffs": context.null_move_cutoffs,
            "lmr_reductions_applied": context.lmr_reductions_applied,
            "lmr_researches": context.lmr_researches,
        })
    if dfpn_gate is not None:
        metadata["dfpn_gate_config"] = dfpn_gate.__dict__
        metadata.update({
            "dfpn_gate_triggers": context.dfpn_gate_triggers,
            "dfpn_gate_mate_elevations": context.dfpn_gate_mate_elevations,
            "dfpn_gate_no_mate_confirmations": context.dfpn_gate_no_mate_confirmations,
            "dfpn_gate_unknown": context.dfpn_gate_unknown,
            "dfpn_gate_nodes_spent": context.dfpn_gate_nodes_spent,
        })
    if epoch4_config is not None:
        metadata["epoch4_config"] = epoch4_config.__dict__
        metadata.update({
            "epoch4_cache_probes": context.epoch4_cache_probes,
            "epoch4_cache_hits": context.epoch4_cache_hits,
            "epoch4_inferences": context.epoch4_inferences,
            "epoch4_inference_time_ms": context.epoch4_inference_time_ms,
            "epoch4_positions_reordered": context.epoch4_positions_reordered,
            "epoch4_moves_covered_total": context.epoch4_moves_covered_total,
            "epoch4_moves_uncovered_total": context.epoch4_moves_uncovered_total,
            "epoch4_fallback_triggered": context.epoch4_fallback_triggered,
        })
    if sequence_policy_config is not None:
        metadata["sequence_policy_config"] = sequence_policy_config.__dict__
        metadata.update({
            "sequence_cache_probes": context.sequence_cache_probes,
            "sequence_cache_hits": context.sequence_cache_hits,
            "sequence_inferences": context.sequence_inferences,
            "sequence_inference_time_ms": context.sequence_inference_time_ms,
            "sequence_positions_reordered": context.sequence_positions_reordered,
            "sequence_moves_covered_total": context.sequence_moves_covered_total,
            "sequence_moves_uncovered_total": context.sequence_moves_uncovered_total,
            "sequence_fallback_triggered": context.sequence_fallback_triggered,
        })
    if fusion_config is not None:
        metadata["fusion_config"] = fusion_config.__dict__
        metadata["fusion_positions_reordered"] = context.fusion_positions_reordered
    if tt is not None:
        metadata.update({
            "tt_probes": tt.probes, "tt_hits": tt.hits, "tt_exact_hits": tt.exact_hits,
            "tt_lower_hits": tt.lower_hits, "tt_upper_hits": tt.upper_hits,
            "tt_stores": tt.stores, "tt_replacements": tt.replacements,
            "tt_collisions": tt.collisions, "tt_occupancy": tt.occupancy(),
            "tt_capacity": tt.capacity, "tt_estimated_bytes": tt.estimated_bytes(),
        })
    if ordering_state is not None:
        metadata.update({
            "hash_move_first_count": ordering_state.hash_move_first_count,
            "killer_cutoffs": ordering_state.killer_cutoffs,
            "history_updates": ordering_state.history_updates,
            "countermove_hits": ordering_state.countermove_hits,
            "ordering_time_ms": ordering_state.ordering_time_ms,
        })
    return metadata


def _build_result(best_score, best_pv, completed_depth, requested_depth, context, stop_reason, tt=None, ordering_state=None, quiescence_limits=None, use_pvs=False, pruning_config=None, aspiration=None, dfpn_gate=None, epoch4_config=None, sequence_policy_config=None, fusion_config=None) -> SearchResult:
    score = best_score if best_score is not None else 0
    score_type = ScoreType.MATE if is_mate_score(score) else ScoreType.CENTIPAWNS
    elapsed = context.elapsed_ms()
    return SearchResult(
        best_move=best_pv[0] if best_pv else None,
        score=score,
        score_type=score_type,
        mate_distance_plies=mate_distance_plies_from_score(score),
        principal_variation=list(best_pv),
        completed_depth=completed_depth,
        requested_depth=requested_depth,
        nodes=context.nodes,
        leaf_nodes=context.leaf_nodes,
        terminal_nodes=context.terminal_nodes,
        alpha_beta_cutoffs=context.alpha_beta_cutoffs,
        elapsed_ms=elapsed,
        nodes_per_second=_nps(context.nodes, elapsed),
        stop_reason=stop_reason,
        completed=completed_depth >= 1,
        legal=bool(best_pv),
        metadata=_telemetry_metadata(context, tt, ordering_state, quiescence_limits, use_pvs, pruning_config, aspiration, dfpn_gate, epoch4_config, sequence_policy_config, fusion_config),
    )


def iterative_deepening_search(
    position: Position,
    limits: SearchLimits,
    algorithm: str = "alpha_beta",
    game_history: RepetitionTracker | None = None,
    tt=None,
    ordering_state=None,
    quiescence_limits=None,
    use_pvs: bool = False,
    pruning_config=None,
    aspiration=None,
    proof_mode: bool = False,
    dfpn_gate=None,
    dfpn_table=None,
    epoch4_evaluator=None,
    epoch4_config=None,
    sequence_policy_evaluator=None,
    sequence_policy_config=None,
    root_move_history_ucis=None,
    fusion_config=None,
    observer=None,
) -> SearchResult:
    """`tt`/`ordering_state` son opt-in (Fase III), `quiescence_limits` es
    opt-in (Fase IV) - omitidos, reproducen exactamente el comportamiento
    de Fase II/III. `tt`, si se provee, avanza de GENERACION una vez al
    inicio de esta llamada (no entre iteraciones de profundidad dentro de
    la misma llamada - las iteraciones SI reutilizan entradas de
    generaciones anteriores dentro de la misma busqueda, que es donde la
    TT realmente aporta valor); `algorithm` debe ser 'alpha_beta' si se
    usa tt/ordering_state/quiescence_limits (negamax.py no los soporta,
    permanece el oraculo puro de Fase II sin cambios).

    Fase V (opt-in, omitidos reproducen exactamente Fase II/III/IV):
    `use_pvs=True` usa `pvs.py` (Principal Variation Search) en vez de
    `alpha_beta.py` para las iteraciones de profundidad - requiere
    `algorithm='alpha_beta'`. `pruning_config` (podas experimentales
    opt-in dentro de PVS) y `aspiration` (ventanas de aspiration,
    ampliadas progresivamente desde ESTA funcion, nunca dentro de pvs.py)
    requieren `use_pvs=True`. Aspiration solo se activa a partir de la
    2da iteracion con score previo completo Y no-mate (ver
    aspiration.py) - la primera iteracion (sin score previo) y cualquier
    iteracion posterior a un score de mate usan ventana completa.

    Fase VI (opt-in, requieren use_pvs=True): `proof_mode=True` desactiva
    incondicionalmente null move/LMR/futility/razoring dentro de pvs.py
    (ignora `pruning_config` por completo). `dfpn_gate` (ver dfpn_gate.py)
    habilita la consulta condicional a DFPN desde nodos en jaque -
    desactivado por defecto. `dfpn_table` permite compartir una tabla de
    DFPN entre llamadas sucesivas (opcional, se crea una nueva por defecto
    si el gate esta activo y no se provee ninguna).

    Fase VII (opt-in, requiere use_pvs=True): `epoch4_config` (ver
    epoch4_provider.py) habilita el reordenamiento de jugadas via Epoch4 -
    desactivado por defecto, SOLO reordena, nunca cambia el resultado
    final de PVS. `epoch4_evaluator` es el objeto ya construido (inyectado
    por quien llama, nunca construido aca) que expone
    `.predict(board) -> (probs, value_white)` - requerido si
    `epoch4_config.enabled`. Un `Epoch4OrderingCache` propio se crea UNA
    VEZ por llamada a esta funcion (mismo patron que `SearchContext`),
    namespace separado de `tt`/`dfpn_table`.

    Fase VIII (opt-in, requiere use_pvs=True): `sequence_policy_config`
    (ver sequence_policy_provider.py) habilita el reordenamiento via
    Sequence Policy V1 FULL - desactivado por defecto, SOLO reordena,
    SOLO en la raiz de la busqueda (`scope="root_only"`, no configurable
    todavia). `sequence_policy_evaluator` es el objeto ya construido
    (inyectado, nunca construido aca) que expone
    `.predict(board, history_ucis) -> dict[uci, float]` - requerido si
    `sequence_policy_config.enabled`. `root_move_history_ucis` es la
    secuencia REAL de jugadas UCI desde el inicio de la partida hasta
    `position` (la raiz) - requerida si `sequence_policy_config.enabled`
    (lista vacia = posicion inicial valida; `None` es un error de wiring
    del llamador). `fusion_config` (ver candidate_fusion.py) habilita la
    fusion RRF de Epoch4+Sequence sobre la union de sus candidatos en
    espacio UCI - si `epoch4_config` y `sequence_policy_config` estan
    ambos habilitados, `fusion_config` habilitado es OBLIGATORIO (define
    el modo D de ablacion; habilitar ambos proveedores sin fusion se
    rechaza en vez de un default silencioso "gana epoch4" - ver
    prior_mode.py para la convencion de nombres de los 4 modos). Un
    `SequencePolicyCache` propio se crea UNA VEZ por llamada, namespace
    separado de `tt`/`dfpn_table`/cache de Epoch4. Una excepcion en
    tiempo de ejecucion de CUALQUIERA de los dos modelos (Epoch4 o
    Sequence Policy) durante la busqueda NUNCA la tumba - pvs.py degrada
    al proveedor restante o al orden clasico y lo cuenta en telemetria
    (`epoch4_fallback_triggered`/`sequence_fallback_triggered`).

    PASO 0 (Search Tree Observatory, opt-in, aditivo): `observer` (ver
    search_v1/observer.py) - None por defecto, CERO cambio de
    comportamiento/resultado. Si se provee un `SearchObserver`, se asigna
    a `context.observer` y pvs()/quiescence() registran un evento por
    nodo visitado (alpha/beta in/out, cutoff, TT status, orden de
    jugadas) mas un resumen por iteracion de profundidad."""
    if algorithm not in ("alpha_beta", "negamax"):
        raise ValueError(f"algorithm invalido: {algorithm!r} (esperado 'alpha_beta' o 'negamax')")
    if algorithm == "negamax" and (
        tt is not None or ordering_state is not None or quiescence_limits is not None
        or use_pvs or pruning_config is not None or aspiration is not None
        or proof_mode or dfpn_gate is not None or epoch4_config is not None
        or sequence_policy_config is not None or fusion_config is not None
    ):
        raise ValueError(
            "negamax no soporta tt/ordering_state/quiescence_limits/pvs/pruning/aspiration/"
            "proof_mode/dfpn_gate/epoch4_config/sequence_policy_config/fusion_config - "
            "son exclusivos de alpha_beta (Fase III/IV/V/VI/VII/VIII)"
        )
    if not use_pvs and (
        pruning_config is not None or aspiration is not None or proof_mode
        or dfpn_gate is not None or epoch4_config is not None
        or sequence_policy_config is not None or fusion_config is not None
    ):
        raise ValueError(
            "pruning_config/aspiration/proof_mode/dfpn_gate/epoch4_config/sequence_policy_config/"
            "fusion_config requieren use_pvs=True (exclusivos de PVS)"
        )
    if sequence_policy_config is not None and sequence_policy_config.enabled:
        if sequence_policy_evaluator is None:
            raise ValueError("sequence_policy_config.enabled requiere sequence_policy_evaluator")
        if root_move_history_ucis is None:
            raise ValueError(
                "sequence_policy_config.enabled requiere root_move_history_ucis "
                "(lista vacia = posicion inicial valida; None es un error de wiring)"
            )
    epoch4_enabled = epoch4_config is not None and epoch4_config.enabled
    sequence_enabled = sequence_policy_config is not None and sequence_policy_config.enabled
    if epoch4_enabled and sequence_enabled and not (fusion_config is not None and fusion_config.enabled):
        raise ValueError(
            "epoch4_config y sequence_policy_config habilitados simultaneamente requieren "
            "fusion_config habilitado (define el modo D de ablacion) - habilitar ambos proveedores "
            "sin fusion se rechaza en vez de un default silencioso"
        )

    if tt is not None:
        tt.new_generation()

    root_terminal = detect_terminal(position)
    root_has_moves = len(generate_legal_moves(position)) > 0
    if root_terminal.is_terminal or not root_has_moves:
        score = terminal_score(root_terminal, 0) if root_terminal.is_terminal else 0
        return SearchResult(
            best_move=None, score=score,
            score_type=ScoreType.MATE if is_mate_score(score) else ScoreType.CENTIPAWNS,
            mate_distance_plies=mate_distance_plies_from_score(score),
            principal_variation=[], completed_depth=0, requested_depth=limits.max_depth,
            nodes=0, leaf_nodes=0, terminal_nodes=1, alpha_beta_cutoffs=0,
            elapsed_ms=0.0, nodes_per_second=0.0, stop_reason=StopReason.TERMINAL_ROOT,
            completed=True, legal=False, metadata={"algorithm": algorithm},
        )

    repetition = game_history if game_history is not None else RepetitionTracker()
    if len(repetition) == 0:
        repetition.push(position)

    context = SearchContext(limits, repetition, quiescence_limits=quiescence_limits)
    context.metadata_algorithm = algorithm  # atributo dinamico, solo para metadata del resultado final
    context.observer = observer  # PASO 0, opt-in - None por defecto, ver docstring

    # Fase VII: cache de Epoch4 propio, UNA VEZ por llamada (mismo patron
    # que SearchContext arriba) - namespace separado de tt/dfpn_table.
    epoch4_cache = (
        Epoch4OrderingCache(epoch4_config.cache_capacity_entries)
        if epoch4_config is not None and epoch4_config.enabled
        else None
    )

    # Fase VIII: cache de Sequence Policy propio, UNA VEZ por llamada
    # (mismo patron que epoch4_cache arriba) - namespace separado de
    # tt/dfpn_table/epoch4_cache.
    sequence_policy_cache = (
        SequencePolicyCache(sequence_policy_config.cache_capacity_entries)
        if sequence_policy_config is not None and sequence_policy_config.enabled
        else None
    )

    best_score, best_pv, completed_depth = None, [], 0
    stop_reason = StopReason.DEPTH_COMPLETE

    for depth in range(1, limits.max_depth + 1):
        context.ply_from_root = 0
        try:
            if algorithm == "negamax":
                score, pv = negamax(position, depth, context)
            elif use_pvs:
                if aspiration is not None and best_score is not None and depth > 1 and not is_mate_score(best_score):
                    score, pv = search_with_aspiration(
                        pvs, position, depth, best_score, context, aspiration,
                        tt=tt, ordering_state=ordering_state, pruning_config=pruning_config,
                        proof_mode=proof_mode, dfpn_gate=dfpn_gate, dfpn_table=dfpn_table,
                        epoch4_evaluator=epoch4_evaluator, epoch4_config=epoch4_config, epoch4_cache=epoch4_cache,
                        sequence_policy_evaluator=sequence_policy_evaluator, sequence_policy_config=sequence_policy_config,
                        sequence_policy_cache=sequence_policy_cache, root_move_history_ucis=root_move_history_ucis,
                        fusion_config=fusion_config,
                    )
                else:
                    score, pv = pvs(
                        position, depth, -INFINITY, INFINITY, context, tt=tt, ordering_state=ordering_state,
                        pruning_config=pruning_config, proof_mode=proof_mode, dfpn_gate=dfpn_gate, dfpn_table=dfpn_table,
                        epoch4_evaluator=epoch4_evaluator, epoch4_config=epoch4_config, epoch4_cache=epoch4_cache,
                        sequence_policy_evaluator=sequence_policy_evaluator, sequence_policy_config=sequence_policy_config,
                        sequence_policy_cache=sequence_policy_cache, root_move_history_ucis=root_move_history_ucis,
                        fusion_config=fusion_config,
                    )
            else:
                score, pv = alpha_beta(position, depth, -INFINITY, INFINITY, context, tt=tt, ordering_state=ordering_state)
        except SearchAborted as exc:
            stop_reason = _STOP_REASON_BY_ABORT[exc.reason]
            break
        best_score, best_pv, completed_depth = score, pv, depth
        if observer is not None:
            observer.record_iteration(
                depth=depth, score=score, pv=pv, nodes_cumulative=context.nodes,
                q_nodes_cumulative=context.q_nodes, elapsed_ms_cumulative=context.elapsed_ms(),
                tt_hits_cumulative=tt.hits if tt is not None else None,
                alpha_beta_cutoffs_cumulative=context.alpha_beta_cutoffs,
            )
    else:
        stop_reason = StopReason.DEPTH_COMPLETE

    return _build_result(best_score, best_pv, completed_depth, limits.max_depth, context, stop_reason,
                          tt=tt, ordering_state=ordering_state, quiescence_limits=quiescence_limits,
                          use_pvs=use_pvs, pruning_config=pruning_config, aspiration=aspiration, dfpn_gate=dfpn_gate,
                          epoch4_config=epoch4_config, sequence_policy_config=sequence_policy_config,
                          fusion_config=fusion_config)
