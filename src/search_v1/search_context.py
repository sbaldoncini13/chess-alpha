"""Estado mutable de una corrida de busqueda de search_v1 (Fase II).

`ply_from_root` vive aca (no como parametro de negamax/alpha_beta) para
mantener firmas de funcion uniformes entre negamax.py y alpha_beta.py y
reducir el riesgo de bugs de paso de parametros - se incrementa/decrementa
alrededor de cada llamada recursiva, siempre dentro de un try/finally para
garantizar que quede correcto incluso si SearchAborted se propaga a mitad
de la recursion.

El limite de NODOS se revisa en cada nodo (comparacion de enteros,
costo despreciable). El limite de TIEMPO usa time.perf_counter()
(reloj MONOTONICO, nunca la hora del sistema - inmune a ajustes de reloj)
pero solo se consulta cada `time_check_frequency` nodos, tal como pide
el pedido ("no comprobar el tiempo en cada operacion si degrada
excesivamente la busqueda")."""

from __future__ import annotations

import time

from search_v1.repetition import RepetitionTracker
from search_v1.search_limits import SearchLimits


class SearchAborted(Exception):
    """Se lanza cuando se alcanza un limite (nodos, tiempo, qnodes o
    extensiones de jaque de quiescence) a mitad de una iteracion de
    profundidad. Quien la captura (iterative_deepening) debe DESCARTAR el
    resultado parcial de esa iteracion, nunca usarlo como definitivo."""

    def __init__(self, reason: str):
        self.reason = reason  # "max_nodes", "max_time", "max_qnodes" o "max_check_extensions"
        super().__init__(f"Busqueda abortada: {reason}")


class SearchContext:
    def __init__(self, limits: SearchLimits, repetition: RepetitionTracker, quiescence_limits=None):
        self.limits = limits
        self.repetition = repetition
        self.nodes = 0
        self.leaf_nodes = 0
        self.terminal_nodes = 0
        self.alpha_beta_cutoffs = 0
        self.ply_from_root = 0
        self.tt_cutoffs = 0  # Fase III: cutoffs resueltos directamente por un hit de TT, sin recursion

        # PASO 0 (Search Tree Observatory): contador siempre-activo, mismo
        # costo que tt_cutoffs de arriba - cuenta CADA vez que
        # is_threefold_repetition() da True en pvs.py/quiescence.py (no
        # gateado por observer, es tan barato como los contadores que ya
        # existian antes de este agregado).
        self.repetition_detections = 0
        self._start = time.perf_counter()
        self._deadline = (
            self._start + limits.max_time_ms / 1000.0 if limits.max_time_ms is not None else None
        )
        self._nodes_since_time_check = 0

        # Fase IV: quiescence (todos en 0 si quiescence nunca se activa)
        self.quiescence_limits = quiescence_limits
        self.q_nodes = 0
        self.q_leaf_nodes = 0
        self.q_terminal_nodes = 0
        self.q_stand_pat_calls = 0
        self.q_stand_pat_cutoffs = 0
        self.q_beta_cutoffs = 0
        self.q_captures_searched = 0
        self.q_en_passant_searched = 0
        self.q_promotions_searched = 0
        self.q_checks_searched = 0
        self.q_evasions_searched = 0
        self.q_max_depth_reached = 0
        self.q_node_limit_stops = 0
        self.q_depth_limit_stops = 0
        self.q_tt_probes = 0
        self.q_tt_hits = 0

        # Fase V: PVS (todos en 0 si pvs() nunca se usa)
        self.pvs_null_window_searches = 0
        self.pvs_researches = 0
        self.pvs_fail_highs = 0
        self.pvs_fail_lows = 0
        self.pvs_full_window_searches = 0
        # Fase VIII: cuantos alpha_beta_cutoffs ocurrieron en la PRIMERA
        # jugada examinada del nodo (i==0) - proxy directo de calidad de
        # ordenamiento (mejor ordenamiento -> mas cutoffs en la primera
        # jugada, menos re-busquedas de PVS).
        self.first_move_cutoffs = 0

        # Fase V: aspiration windows (todos en 0 si iterative_deepening no
        # las usa - ver aspiration.py, solo se incrementan desde ahi)
        self.aspiration_fail_lows = 0
        self.aspiration_fail_highs = 0
        self.aspiration_researches = 0
        self.aspiration_fallbacks = 0
        self.aspiration_successful_windows = 0

        # Fase V: podas experimentales opt-in (todos en 0 si pruning_config
        # no las habilita)
        self.null_move_attempts = 0
        self.null_move_cutoffs = 0
        self.lmr_reductions_applied = 0
        self.lmr_researches = 0
        self.futility_prunes = 0
        self.razoring_prunes = 0

        # Fase VI: gate PVS+DFPN (todos en 0 si dfpn_gate no esta habilitado)
        self.dfpn_gate_triggers = 0
        self.dfpn_gate_mate_elevations = 0
        self.dfpn_gate_no_mate_confirmations = 0
        self.dfpn_gate_unknown = 0
        self.dfpn_gate_nodes_spent = 0

        # Fase VII: Epoch4 ordering-only provider (todos en 0 si
        # epoch4_config no esta habilitado - ver epoch4_provider.py)
        self.epoch4_cache_probes = 0
        self.epoch4_cache_hits = 0
        self.epoch4_inferences = 0
        self.epoch4_inference_time_ms = 0.0
        self.epoch4_positions_reordered = 0
        self.epoch4_moves_covered_total = 0
        self.epoch4_moves_uncovered_total = 0

        # Fase VIII: Sequence Policy (root-only) + Fusion (todos en 0 si
        # sequence_policy_config/fusion_config no estan habilitados - ver
        # sequence_policy_provider.py/candidate_fusion.py)
        self.sequence_cache_probes = 0
        self.sequence_cache_hits = 0
        self.sequence_inferences = 0
        self.sequence_inference_time_ms = 0.0
        self.sequence_positions_reordered = 0
        self.sequence_moves_covered_total = 0
        self.sequence_moves_uncovered_total = 0
        self.fusion_positions_reordered = 0

        # Fase VIII: cadena de fallback (nunca fingir exito en silencio -
        # 0 en una corrida sana; un valor distinto de 0 es una senal de
        # alerta, no un evento rutinario)
        self.epoch4_fallback_triggered = 0
        self.sequence_fallback_triggered = 0

        # PASO 0 (Search Tree Observatory): instrumentacion OPT-IN,
        # ADITIVA, por-nodo - ver search_v1/observer.py. None (default)
        # SIEMPRE que quien llama no pida tracing explicitamente - todo
        # punto de instrumentacion en pvs.py/quiescence.py/
        # iterative_deepening.py esta guardado por
        # `if context.observer is not None:`, cero cambio de
        # comportamiento con el valor por defecto.
        self.observer = None

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def _check_time(self) -> None:
        if self._deadline is not None:
            self._nodes_since_time_check += 1
            if self._nodes_since_time_check >= self.limits.time_check_frequency:
                self._nodes_since_time_check = 0
                if time.perf_counter() >= self._deadline:
                    raise SearchAborted("max_time")

    def visit_node(self) -> None:
        """Llamar al ENTRAR a negamax()/alpha_beta() para cada nodo
        (interno, hoja o terminal) - incrementa el contador y revisa
        limites ANTES de hacer cualquier trabajo en ese nodo."""
        self.nodes += 1

        if self.limits.max_nodes is not None and self.nodes >= self.limits.max_nodes:
            raise SearchAborted("max_nodes")

        self._check_time()

    def visit_qnode(self) -> None:
        """Equivalente de visit_node() para quiescence.py - cuenta contra
        `q_nodes` (separado de `nodes`) y contra el techo GLOBAL
        `quiescence_limits.max_qnodes` (compartido por toda la llamada a
        iterative_deepening_search, no por cada sub-arbol de quiescence -
        previene explosiones combinatorias reales). El tiempo se comparte
        con la busqueda principal (un unico deadline para toda la
        corrida, no uno separado para quiescence)."""
        self.q_nodes += 1

        if self.quiescence_limits is not None and self.quiescence_limits.max_qnodes is not None:
            if self.q_nodes >= self.quiescence_limits.max_qnodes:
                raise SearchAborted("max_qnodes")

        self._check_time()
