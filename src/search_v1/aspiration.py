"""Aspiration windows de search_v1 (Fase V) - viven en iterative_deepening.py
(la propia especificacion de Fase V exige "aplicar SOLAMENTE desde
iterative deepening y cuando exista un score previo completo") - este
modulo solo define la configuracion + la funcion de re-busqueda con
ventana progresiva, PVS (`pvs.py`) no sabe nada de aspiration windows.

ALGORITMO (`search_with_aspiration`):
1. Ventana inicial: [score_previo - initial_window_cp, score_previo + initial_window_cp].
2. Si el resultado hace fail-low (score <= alpha Y alpha > -INFINITY) o
   fail-high (score >= beta Y beta < INFINITY): se amplia SOLO el lado que
   fallo (multiplicando la ventana de ESE lado por `widening_factor`) y se
   reintenta - nunca se ensancha el lado que ya estaba bien.
3. Tras `max_retries` reintentos sin exito, fallback a ventana completa
   (-INFINITY, INFINITY) - se reporta como fallback, nunca se finge que la
   ventana angosta funciono.
4. Si `SearchAborted` se propaga en cualquier intento (limite de nodos/
   tiempo/qnodes/check_extensions), se deja propagar tal cual - quien
   llama (iterative_deepening_search) ya descarta la iteracion COMPLETA
   ante SearchAborted (mismo mecanismo de Fase II-IV), por lo que nunca se
   devuelve una iteracion de aspiration parcial como si fuera completa.

No altera mate scores: la ventana se calcula aritmeticamente sobre
`score_previo` sin ninguna logica especial para valores de mate - si
`score_previo` ya es un mate score, la ventana ampliada rapidamente cubre
practicamente todo el rango relevante (comportamiento correcto, no
requiere casos especiales; ver guarda en iterative_deepening.py que evita
siquiera intentar una ventana angosta cuando el score previo ya es mate,
por eficiencia, no por correccion)."""

from __future__ import annotations

from dataclasses import dataclass

from search_v1.mate_scores import INFINITY


@dataclass(frozen=True)
class AspirationConfig:
    initial_window_cp: int = 25
    widening_factor: int = 4
    max_retries: int = 4

    def __post_init__(self):
        if self.initial_window_cp < 1:
            raise ValueError("initial_window_cp debe ser >= 1")
        if self.widening_factor < 2:
            raise ValueError("widening_factor debe ser >= 2")
        if self.max_retries < 0:
            raise ValueError("max_retries no puede ser negativo")


def search_with_aspiration(pvs_fn, position, depth, prev_score, context, config: AspirationConfig, **pvs_kwargs):
    """`pvs_fn` se recibe como parametro (en vez de importar pvs.py
    directamente) para evitar un ciclo de import iterative_deepening.py
    <-> pvs.py y para que sea trivialmente testeable con un stub."""
    window = config.initial_window_cp
    alpha = prev_score - window
    beta = prev_score + window
    retries = 0

    while True:
        a = max(alpha, -INFINITY)
        b = min(beta, INFINITY)
        score, pv = pvs_fn(position, depth, a, b, context, **pvs_kwargs)

        failed_low = score <= a and a > -INFINITY
        failed_high = score >= b and b < INFINITY
        if not failed_low and not failed_high:
            context.aspiration_successful_windows += 1
            return score, pv

        if failed_low:
            context.aspiration_fail_lows += 1
        if failed_high:
            context.aspiration_fail_highs += 1

        retries += 1
        if retries > config.max_retries:
            context.aspiration_fallbacks += 1
            return pvs_fn(position, depth, -INFINITY, INFINITY, context, **pvs_kwargs)

        context.aspiration_researches += 1
        window *= config.widening_factor
        if failed_low:
            alpha = prev_score - window
        if failed_high:
            beta = prev_score + window
