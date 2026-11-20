"""Ventanas causales con contexto solapado y loss_mask sin duplicar/perder
targets (seccion 5 del pedido).

Secuencia logica de una partida: S = [BOS, a1, a2, ..., an, EOS], indices
S-index 0..n+1 (S-index 0=BOS, 1..n=movimientos reales, n+1=EOS). BOS/EOS son
virtuales: no se guardan en tokenization_v1, se agregan al construir el
tensor de entrada de cada ventana.

Cada ventana de contexto C cubre como INPUT los S-index [context_start,
context_end) y como TARGETS POSIBLES los S-index [context_start+1,
context_end] (prediccion causal: la posicion i predice S[i+1]). Solo se
cuenta en la loss el sub-rango [loss_start, loss_end] que todavia no fue
cubierto por una ventana anterior de la MISMA partida -- eso es lo que
garantiza cobertura=100% y duplicacion=0 sin overlap de loss.

stride = context_length - 1: la ventana siguiente arranca un token antes del
ultimo target ya cubierto, para tener contexto completo (C tokens) al
predecir el primer target nuevo.
"""

BOS_STD = "BOS_STD"  # placeholder simbolico; el id numerico real vive en constants.py


def compute_windows_for_game(n_plies: int, context_length: int):
    """Devuelve una lista de dicts (uno por ventana) para una partida de
    n_plies movimientos reales. S-index: 0=BOS, 1..n_plies=movimientos,
    n_plies+1=EOS. No requiere los tokens en si -- es pura aritmetica sobre
    longitudes, reusable tanto para construir el manifest como para el
    auditor."""
    N = n_plies + 2  # BOS + movimientos + EOS
    stride = context_length - 1
    assert stride >= 1, "context_length debe ser >= 2"

    windows = []
    next_uncovered_target = 1  # primer target real es S[1]
    context_start = 0
    last_target = N - 1  # S-index del EOS, ultimo target posible

    while next_uncovered_target <= last_target:
        context_end = min(context_start + context_length, N)
        if context_end <= context_start:
            break
        target_start = context_start + 1
        # La ultima posicion de entrada (S-index context_end-1) predice
        # S[context_end] -- valido solo si context_end <= N-1. Si la ventana
        # alcanza el final exacto (context_end == N, el input incluye EOS
        # como ultimo token), el ultimo target valido es N-1 (el propio EOS),
        # no N (que esta fuera de rango). Bug real encontrado por el audit:
        # usar context_end directo generaba un target_end=N inexistente.
        target_end = min(context_end, N - 1)  # inclusive, ultimo target que esta ventana PUEDE predecir
        loss_start = max(target_start, next_uncovered_target)
        loss_end = target_end
        if loss_start <= loss_end:
            windows.append({
                "context_start": context_start,
                "context_end": context_end,
                "window_length": context_end - context_start,
                "target_start": target_start,
                "target_end": target_end,
                "loss_start": loss_start,
                "loss_end": loss_end,
            })
            next_uncovered_target = loss_end + 1
        if context_end >= N:
            break
        context_start += stride

    return windows


def validate_windows_cover_exactly(n_plies: int, context_length: int):
    """Auditoria pura (sin tokens): reconstruye el conjunto de targets
    cubiertos por TODAS las ventanas y verifica cobertura=100%, dup=0,
    EOS unico, orden correcto. Usado por 04_build_sequence_dataset.py y por
    el reporte WINDOWING_AUDIT."""
    windows = compute_windows_for_game(n_plies, context_length)
    N = n_plies + 2
    expected_targets = set(range(1, N))  # 1..N-1 (incluye EOS = N-1)

    covered = []
    for w in windows:
        covered.extend(range(w["loss_start"], w["loss_end"] + 1))

    covered_set = set(covered)
    n_duplicates = len(covered) - len(covered_set)
    missing = expected_targets - covered_set
    extra = covered_set - expected_targets
    eos_index = N - 1
    eos_count = covered.count(eos_index)

    # orden: los loss_start deben ser estrictamente crecientes entre ventanas
    loss_starts = [w["loss_start"] for w in windows]
    order_ok = loss_starts == sorted(loss_starts) and len(set(loss_starts)) == len(loss_starts)

    return {
        "n_plies": n_plies,
        "context_length": context_length,
        "n_windows": len(windows),
        "coverage_pct": 100.0 * len(covered_set & expected_targets) / len(expected_targets) if expected_targets else 100.0,
        "n_duplicates": n_duplicates,
        "n_missing": len(missing),
        "missing_sample": sorted(missing)[:10],
        "n_extra": len(extra),
        "eos_index": eos_index,
        "eos_covered_exactly_once": eos_count == 1,
        "order_ok": order_ok,
        "windows": windows,
    }
