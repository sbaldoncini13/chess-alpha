"""Seleccion estratificada y determinista de una muestra de posiciones para
la prueba controlada de Stockfish (12_stockfish_validation.py). Funcion pura
-- no lee ni escribe archivos, para poder testearse con un DataFrame
sintetico pequeno sin depender del piloto real de 818.611 filas.

Estrategia (ver SCHEMA_V2.md 'Prueba controlada'): cuotas fijas para
jaque/tactico (para garantizar cobertura de esos casos), el resto repartido
lo mas parejo posible entre las 3 fases de juego. No fuerza distribuciones
imposibles: si una cuota no tiene suficientes candidatos disponibles, toma
los que haya y sigue.
"""
import numpy as np


def _pick(rng, sub_df, k):
    k = min(k, len(sub_df))
    if k == 0:
        return np.array([], dtype=np.int64)
    idx = rng.choice(len(sub_df), size=k, replace=False)
    return sub_df.iloc[idx]["row_id"].to_numpy()


def select_stratified_sample(df, n_total: int, seed: int, n_check_quota: int, n_tactical_quota: int) -> np.ndarray:
    """`df` debe tener columnas row_id, is_check, tactical_or_critical,
    game_phase, y NO debe incluir filas terminales (responsabilidad del
    llamador filtrar human_move_id != -1 antes de llamar). Devuelve un
    array ordenado de row_id, sin duplicados, tamano <= n_total."""
    rng = np.random.default_rng(seed)
    pool = df

    check_ids = _pick(rng, pool[pool["is_check"]], n_check_quota)
    pool = pool[~pool["row_id"].isin(check_ids)]

    tactical_ids = _pick(rng, pool[pool["tactical_or_critical"]], n_tactical_quota)
    pool = pool[~pool["row_id"].isin(tactical_ids)]

    remaining = max(0, n_total - len(check_ids) - len(tactical_ids))
    phases = ["opening", "middlegame", "endgame"]
    per_phase = remaining // len(phases)
    extra = remaining - per_phase * len(phases)

    phase_ids_all = []
    for i, phase in enumerate(phases):
        k = per_phase + (1 if i < extra else 0)
        sub = pool[pool["game_phase"] == phase]
        ids = _pick(rng, sub, k)
        phase_ids_all.append(ids)
        pool = pool[~pool["row_id"].isin(ids)]

    all_ids = np.concatenate([check_ids, tactical_ids] + phase_ids_all)
    return np.sort(np.unique(all_ids))
