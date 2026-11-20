"""#19 -- trayectoria de evaluacion. Usa EXCLUSIVAMENTE engine_eval.parquet
YA EXISTENTE del Golden (best_eval_unified_cp_equivalent). NUNCA vuelve a
invocar Stockfish. Solo mira hacia atras (t, t-1, t-2, ...), nunca t+1 --
si se necesitara futuro para otro analisis, esa es otra feature marcada
FUTURE_DEPENDENT (ver terminal_distance.py), no esta.

Normalizacion de perspectiva (CRITICO, mismo cuidado que en todo el
proyecto con la convencion side-to-move -- ver
tokenization_v1/scripts/value_target.py, solo lectura): cada
`best_eval_unified_cp_equivalent` viene desde la perspectiva de quien
mueve EN ESA posicion, que alterna cada ply. Para que un delta entre t y
t-1 sea aritmeticamente significativo, TODO se normaliza a perspectiva de
BLANCAS antes de restar:
    eval_white_pov = eval_stm if side_to_move_white else -eval_stm
delta_eval_k = eval_white_pov[t] - eval_white_pov[t-k], positivo = mejoro
para blancas en esos k plies.
"""
import numpy as np
import pandas as pd


def build_eval_white_pov(positions: pd.DataFrame, engine_eval: pd.DataFrame) -> pd.Series:
    """Serie indexada por row_id, NaN donde no hay engine_eval (filas
    terminales)."""
    ev = engine_eval.set_index("row_id")["best_eval_unified_cp_equivalent"]
    stm = positions.set_index("row_id")["side_to_move"]
    aligned_stm = stm.reindex(ev.index)
    white_pov = np.where(aligned_stm, ev, -ev)
    return pd.Series(white_pov, index=ev.index, dtype="float64")


def compute_eval_trajectory(positions: pd.DataFrame, game_index: pd.DataFrame, eval_white_pov: pd.Series,
                             rolling_window: int = 4, volatility_window: int = 8) -> pd.DataFrame:
    row_ids = positions["row_id"].to_numpy()
    game_ids = positions["game_id"].to_numpy()
    row_start = game_index["row_id_start"].loc[game_ids].to_numpy()

    n = len(row_ids)
    eval_t = np.full(n, np.nan)
    delta_1 = np.full(n, np.nan)
    delta_2 = np.full(n, np.nan)
    rolling_change = np.full(n, np.nan)
    volatility = np.full(n, np.nan)

    ev_map = eval_white_pov.to_dict()

    for i in range(n):
        r = row_ids[i]
        eval_t[i] = ev_map.get(r, np.nan)

        if r - 1 >= row_start[i] and (r - 1) in ev_map:
            delta_1[i] = ev_map.get(r) - ev_map.get(r - 1) if r in ev_map else np.nan
        if r - 2 >= row_start[i] and (r - 2) in ev_map and r in ev_map:
            delta_2[i] = ev_map.get(r) - ev_map.get(r - 2)

        window_start = max(row_start[i], r - rolling_window)
        window_vals = [ev_map[w] for w in range(window_start, r + 1) if w in ev_map]
        if len(window_vals) >= 2:
            diffs = np.diff(window_vals)
            rolling_change[i] = float(np.mean(np.abs(diffs)))

        vol_start = max(row_start[i], r - volatility_window)
        vol_vals = [ev_map[w] for w in range(vol_start, r + 1) if w in ev_map]
        if len(vol_vals) >= 2:
            volatility[i] = float(np.std(vol_vals))

    return pd.DataFrame({
        "row_id": row_ids, "eval_t_white_pov": eval_t, "delta_eval_1": delta_1, "delta_eval_2": delta_2,
        "rolling_eval_change": rolling_change, "evaluation_volatility": volatility,
    })
