"""#23 -- distancia al final de partida. FUTURE_DEPENDENT SIEMPRE (usa el
largo TOTAL de la partida, informacion que no existe en el momento de la
posicion). input_safe=False de forma permanente -- NUNCA debe entrar como
feature de inferencia. Util solo para curriculum/sampling/analisis."""
import pandas as pd


def compute_terminal_distance(positions: pd.DataFrame, game_index: pd.DataFrame) -> pd.DataFrame:
    game_ids = positions["game_id"].to_numpy()
    ply = positions["ply"].to_numpy()
    last_ply = (game_index["n_rows"] - 1).loc[game_ids].to_numpy()
    plies_to_terminal = last_ply - ply
    return pd.DataFrame({
        "row_id": positions["row_id"].to_numpy(),
        "plies_to_terminal": plies_to_terminal,
        "leakage_for_input": True,
    })
