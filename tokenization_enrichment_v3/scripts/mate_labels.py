"""#24 -- mate distance labels. Extraidos DIRECTAMENTE de engine_eval.parquet
YA EXISTENTE (best_eval_type/best_eval_mate_in) -- no se reejecuta
Stockfish, no se infiere mate distance desde CP (prohibido explicitamente
en el encargo). Si best_eval_type != 'mate', estos campos son null (no
aplica, no es un error)."""
import pandas as pd


def compute_mate_labels(engine_eval: pd.DataFrame) -> pd.DataFrame:
    is_mate = engine_eval["best_eval_type"] == "mate"
    mate_in = engine_eval["best_eval_mate_in"]

    # mate_in > 0 => mate a favor de quien mueve; < 0 => mate en contra
    mate_for_side = pd.Series(index=engine_eval.index, dtype="object")
    mate_for_side[is_mate & (mate_in > 0)] = "side_to_move"
    mate_for_side[is_mate & (mate_in < 0)] = "opponent"

    return pd.DataFrame({
        "row_id": engine_eval["row_id"],
        "mate_available": is_mate,
        "mate_for_side": mate_for_side,
        "mate_in_n": mate_in.abs().where(is_mate),
        "absolute_mate_distance": mate_in.abs().where(is_mate),
    })
