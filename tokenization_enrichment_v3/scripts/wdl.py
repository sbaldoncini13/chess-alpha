"""#22 -- WDL derivado de la evaluacion existente. Investigacion previa
(documentada en config.wdl y aca): el repo YA tiene una transformacion
cp->probabilidad reutilizable y validada:

  win_probability = 1 / (1 + 10^(-cp/400))
  (src/19_analyze_elo_games_errors.py:cp_to_winprob, solo lectura, no
  importada -- se reimplementa la MISMA formula documentada aca para
  mantener tokenization_enrichment_v3/ independiente).

Esto da win_probability (asumiendo "ganar o mejor" en un sentido de
evaluacion, NO un WDL de 3 vias real). El repo NO tiene ninguna
transformacion reproducible de 3 vias (win/draw/loss) ya validada -- el
WDL nativo de Stockfish requeriria una NUEVA consulta UCI (prohibida en
esta tarea). Por lo tanto:

  win_probability_from_cp  -> IMPLEMENTED (formula existente reutilizada)
  draw_probability / loss_probability (WDL completo) -> NOT_IMPLEMENTED_REQUIRES_DEFINITION

Los CP/mate originales de engine_eval NUNCA se modifican; esto es
puramente una columna derivada adicional.
"""
import numpy as np

CP_MATE_SCORE = 100_000


def cp_to_winprob(cp: float) -> float:
    return 1.0 / (1.0 + 10 ** (-cp / 400.0))


def compute_wdl(eval_unified_cp_equivalent: np.ndarray, eval_type: np.ndarray) -> dict:
    """eval_unified_cp_equivalent: array de best_eval_unified_cp_equivalent
    (perspectiva side-to-move, igual que en engine_eval.parquet). Para
    mates se satura a +-1.0 de forma natural via la formula logistica
    aplicada al valor unificado (ya de magnitud ~100000), documentado."""
    cp = np.asarray(eval_unified_cp_equivalent, dtype=np.float64)
    win_prob = cp_to_winprob(cp)
    return {
        "win_probability_from_cp": win_prob,
        "draw_probability": None,
        "loss_probability": None,
        "wdl_status": "NOT_IMPLEMENTED_REQUIRES_DEFINITION (solo win_probability tiene formula reutilizable)",
    }
