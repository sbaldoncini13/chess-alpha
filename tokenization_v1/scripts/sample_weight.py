"""PASO 29: infraestructura de sample_weight. Valor por defecto 1.0 para
toda fila del piloto -- este paso NO decide como ponderar, solo deja la
columna lista para que una decision posterior (fuera de este alcance)
pueda recalcularla a partir de calidad de partida / Elo / rareza / fase /
posicion critica / duplicacion / calidad Stockfish, sin tener que volver a
tokenizar nada.
"""

DEFAULT_SAMPLE_WEIGHT = 1.0


def default_sample_weight() -> float:
    return DEFAULT_SAMPLE_WEIGHT


def recompute_sample_weight(*, position_occurrence_count=None, tactical_or_critical=None,
                             white_elo=None, black_elo=None, engine_move_quality=None) -> float:
    """Punto de extension documentado, NO invocado por el pipeline del
    piloto (que siempre usa default_sample_weight()==1.0). Deliberadamente
    no implementado mas alla de la firma: la formula real de ponderacion es
    una decision de entrenamiento posterior, fuera del alcance de esta
    tarea (ver CLAUDE.md / encargo: "no eliminar datos automaticamente
    basandose en estos pesos", "el proposito es preparar el dataset para
    decisiones posteriores")."""
    raise NotImplementedError(
        "recompute_sample_weight es un placeholder documentado para una fase "
        "de entrenamiento futura, fuera del alcance de tokenization_v1. "
        "Usar default_sample_weight() en el pipeline del piloto."
    )
