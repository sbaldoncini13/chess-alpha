"""PASO 27 (posiciones tacticas/criticas) y PASO 28 (dificultad), version
BASE (sin Stockfish) -- unicas senales disponibles en la tokenizacion
determinista: cantidad de movimientos legales, jaque, jugada forzada, fase.
Formulas configurables via tokenization_v2_config.json (nunca hardcodeadas
en silencio) -- ver SCHEMA_V2.md seccion "Formulas base" para la derivacion
exacta. Cuando el enriquecimiento Stockfish esta habilitado, un modulo
separado (engine_criticality.py) agrega columnas engine_* SIN sobrescribir
estas.
"""


def difficulty_score(legal_move_count: int, is_check: bool, game_phase: str, weights: dict) -> float:
    inv_mobility = 1.0 - min(legal_move_count, 60) / 60.0
    is_forced = legal_move_count == 1
    is_endgame = game_phase == "endgame"
    score = (
        weights["w_inverse_legal_moves"] * inv_mobility
        + weights["w_is_check"] * float(is_check)
        + weights["w_forced_move"] * float(is_forced)
        + weights["w_phase_endgame_bonus"] * float(is_endgame)
    )
    return max(0.0, min(1.0, score))


def difficulty_bucket(score: float) -> str:
    if score < 0.33:
        return "low"
    elif score < 0.66:
        return "medium"
    return "high"


def criticality_score(legal_move_count: int, is_check: bool, thresholds: dict) -> float:
    low_branching = legal_move_count <= thresholds["low_branching_max_legal_moves"]
    forced = legal_move_count <= thresholds["forced_move_max_legal_moves"]
    score = 0.5 * float(low_branching) + 0.3 * float(is_check) + 0.2 * float(forced)
    return max(0.0, min(1.0, score))


def is_tactical_or_critical(score: float) -> bool:
    return score >= 0.5
