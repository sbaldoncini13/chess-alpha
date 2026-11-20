"""Registro anti-leakage de las 28 familias de V3. Clasificacion por
FAMILIA (no por cada uno de los 200+ campos individuales -- todos los
campos de una misma familia comparten la misma clasificacion de origen,
ver docstrings de cada modulo para el detalle campo-por-campo).

Categorias (PASO 42 del encargo):
  POSITION_AVAILABLE : se conoce solo con la posicion actual + su
                        historial legal previo. Candidato potencial a INPUT.
  ENGINE_DERIVED      : viene de Stockfish ya calculado (Golden). Por
                        defecto input_safe=False salvo decision posterior
                        explicita (PASO 42).
  FUTURE_DEPENDENT    : usa informacion posterior al momento de la
                        posicion. input_safe=SIEMPRE False.
"""
import json

POSITION_AVAILABLE = "POSITION_AVAILABLE"
ENGINE_DERIVED = "ENGINE_DERIVED"
FUTURE_DEPENDENT = "FUTURE_DEPENDENT"

FAMILIES = [
    {"id": 1, "name": "attack_maps", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_tactical.parquet", "module": "attack_maps.py",
     "description": "Mapas de ataque compactos (bitboard 64b) + conteos por color."},
    {"id": 2, "name": "attack_density", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "attack_maps/ (attackers_count_white/black uint8[64])", "module": "attack_maps.py",
     "description": "Densidad de atacantes/defensores por casilla."},
    {"id": 3, "name": "hanging_pieces", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_tactical.parquet", "module": "hanging_pieces.py",
     "description": "Piezas atacadas/defendidas/colgadas (definicion: atacada Y cero defensores)."},
    {"id": 4, "name": "mobility", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_strategic.parquet", "module": "mobility.py",
     "description": "Movilidad total y por tipo de pieza, ambos colores."},
    {"id": 5, "name": "king_safety", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "king_safety.parquet", "module": "king_safety.py",
     "description": "King ring, pawn shield, open/semi-open files, proximidad dama/torre, exposure score."},
    {"id": 6, "name": "pawn_structure", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "pawn_structure.parquet", "module": "pawn_structure.py",
     "description": "Isolated/doubled/tripled/backward/connected/blocked/islands."},
    {"id": 7, "name": "passed_pawns", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "pawn_structure.parquet", "module": "pawn_structure.py",
     "description": "Passed/connected passed/protected passed + rank relativo maximo."},
    {"id": 8, "name": "pawn_majorities", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "pawn_structure.parquet", "module": "pawn_structure.py",
     "description": "Mayoria de peones kingside/queenside/center por color."},
    {"id": 9, "name": "outposts", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_strategic.parquet", "module": "outposts.py",
     "description": "Casillas debiles/holes/outposts de caballo, heuristica conservadora."},
    {"id": 10, "name": "pins", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_tactical.parquet", "module": "pins_skewers.py",
     "description": "Pins absolutos (board.is_pinned) + relativos (geometria propia)."},
    {"id": 11, "name": "skewers_xray", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_tactical.parquet", "module": "pins_skewers.py",
     "description": "Skewers y x-ray attacks via escaneo de rayos."},
    {"id": 12, "name": "overloaded_pieces", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_tactical.parquet", "module": "overloaded.py",
     "description": "Piezas unica defensora de 2+ casillas disputadas simultaneamente."},
    {"id": 13, "name": "see", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "see_moves.parquet + position_tactical.parquet (resumen)", "module": "see.py",
     "description": "Static Exchange Evaluation por captura legal, algoritmo swap-list estandar."},
    {"id": 14, "name": "threats", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_tactical.parquet", "module": "threats.py",
     "description": "Mate en 1, pieza mayor colgada, amenaza de promocion, jaque/captura forzante disponible."},
    {"id": 15, "name": "tactical_motifs", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_tactical.parquet", "module": "tactical_motifs.py",
     "description": "Fork, discovered attack/check, double check, back rank weakness (consolida #10-12)."},
    {"id": 16, "name": "space", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_strategic.parquet", "module": "space.py",
     "description": "Casillas controladas en campo rival + centro extendido."},
    {"id": 17, "name": "piece_activity", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_strategic.parquet", "module": "piece_activity.py",
     "description": "Torres (open/semi-open/7ma/conectadas), alfiles (par/malos/movilidad), caballos (centralizacion/rim), dama."},
    {"id": 18, "name": "center_structure", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "position_strategic.parquet", "module": "center_structure.py",
     "description": "Clasificacion open/closed/locked/semi_open/mobile del centro d/e."},
    {"id": 19, "name": "eval_trajectory", "source": ENGINE_DERIVED, "input_safe": False,
     "table": "engine_derived.parquet", "module": "eval_trajectory.py",
     "description": "delta_eval_1/2, rolling change, volatilidad -- SOLO pasado, normalizado a POV blancas."},
    {"id": 20, "name": "policy_entropy", "source": ENGINE_DERIVED, "input_safe": False,
     "table": "engine_derived.parquet", "module": "policy_entropy_margins.py",
     "description": "Entropia de la Soft Policy ya persistida + top1/2/3 probability."},
    {"id": 21, "name": "decision_margins", "source": ENGINE_DERIVED, "input_safe": False,
     "table": "engine_derived.parquet", "module": "policy_entropy_margins.py",
     "description": "Gaps top1/2/3 (cp unificado, mate explicito) + categoria forced/clear/competitive."},
    {"id": 22, "name": "wdl", "source": ENGINE_DERIVED, "input_safe": False,
     "table": "engine_derived.parquet", "module": "wdl.py",
     "description": "win_probability_from_cp (formula existente reutilizada). WDL 3-vias: NOT_IMPLEMENTED_REQUIRES_DEFINITION."},
    {"id": 23, "name": "terminal_distance", "source": FUTURE_DEPENDENT, "input_safe": False,
     "table": "engine_derived.parquet", "module": "terminal_distance.py",
     "description": "plies_to_terminal -- FUTURE_DEPENDENT SIEMPRE, solo curriculum/sampling/analisis."},
    {"id": 24, "name": "mate_labels", "source": ENGINE_DERIVED, "input_safe": False,
     "table": "engine_derived.parquet", "module": "mate_labels.py",
     "description": "mate_for_side/mate_in_n extraidos de engine_eval ya existente."},
    {"id": 25, "name": "tablebase", "source": ENGINE_DERIVED, "input_safe": False,
     "table": "tablebase.parquet", "module": "tablebase.py",
     "description": "Syzygy WDL/DTZ opcional, sin descargas. tablebase_available=false si no hay SYZYGY_PATH."},
    {"id": 26, "name": "human_consensus", "source": POSITION_AVAILABLE, "input_safe": False,
     "table": "human_consensus.parquet", "module": "human_consensus.py",
     "description": "Distribucion de movimientos humanos por position_id (consensus). input_safe=False porque agrega informacion de OTRAS partidas/jugadores, no solo la posicion actual -- metadata/analysis, no input de inferencia de una sola posicion."},
    {"id": 27, "name": "human_engine_agreement", "source": ENGINE_DERIVED, "input_safe": False,
     "table": "human_consensus.parquet", "module": "human_consensus.py",
     "description": "Acuerdo consenso humano vs motor (Jensen-Shannon)."},
    {"id": 28, "name": "sequences", "source": POSITION_AVAILABLE, "input_safe": True,
     "table": "sequences/", "module": "sequences.py",
     "description": "Historial temporal SOLO pasado, sin cruce entre partidas -- PRIORIDAD MAXIMA."},
]


def build_feature_registry(enrichment_schema_version: str) -> dict:
    return {
        "enrichment_schema_version": enrichment_schema_version,
        "classification_legend": {
            POSITION_AVAILABLE: "conocible solo con la posicion + historial legal previo",
            ENGINE_DERIVED: "proviene de Stockfish ya calculado -- input_safe=False por defecto",
            FUTURE_DEPENDENT: "usa informacion posterior a la posicion -- input_safe=False SIEMPRE",
        },
        "families": FAMILIES,
        "summary": {
            "total_families": len(FAMILIES),
            "position_available": sum(1 for f in FAMILIES if f["source"] == POSITION_AVAILABLE),
            "engine_derived": sum(1 for f in FAMILIES if f["source"] == ENGINE_DERIVED),
            "future_dependent": sum(1 for f in FAMILIES if f["source"] == FUTURE_DEPENDENT),
            "input_safe_count": sum(1 for f in FAMILIES if f["input_safe"]),
            "not_input_safe_count": sum(1 for f in FAMILIES if not f["input_safe"]),
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_feature_registry("tokenization_enrichment_v3.0"), indent=2, ensure_ascii=False))
