import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

PGN_PATH = DATA_RAW_DIR / "AJ-OTB-PGN-000.pgn"

POSITIONS_PATH = DATA_PROCESSED_DIR / "positions_sample.parquet"
FEATURES_PATH = DATA_PROCESSED_DIR / "positions_features_sample.parquet"
TRANSPOSITIONS_PATH = DATA_PROCESSED_DIR / "transpositions_report.csv"
TRAINING_DATASET_PATH = DATA_PROCESSED_DIR / "training_dataset.parquet"
ENGINE_EVAL_PATH = DATA_PROCESSED_DIR / "engine_eval_sample.parquet"

POLICY_VOCAB_PATH = MODELS_DIR / "policy_vocab.json"
SUPERVISED_MODEL_PATH = MODELS_DIR / "policy_value_net.pt"

# Corrida a mayor escala: partidas de Lichess (2026-06) filtradas por Elo,
# red mas grande (256 canales / 10 bloques). Paths separados de los de
# arriba para no pisar el modelo chico ya entrenado.
LICHESS_PGN_PATH = DATA_RAW_DIR / "lichess_db_standard_rated_2026-06.pgn"
LICHESS_POSITIONS_PATH = DATA_PROCESSED_DIR / "positions_lichess_20k.parquet"
LICHESS_TRAINING_DATASET_PATH = DATA_PROCESSED_DIR / "training_dataset_lichess_20k.parquet"
LICHESS_POLICY_VOCAB_PATH = MODELS_DIR / "policy_vocab_lichess_20k.json"
LICHESS_TENSORS_DIR = DATA_PROCESSED_DIR / "tensors_lichess_20k"
LICHESS_MODEL_PATH = MODELS_DIR / "policy_value_net_lichess_20k.pt"

# Epoch 2: segunda tanda de partidas de Lichess (mismo rango de Elo 1800-2100,
# pero filtradas ademas por tener [%eval] de Stockfish y por que la jugada
# jugada haya mejorado la posicion de quien la hizo). Son partidas nuevas,
# posteriores a las primeras 300.000 ya usadas para el dataset lichess_20k.
EPOCH2_GAMES_PGN_PATH = DATA_RAW_DIR / "lichess_epoch2_games.pgn"
EPOCH2_TRAINING_DATASET_PATH = DATA_PROCESSED_DIR / "training_dataset_lichess_epoch2.parquet"

# Epoch 3: tercera tanda de partidas de Lichess (mismo rango de Elo
# 1800-2100), compuesta 70% partidas nuevas + 20% replay (epoch1+2) + 10%
# ejemplos que arrancan en medio juego. Ver 33_build_epoch3_dataset.py.
EPOCH1_SITE_IDS_PATH = DATA_PROCESSED_DIR / "epoch1_site_ids.json"
EPOCH3_NEW_GAMES_PGN_PATH = DATA_RAW_DIR / "lichess_epoch3_new_games.pgn"
EPOCH3_MIDGAME_GAMES_PGN_PATH = DATA_RAW_DIR / "lichess_epoch3_midgame_games.pgn"
EPOCH3_NEW_DATASET_PATH = DATA_PROCESSED_DIR / "training_dataset_lichess_epoch3_new.parquet"
EPOCH3_MIDGAME_DATASET_PATH = DATA_PROCESSED_DIR / "training_dataset_lichess_epoch3_midgame.parquet"
EPOCH3_REPLAY_DATASET_PATH = DATA_PROCESSED_DIR / "training_dataset_lichess_epoch3_replay.parquet"
EPOCH3_COMBINED_DATASET_PATH = DATA_PROCESSED_DIR / "training_dataset_lichess_epoch3_combined.parquet"
EPOCH3_DELTA_MOVER_DATASET_PATH = DATA_PROCESSED_DIR / "training_dataset_lichess_epoch3_delta_mover.parquet"
EPOCH3_PROGRESS_PATH = MODELS_DIR / "epoch3_progress.json"
EPOCH3_REPORT_PATH = MODELS_DIR / "epoch3_dataset_report.json"
EPOCH3_CHECKPOINT_PATH = DATA_PROCESSED_DIR / "epoch3_scan_checkpoint.pkl"

# Bloque 4: Arena de validacion por partidas completas (finalistas del
# Bloque 3 vs el motor baseline pre-Bloque-3).
ARENA_DIR = DATA_PROCESSED_DIR / "arena_block4"
ARENA_GAMES_PATH = ARENA_DIR / "arena_games.parquet"
ARENA_PLIES_PATH = ARENA_DIR / "arena_plies.parquet"
ARENA_CPL_PATH = ARENA_DIR / "arena_cpl_sample.parquet"
ARENA_PROGRESS_PATH = ARENA_DIR / "arena_progress.json"
ARENA_SUMMARY_PATH = ARENA_DIR / "arena_summary.json"
ARENA_ERRORS_PATH = ARENA_DIR / "arena_errors.jsonl"

# v1.1: continuacion del entrenamiento del hibrido desde el checkpoint de
# epoch 5, con dataset correctivo/tactico y mezcla controlada por batch.
# Ver src/configs/config_v11.yaml y src/35_build_corrective_dataset.py.
V11_DIR = DATA_PROCESSED_DIR / "v1_1"
V11_CORRECTIVE_DATASET_PATH = V11_DIR / "corrective_dataset.parquet"
V11_TACTICAL_DATASET_PATH = V11_DIR / "tactical_dataset.parquet"
V11_VALIDATION_SET_PATH = V11_DIR / "validation_set.parquet"
V11_MODELS_DIR = MODELS_DIR / "v1_1"
V11_LOGS_DIR = V11_DIR / "logs"

# Evaluacion decisiva v1.1 (CPL/blunders + Arena checkpoint vs checkpoint,
# post plan de 7 puntos). Directorios propios, separados de arena_block4/ y
# de models/benchmark_results/checkpoint_comparison/ para no pisar nada de
# lo ya corrido (N=100/30 original, Bloque 4).
V11_EVAL_DIR = DATA_PROCESSED_DIR / "v1_1_eval"
V11_EVAL_MANIFEST_PATH = V11_EVAL_DIR / "positions_manifest.json"
V11_EVAL_ARENA_DIR = V11_EVAL_DIR / "arena"
V11_EVAL_MODELS_DIR = MODELS_DIR / "benchmark_results" / "v1_1_eval"

# Reportes generales (no especificos de v1.1) - directorio raiz para
# entregables de lectura humana que no encajan en reports/v1_1/.
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_V11_DIR = REPORTS_DIR / "v1_1"

# Manifests: referencias livianas a subconjuntos de datos ya existentes
# (game_id, hash, metadata) SIN duplicar el contenido pesado (movetext,
# posiciones) - para filtros/auditorias que no deben alterar la fuente.
MANIFESTS_DIR = ROOT_DIR / "data" / "manifests"

# Auditoria de partidas de maquinas (CCRL) previa a un futuro Dataset Epoch 4.
# CCRL = Computer Chess Rating Lists, partidas motor-vs-motor (no humanas) -
# confirmado por headers (Event="CCRL 40/15", White/Black son nombres de
# motores, comentarios con eval+profundidad+tiempo tipo "{+0.60/16 193s}").
CCRL_PGN_PATH = DATA_RAW_DIR / "CCRL-4040-commented.[2389233].pgn"
CCRL_RAW_MANIFEST_PATH = DATA_PROCESSED_DIR / "machine_games_ccrl_raw_manifest.parquet"
MACHINE_GAMES_MANIFEST_1800_2100_PATH = MANIFESTS_DIR / "machine_games_1800_2100_manifest.parquet"

# Auditoria de partidas HUMANAS de Lichess (base completa, no la usada para
# Epoch1/2/3) previa a un futuro Dataset Epoch 4 - alternativa a CCRL.
# Directorio de checkpoints resumibles del pase unico sobre el PGN de 201GB.
HUMAN_AUDIT_CHECKPOINT_DIR = DATA_PROCESSED_DIR / "human_lichess_audit_checkpoints"

# Ruta al binario de Stockfish. No viene instalado por defecto: configurar via
# variable de entorno STOCKFISH_PATH o editar este valor directamente.
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "")

# Modelo operativo seleccionado (promocion post-Arena, no un checkpoint de
# entrenamiento mas). Copia estable, desacoplada del path de la rama de
# entrenamiento que la origino - ver models/selected/selected_model_manifest.md
# para el motivo de seleccion y la evidencia (Arena confirmatoria de 500).
# SELECTED_MODEL_ROLLBACK_PATH es el checkpoint operativo anterior, conservado
# intacto a proposito para poder revertir sin re-entrenar ni re-copiar nada.
SELECTED_MODEL_DIR = MODELS_DIR / "selected"
SELECTED_MODEL_PATH = SELECTED_MODEL_DIR / "policy_value_hybrid_selected_epoch4.pt"
SELECTED_MODEL_ROLLBACK_PATH = V11_MODELS_DIR / "policy_value_hybrid_v1_1_selected_control_epoch02.pt"

# Bibliografia de ajedrez (libros/repertorios/boletines/papers) - carpeta fuente
# no estructurada, solo lectura. Todo lo derivado (inventario, conocimiento
# extraido, datasets candidatos) vive en subcarpetas propias, nunca se escribe
# dentro de la carpeta fuente.
CHESS_BIBLIOGRAPHY_SOURCE_DIR = ROOT_DIR / "bibliografia ajedrez"
CHESS_BIBLIOGRAPHY_DIR = DATA_PROCESSED_DIR.parent / "chess_bibliography"
CHESS_BIBLIOGRAPHY_INVENTORY_DIR = CHESS_BIBLIOGRAPHY_DIR / "inventory"
CHESS_BIBLIOGRAPHY_TEXT_CACHE_DIR = CHESS_BIBLIOGRAPHY_DIR / "text_cache"
CHESS_BIBLIOGRAPHY_OPENINGS_DIR = CHESS_BIBLIOGRAPHY_DIR / "openings"
CHESS_BIBLIOGRAPHY_TRAINING_DIR = CHESS_BIBLIOGRAPHY_DIR / "training"
CHESS_BIBLIOGRAPHY_PAPERS_DIR = CHESS_BIBLIOGRAPHY_DIR / "papers"
CHESS_BIBLIOGRAPHY_PROGRESS_DIR = CHESS_BIBLIOGRAPHY_DIR / "progress"
REPORTS_CHESS_BIBLIOGRAPHY_DIR = REPORTS_DIR / "chess_bibliography"

# SEARCH TREE DATASET V1.1: infraestructura de datos (40.000 posiciones de
# partidas humanas nuevas, Elo 2200-2300 ambos jugadores) para desarrollar
# despues un buscador PVS/Negamax+Alpha-Beta+TT+quiescence+DFPN+Policy
# ordering. Fase exclusivamente de datos - no implementa el buscador
# operativo. Ver reports/search_tree_v1/*.md para el diseno completo.
SEARCH_TREE_V1_DIR = DATA_PROCESSED_DIR / "search_tree_v1"
SEARCH_TREE_V1_CANDIDATES_DIR = SEARCH_TREE_V1_DIR / "candidates"
SEARCH_TREE_V1_CHECKPOINTS_DIR = SEARCH_TREE_V1_DIR / "checkpoints"
SEARCH_TREE_V1_FINAL_DIR = SEARCH_TREE_V1_DIR / "final"
SEARCH_TREE_V1_LEVEL_C_DIR = SEARCH_TREE_V1_DIR / "level_c"
REPORTS_SEARCH_TREE_V1_DIR = REPORTS_DIR / "search_tree_v1"
TESTS_REGRESSION_DIR = ROOT_DIR / "tests" / "regression"
LOGS_SEARCH_TREE_V1_DIR = ROOT_DIR / "logs" / "search_tree_v1"

for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, ARENA_DIR, V11_DIR, V11_MODELS_DIR, V11_LOGS_DIR,
          V11_EVAL_DIR, V11_EVAL_ARENA_DIR, V11_EVAL_MODELS_DIR, REPORTS_V11_DIR, REPORTS_DIR, MANIFESTS_DIR,
          HUMAN_AUDIT_CHECKPOINT_DIR, SELECTED_MODEL_DIR,
          CHESS_BIBLIOGRAPHY_DIR, CHESS_BIBLIOGRAPHY_INVENTORY_DIR, CHESS_BIBLIOGRAPHY_TEXT_CACHE_DIR,
          CHESS_BIBLIOGRAPHY_OPENINGS_DIR, CHESS_BIBLIOGRAPHY_TRAINING_DIR, CHESS_BIBLIOGRAPHY_PAPERS_DIR,
          CHESS_BIBLIOGRAPHY_PROGRESS_DIR, REPORTS_CHESS_BIBLIOGRAPHY_DIR,
          SEARCH_TREE_V1_DIR, SEARCH_TREE_V1_CANDIDATES_DIR, SEARCH_TREE_V1_CHECKPOINTS_DIR,
          SEARCH_TREE_V1_FINAL_DIR, SEARCH_TREE_V1_LEVEL_C_DIR, REPORTS_SEARCH_TREE_V1_DIR,
          TESTS_REGRESSION_DIR, LOGS_SEARCH_TREE_V1_DIR):
    d.mkdir(parents=True, exist_ok=True)
