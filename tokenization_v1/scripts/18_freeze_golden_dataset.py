"""Congelamiento formal del piloto test_10000 como GOLDEN DATASET. SOLO
LECTURA sobre shards_v2/test_10000/, outputs_v2/test_10000/stockfish_full/,
config/ y el propio codigo -- no modifica, no regenera, no duplica datos.

Produce en tokenization_v1/golden/<golden_dataset_id>/:
  golden_manifest.json, golden_file_index.json, golden_config_snapshot.json,
  golden_code_index.json, golden_baseline_metrics.json

El manifest REFERENCIA los artefactos existentes por ruta relativa + hash,
nunca los copia (ver GOLDEN_DATASET.md, principio de no-duplicacion).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_metrics import compute_baseline_metrics  # noqa: E402
from shard_integrity import sha256_of_file  # noqa: E402

GOLDEN_DATASET_ID = "chess_alpha_tokenization_golden_test10000_v1"
GOLDEN_DATASET_VERSION = "1.0.0"
TEST_NAME = "test_10000"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARD_DIR = os.path.join(BASE_DIR, "shards_v2", TEST_NAME)
OUT_DIR = os.path.join(BASE_DIR, "outputs_v2", TEST_NAME)
FULL_DIR = os.path.join(OUT_DIR, "stockfish_full")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "tokenization_v2_config.json")
GOLDEN_DIR = os.path.join(BASE_DIR, "golden", "test_10000_v1")

CODE_INDEX_FILES = [
    ("scripts/game_parser.py", "PGN parser (tokv1.0, reutilizado sin cambios)"),
    ("scripts/tokenizer.py", "vocabulario universal de movimientos (action_id, tokv1.0)"),
    ("scripts/06_enrich_positions.py", "pipeline base v2: reconstruccion de posicion + orquestacion de shards"),
    ("scripts/position_state.py", "reconstruccion de estado / tensor / material / fase / flags"),
    ("scripts/legal_moves.py", "legal move mask"),
    ("scripts/value_target.py", "value target side-to-move"),
    ("scripts/repetition.py", "repeticion dentro de partida"),
    ("scripts/position_dedup.py", "deduplicacion de posiciones (zobrist)"),
    ("scripts/history.py", "referencia de historial de movimientos"),
    ("scripts/sample_weight.py", "infraestructura de sample_weight"),
    ("scripts/difficulty_criticality.py", "difficulty_score / criticality_score base"),
    ("scripts/io_utils_v2.py", "escritura atomica / sharding IO"),
    ("scripts/shard_integrity.py", "checksums SHA-256 de shards"),
    ("scripts/stockfish_enrichment.py", "wrapper de motor: evaluation/best_move/topk/cpl/quality"),
    ("scripts/engine_records.py", "construccion de filas engine_eval/topk/policy"),
    ("scripts/engine_criticality.py", "engine_criticality_score"),
    ("scripts/engine_schemas.py", "schemas parquet compartidos de motor"),
    ("scripts/engine_cache.py", "infraestructura de cache (DESACTIVADA en el run que produjo este golden)"),
    ("scripts/engine_cached_analysis.py", "composicion cache+motor (DESACTIVADA en el run que produjo este golden)"),
    ("scripts/16_stockfish_full_enrichment.py", "orquestador del enriquecimiento Stockfish completo"),
]


def sha256_file_bytes(path):
    return sha256_of_file(path)


def build_base_file_index():
    entries = []
    shard_tags = sorted({f.split(".")[0] for f in os.listdir(SHARD_DIR) if f.endswith(".positions.parquet")})
    for tag in shard_tags:
        pq_path = os.path.join(SHARD_DIR, f"{tag}.positions.parquet")
        bin_path = os.path.join(SHARD_DIR, f"{tag}.legalmoves.bin")
        off_path = os.path.join(SHARD_DIR, f"{tag}.legalmoves_offsets.npy")
        n_rows = pq.read_table(pq_path, columns=["row_id"]).num_rows
        entries.append({
            "component": "base_dataset", "shard_id": tag,
            "relative_path": f"shards_v2/{TEST_NAME}/{tag}.positions.parquet",
            "file_type": "parquet", "size_bytes": os.path.getsize(pq_path),
            "sha256": sha256_file_bytes(pq_path), "expected_rows": n_rows, "schema_role": "BASE_POSITION",
        })
        entries.append({
            "component": "base_dataset", "shard_id": tag,
            "relative_path": f"shards_v2/{TEST_NAME}/{tag}.legalmoves.bin",
            "file_type": "binary_uint16", "size_bytes": os.path.getsize(bin_path),
            "sha256": sha256_file_bytes(bin_path),
            "expected_rows": os.path.getsize(bin_path) // 2, "schema_role": "LEGAL_MOVES",
        })
        entries.append({
            "component": "base_dataset", "shard_id": tag,
            "relative_path": f"shards_v2/{TEST_NAME}/{tag}.legalmoves_offsets.npy",
            "file_type": "npy", "size_bytes": os.path.getsize(off_path),
            "sha256": sha256_file_bytes(off_path), "expected_rows": n_rows, "schema_role": "LEGAL_MOVE_OFFSETS",
        })
    return entries, shard_tags


def build_engine_file_index():
    entries = []
    shard_tags = sorted(
        d for d in os.listdir(FULL_DIR) if os.path.isdir(os.path.join(FULL_DIR, d)) and d.startswith("shard_")
    )
    role_map = {
        "engine_eval.parquet": "ENGINE_EVAL", "engine_topk.parquet": "ENGINE_TOPK",
        "engine_policy_target.parquet": "ENGINE_POLICY", "engine_criticality.parquet": "ENGINE_CRITICALITY",
    }
    for tag in shard_tags:
        sdir = os.path.join(FULL_DIR, tag)
        for fname, role in role_map.items():
            path = os.path.join(sdir, fname)
            n_rows = pq.read_table(path, columns=[]).num_rows if os.path.getsize(path) else 0
            entries.append({
                "component": "engine_enrichment", "shard_id": tag,
                "relative_path": f"outputs_v2/{TEST_NAME}/stockfish_full/{tag}/{fname}",
                "file_type": "parquet", "size_bytes": os.path.getsize(path),
                "sha256": sha256_file_bytes(path), "expected_rows": n_rows, "schema_role": role,
            })
    return entries, shard_tags


def build_reference_file_index():
    refs = [
        (f"outputs_v2/{TEST_NAME}/manifest.json", "MANIFEST"),
        (f"outputs_v2/{TEST_NAME}/integrity_manifest.json", "MANIFEST"),
        (f"outputs_v2/{TEST_NAME}/stockfish_full/checkpoint.json", "CHECKPOINT"),
        (f"outputs_v2/{TEST_NAME}/stockfish_full/stockfish_full_integrity_manifest.json", "MANIFEST"),
        ("config/tokenization_v2_config.json", "CONFIG"),
        ("SCHEMA_V2.md", "OTHER_REFERENCE"),
        (f"reports/{TEST_NAME}_v2_report.json", "REPORT"),
        (f"reports/{TEST_NAME}_cache_audit.json", "REPORT"),
        (f"reports/{TEST_NAME}_cache_validation.json", "REPORT"),
        (f"reports/{TEST_NAME}_stockfish_full_validation.json", "REPORT"),
        (f"reports/{TEST_NAME}_stockfish_full_pre_run_estimation.json", "REPORT"),
    ]
    entries = []
    for rel, role in refs:
        path = os.path.join(BASE_DIR, rel)
        if not os.path.isfile(path):
            continue
        entries.append({
            "component": "reference", "shard_id": None, "relative_path": rel,
            "file_type": os.path.splitext(rel)[1].lstrip("."), "size_bytes": os.path.getsize(path),
            "sha256": sha256_file_bytes(path), "expected_rows": None, "schema_role": role,
        })
    return entries


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    print("Calculando indice de archivos base (27)...")
    base_entries, base_shard_tags = build_base_file_index()
    print("Calculando indice de archivos de motor (36)...")
    engine_entries, engine_shard_tags = build_engine_file_index()
    reference_entries = build_reference_file_index()

    file_index = base_entries + engine_entries + reference_entries
    file_index_path = os.path.join(GOLDEN_DIR, "golden_file_index.json")
    with open(file_index_path, "w", encoding="utf-8") as f:
        json.dump(file_index, f, indent=2, ensure_ascii=False)

    print("Calculando snapshot de configuracion...")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config_content = json.load(f)
    config_snapshot = {
        "golden_dataset_id": GOLDEN_DATASET_ID, "snapshot_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_config_path": "config/tokenization_v2_config.json",
        "source_config_sha256": sha256_file_bytes(CONFIG_PATH),
        "config": config_content,
    }
    with open(os.path.join(GOLDEN_DIR, "golden_config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    print("Calculando indice de codigo critico...")
    code_index = []
    for rel, role in CODE_INDEX_FILES:
        path = os.path.join(BASE_DIR, rel)
        if not os.path.isfile(path):
            code_index.append({"relative_path": rel, "role": role, "status": "MISSING"})
            continue
        code_index.append({
            "relative_path": rel, "role": role, "size_bytes": os.path.getsize(path),
            "sha256": sha256_file_bytes(path),
        })
    is_git_repo = os.path.isdir(os.path.join(os.path.dirname(BASE_DIR), ".git")) or os.path.isdir(os.path.join(BASE_DIR, ".git"))
    code_snapshot = {
        "is_git_repo": is_git_repo,
        "git_note": "El proyecto NO es un repositorio git (verificado con 'git status' -> fatal: not a git repository). Se identifica el codigo exclusivamente por hash SHA-256 de cada archivo critico.",
        "files": code_index,
    }
    with open(os.path.join(GOLDEN_DIR, "golden_code_index.json"), "w", encoding="utf-8") as f:
        json.dump(code_snapshot, f, indent=2, ensure_ascii=False)

    print("Recalculando metricas baseline desde los artefactos actuales (puede tardar)...")
    metrics = compute_baseline_metrics(SHARD_DIR, FULL_DIR)
    with open(os.path.join(GOLDEN_DIR, "golden_baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    manifest = {
        "identity": {
            "golden_dataset_id": GOLDEN_DATASET_ID, "golden_dataset_version": GOLDEN_DATASET_VERSION,
            "creation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "project": "chess-alpha", "module": "tokenization_v1", "dataset_name": "test_10000_v1",
            "test_name": TEST_NAME,
        },
        "dataset_base": {
            "games": metrics["base"]["games"], "plies": metrics["base"]["plies"],
            "terminal_rows": metrics["base"]["terminal_rows"], "total_positions": metrics["base"]["total_rows"],
            "engine_eligible_positions": len([e for e in engine_entries if e["schema_role"] == "ENGINE_EVAL"]) and sum(
                e["expected_rows"] for e in engine_entries if e["schema_role"] == "ENGINE_EVAL"
            ),
            "number_of_base_shards": len(base_shard_tags),
            "row_id_min": 0, "row_id_max": metrics["base"]["total_rows"] - 1,
        },
        "schema": {
            "tokenization_schema_version": "tokenization_schema_v2",
            "engine_schema_extension_version": config_content["engine_schema_extension_version"],
            "move_vocabulary_version": config_content["action_vocab"]["version"],
            "tensor_semantic_version": "board_to_tensor_v2 (23,8,8) -- ver SCHEMA_V2.md, NO persistido para el piloto completo",
            "history_representation_version": "referencia O(1) via history.py (no persistida como columna) -- ver SCHEMA_V2.md",
        },
        "engine": {
            "engine_name": "Stockfish", "depth": config_content["stockfish_depth"],
            "multipv": config_content["stockfish_multipv"], "threads": config_content["stockfish_full"]["threads"],
            "hash_mb": config_content["stockfish_full"]["hash_mb"],
            "soft_policy_temperature_cp": config_content["stockfish_soft_policy_temperature_cp"],
            "engine_criticality_config": config_content["engine_criticality"],
            "move_quality_thresholds_cp": config_content["quality_thresholds_cp"],
            "cache_enabled": config_content["stockfish_full"]["cache_enabled"],
        },
        "coverage": {
            "engine_eval": "FULL_PILOT_APPLIED", "best_move": "FULL_PILOT_APPLIED",
            "top_k": "FULL_PILOT_APPLIED", "soft_policy": "FULL_PILOT_APPLIED",
            "centipawn_loss": "FULL_PILOT_APPLIED", "move_quality": "FULL_PILOT_APPLIED",
            "engine_criticality": "FULL_PILOT_APPLIED",
        },
        "file_counts": {
            "base_files": len(base_entries), "engine_files": len(engine_entries),
            "reference_files": len(reference_entries),
        },
        "file_index_path": "golden_file_index.json",
        "config_snapshot_path": "golden_config_snapshot.json",
        "code_index_path": "golden_code_index.json",
        "baseline_metrics_path": "golden_baseline_metrics.json",
        "scope": {
            "training_performed": False, "search_tree_modified": False,
            "model_modified": False, "full_corpus_processed": False,
        },
        "immutability_note": (
            "Este manifest REFERENCIA los artefactos existentes en shards_v2/test_10000/ y "
            "outputs_v2/test_10000/stockfish_full/ por ruta relativa + SHA-256 -- no los "
            "copia. La inmutabilidad se garantiza verificando estos hashes con "
            "scripts/20_validate_golden_dataset.py, no por duplicacion fisica de datos."
        ),
    }
    manifest_path = os.path.join(GOLDEN_DIR, "golden_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    manifest_sha256 = sha256_file_bytes(manifest_path)
    lock = {
        "golden_dataset_id": GOLDEN_DATASET_ID, "golden_dataset_version": GOLDEN_DATASET_VERSION,
        "test_name": TEST_NAME, "manifest_sha256": manifest_sha256,
        "locked_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "Este archivo protege 'test_10000' contra sobrescritura accidental. "
            "06_enrich_positions.py y 16_stockfish_full_enrichment.py se detienen si "
            "encuentran este lock, salvo --allow-golden-overwrite explicito."
        ),
    }
    with open(os.path.join(GOLDEN_DIR, "GOLDEN_DATASET.lock"), "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "golden_dataset_id": GOLDEN_DATASET_ID, "golden_dataset_version": GOLDEN_DATASET_VERSION,
        "golden_dir": GOLDEN_DIR, "manifest_sha256": manifest_sha256,
        "n_base_files": len(base_entries), "n_engine_files": len(engine_entries),
        "n_reference_files": len(reference_entries),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
