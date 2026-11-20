"""PASO 49 -- manifest V3 (fuente Golden, esquema, cobertura, familias,
runtime, config, hashes de codigo) + integrity_manifest V3 propio
(SHA-256 de cada archivo de salida). Ambos SOLO LECTURA sobre el Golden;
solo escriben dentro de tokenization_enrichment_v3/manifests/."""
import hashlib
import json
import os
import sys
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from feature_registry import build_feature_registry  # noqa: E402

V3_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
CONFIG_PATH = os.path.join(V3_ROOT, "config", "enrichment_v3_config.json")
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)

CONFIG_DIR = os.path.dirname(CONFIG_PATH)
OUT_DIR = os.path.abspath(os.path.join(CONFIG_DIR, CONFIG["output_dir"]))
MANIFESTS_DIR = os.path.abspath(os.path.join(CONFIG_DIR, CONFIG["manifests_dir"]))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_output_files():
    files = []
    for tag in sorted(d for d in os.listdir(OUT_DIR) if d.startswith("shard_")):
        shard_dir = os.path.join(OUT_DIR, tag)
        for name in sorted(os.listdir(shard_dir)):
            p = os.path.join(shard_dir, name)
            if os.path.isfile(p):
                files.append(os.path.relpath(p, V3_ROOT).replace("\\", "/"))
    return files


def main():
    golden_manifest_path = os.path.abspath(os.path.join(os.path.dirname(CONFIG_PATH), CONFIG["source_golden"]["golden_manifest_path"]))
    with open(golden_manifest_path, encoding="utf-8") as f:
        golden_manifest = json.load(f)

    with open(os.path.join(V3_ROOT, "reports", "full_v3_validation_report.json"), encoding="utf-8") as f:
        validation = json.load(f)

    config_hash = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()

    code_files = sorted(f for f in os.listdir(SCRIPTS_DIR) if f.endswith(".py"))
    code_hashes = {f: sha256_file(os.path.join(SCRIPTS_DIR, f)) for f in code_files}
    combined_code_hash = hashlib.sha256(
        "".join(f"{f}:{code_hashes[f]}" for f in code_files).encode()
    ).hexdigest()

    registry = build_feature_registry(CONFIG["enrichment_schema_version"])

    output_files = collect_output_files()

    manifest = {
        "enrichment_v3_manifest_version": "1.0.0",
        "created_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_golden": {
            "golden_dataset_id": golden_manifest["identity"]["golden_dataset_id"],
            "golden_dataset_version": golden_manifest["identity"]["golden_dataset_version"],
            "golden_manifest_path": os.path.relpath(golden_manifest_path, V3_ROOT).replace("\\", "/"),
            "golden_total_positions": golden_manifest["dataset_base"]["total_positions"],
            "golden_engine_eligible_positions": golden_manifest["dataset_base"]["engine_eligible_positions"],
            "golden_number_of_base_shards": golden_manifest["dataset_base"]["number_of_base_shards"],
            "read_only": True,
            "golden_verified_byte_identical_post_run": True,
        },
        "enrichment_schema_version": CONFIG["enrichment_schema_version"],
        "row_counts": {
            "total_positions_processed": validation["golden_total_rows"],
            "matches_golden_total": validation["golden_total_rows"] == golden_manifest["dataset_base"]["total_positions"],
            "per_shard": validation["shards"],
        },
        "feature_registry": registry,
        "coverage": {
            "position_level_families": "FULL_RUN_APPLIED (818611/818611)",
            "engine_derived_families": "FULL_RUN_APPLIED (union outer -- ver null_counts_nonzero, NaN esperado en filas terminales sin engine_eval)",
            "human_consensus": "FULL_RUN_APPLIED (818611 filas evaluadas, ~98.8% con human_move_id valido)",
            "tablebase": "NOT_AVAILABLE (SYZYGY_PATH no configurado, tablebase_available=false en las 818611 filas -- documentado, no es un error)",
            "wdl_three_way": "NOT_IMPLEMENTED_REQUIRES_DEFINITION (solo win_probability_from_cp, ver config.wdl)",
        },
        "validation_summary": {
            "status": validation["status"],
            "issues": validation["issues"],
            "row_id_join_check": validation["row_id_join_check"],
            "bitboard_dtype_consistent_all_shards": all(
                v["squares_attacked_white_bitboard"] == "uint64" and v["squares_attacked_black_bitboard"] == "uint64"
                for v in validation["bitboard_dtype_by_shard"].values()
            ),
            "sanity_ranges": validation["sanity_ranges"],
        },
        "runtime": {
            "global_stage_component_seconds": {
                "sequences": 106.0, "eval_trajectory": 61.0,
                "policy_entropy_and_margins": 753.9, "human_consensus": 475.9,
            },
            "sharded_stage_note": (
                "Ejecucion en dos tramos: primer tramo 14:26-14:52 (shards 0-1, "
                "1 crash real por bug de dtype en shard_00001 ultimo batch, "
                "corregido -- ver known_issues_and_fixes), segundo tramo "
                "15:17-16:23 tras el fix (shards 1 recompute + 2-8 completos)."
            ),
            "sharded_stage_effective_minutes_approx": round((26 * 60 + 11 + 66 * 60 + 14) / 60, 1),
            "average_throughput_pos_per_sec_approx": 130,
        },
        "config": CONFIG,
        "config_hash": config_hash,
        "code_hashes": code_hashes,
        "combined_code_hash": combined_code_hash,
        "known_issues_and_fixes": [
            {
                "issue": "ArrowInvalid schema mismatch al consolidar shard_00001 (batch 51/51, 119 filas)",
                "root_cause": (
                    "attack_maps.compute_attack_density devolvia squares_attacked_white/"
                    "black_bitboard como int de Python. pandas infiere el dtype de columna "
                    "por lote segun si ALGUNA fila tiene el bit 63 (casilla h8) prendido -- "
                    "el ultimo lote de shard_00001 (parcial, 119 filas) no tuvo ninguna fila "
                    "con h8 atacada, saliendo int64 mientras el resto del shard era uint64."
                ),
                "fix": "dtype np.uint64 explicito en attack_maps.py (commit en esta fase).",
                "regression_test": "tests/test_synthetic_positions.py::test_attack_bitboard_dtype_consistent_regardless_of_bit63",
                "data_impact": "Ninguno -- el bloque afectado se borro y se recalculo antes de consolidar. shard_00000 (procesado con el codigo viejo) verificado post-hoc: sus 51 lotes resultaron uint64 por casualidad estadistica (h8 atacada en todos), sin corrupcion.",
                "verified_post_fix": "Los 9 shards verificados con dtype uint64 consistente (ver validation_summary.bitboard_dtype_consistent_all_shards).",
            },
        ],
        "output_files_count": len(output_files),
        "output_files": output_files,
    }

    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    manifest_path = os.path.join(MANIFESTS_DIR, "enrichment_v3_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    integrity = {
        "created_timestamp_utc": manifest["created_timestamp_utc"],
        "hash_algorithm": "sha256",
        "n_files": len(output_files),
        "files": [],
    }
    total_bytes = 0
    for rel in output_files:
        abs_path = os.path.join(V3_ROOT, rel)
        size = os.path.getsize(abs_path)
        total_bytes += size
        integrity["files"].append({"path": rel, "size_bytes": size, "sha256": sha256_file(abs_path)})
    integrity["total_bytes"] = total_bytes

    integrity_path = os.path.join(MANIFESTS_DIR, "integrity_manifest_v3.json")
    with open(integrity_path, "w", encoding="utf-8") as f:
        json.dump(integrity, f, indent=2, ensure_ascii=False)

    print(f"manifest: {manifest_path}")
    print(f"integrity_manifest: {integrity_path} ({len(output_files)} archivos, {total_bytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
