"""Validador READ-ONLY del Golden Dataset. Nunca modifica nada -- ni los
datos referenciados, ni el propio golden_manifest.json.

--quick : archivos existen, tamanos, SHA-256, conteos de filas principales.
--full  : ademas de --quick, recalcula golden_baseline_metrics.json desde
          los datos reales y compara campo por campo (deep diff).

Uso:
  python 20_validate_golden_dataset.py --golden-dir ../golden/test_10000_v1 --mode quick
  python 20_validate_golden_dataset.py --golden-dir ../golden/test_10000_v1 --mode full
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_metrics import compute_baseline_metrics  # noqa: E402
from shard_integrity import sha256_of_file  # noqa: E402

DEFAULT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def check(cond, msg, failures):
    if not cond:
        failures.append(msg)


def deep_diff(expected, actual, path, diffs):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for k in expected:
            if k not in actual:
                diffs.append(f"{path}.{k}: falta en el recalculo (esperado={expected[k]!r})")
                continue
            deep_diff(expected[k], actual[k], f"{path}.{k}", diffs)
        for k in actual:
            if k not in expected:
                diffs.append(f"{path}.{k}: campo nuevo no presente en el baseline congelado (encontrado={actual[k]!r})")
    elif isinstance(expected, float) and isinstance(actual, (int, float)):
        if abs(expected - actual) > 1e-6:
            diffs.append(f"{path}: esperado={expected!r} encontrado={actual!r}")
    else:
        if expected != actual:
            diffs.append(f"{path}: esperado={expected!r} encontrado={actual!r}")


def run_quick(golden_dir, base_dir, failures, info):
    manifest_path = os.path.join(golden_dir, "golden_manifest.json")
    check(os.path.isfile(manifest_path), f"falta {manifest_path}", failures)
    if not os.path.isfile(manifest_path):
        return None, None

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as exc:
        failures.append(f"manifest corrupto (JSON invalido) en {manifest_path}: {exc}")
        return None, None

    lock_path = os.path.join(golden_dir, "GOLDEN_DATASET.lock")
    check(os.path.isfile(lock_path), f"falta {lock_path}", failures)
    if os.path.isfile(lock_path):
        with open(lock_path, encoding="utf-8") as f:
            lock = json.load(f)
        actual_manifest_sha = sha256_of_file(manifest_path)
        check(
            lock.get("manifest_sha256") == actual_manifest_sha,
            f"manifest_sha256 del lock ({lock.get('manifest_sha256')}) no coincide con el manifest "
            f"actual ({actual_manifest_sha}) -- el manifest pudo haber sido regenerado sin actualizar el lock",
            failures,
        )
        check(lock.get("golden_dataset_id") == manifest["identity"]["golden_dataset_id"],
              "golden_dataset_id del lock no coincide con el manifest", failures)

    file_index_path = os.path.join(golden_dir, manifest["file_index_path"])
    check(os.path.isfile(file_index_path), f"falta {file_index_path}", failures)
    with open(file_index_path, encoding="utf-8") as f:
        file_index = json.load(f)

    n_checked = 0
    for entry in file_index:
        path = os.path.join(base_dir, entry["relative_path"])
        if not os.path.isfile(path):
            failures.append(f"archivo faltante: {entry['relative_path']} (role={entry['schema_role']})")
            continue
        actual_size = os.path.getsize(path)
        if actual_size != entry["size_bytes"]:
            failures.append(
                f"tamano distinto en {entry['relative_path']}: esperado={entry['size_bytes']} actual={actual_size}"
            )
            continue
        actual_sha = sha256_of_file(path)
        if actual_sha != entry["sha256"]:
            failures.append(f"sha256 distinto en {entry['relative_path']}")
            continue

        if entry["file_type"] == "parquet" and entry.get("expected_rows") is not None:
            actual_rows = pq.read_table(path, columns=[]).num_rows
            if actual_rows != entry["expected_rows"]:
                failures.append(
                    f"row count distinto en {entry['relative_path']}: esperado={entry['expected_rows']} actual={actual_rows}"
                )
        elif entry["schema_role"] == "LEGAL_MOVE_OFFSETS":
            arr = np.load(path)
            if len(arr) != entry["expected_rows"]:
                failures.append(
                    f"row count distinto en {entry['relative_path']}: esperado={entry['expected_rows']} actual={len(arr)}"
                )
        n_checked += 1

    info["n_files_in_index"] = len(file_index)
    info["n_files_verified_ok"] = n_checked

    db = manifest["dataset_base"]
    check(db["total_positions"] == db["plies"] + db["terminal_rows"],
          "total_positions != plies + terminal_rows en el manifest", failures)
    check(db["row_id_max"] - db["row_id_min"] + 1 == db["total_positions"],
          "row_id_max - row_id_min + 1 != total_positions", failures)

    return manifest, file_index


def run_full(golden_dir, base_dir, manifest, failures, info):
    baseline_path = os.path.join(golden_dir, manifest["baseline_metrics_path"])
    check(os.path.isfile(baseline_path), f"falta {baseline_path}", failures)
    if not os.path.isfile(baseline_path):
        return
    with open(baseline_path, encoding="utf-8") as f:
        expected_metrics = json.load(f)

    shard_dir = os.path.join(base_dir, "shards_v2", manifest["identity"]["test_name"])
    full_dir = os.path.join(base_dir, "outputs_v2", manifest["identity"]["test_name"], "stockfish_full")
    actual_metrics = compute_baseline_metrics(shard_dir, full_dir)

    diffs = []
    deep_diff(expected_metrics, actual_metrics, "golden_baseline_metrics", diffs)
    for d in diffs:
        failures.append(f"REGRESION DETECTADA: {d}")
    info["baseline_metrics_diffs"] = diffs
    info["baseline_metrics_recomputed_matches_frozen"] = len(diffs) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--mode", choices=["quick", "full"], required=True)
    ap.add_argument(
        "--project-root", default=DEFAULT_BASE_DIR,
        help="Raiz contra la que se resuelven las relative_path del file index (por defecto tokenization_v1/). "
             "Solo se necesita cambiar en tests con datasets sinteticos.",
    )
    args = ap.parse_args()

    golden_dir = os.path.abspath(args.golden_dir)
    base_dir = os.path.abspath(args.project_root)
    failures = []
    info = {}

    manifest, file_index = run_quick(golden_dir, base_dir, failures, info)

    if args.mode == "full" and manifest is not None:
        run_full(golden_dir, base_dir, manifest, failures, info)

    result = {
        "golden_dir": golden_dir, "mode": args.mode,
        "golden_dataset_id": manifest["identity"]["golden_dataset_id"] if manifest else None,
        "golden_dataset_version": manifest["identity"]["golden_dataset_version"] if manifest else None,
        "info": info,
        "n_failures": len(failures), "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
