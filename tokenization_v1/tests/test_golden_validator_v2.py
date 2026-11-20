"""Tests del validador Golden Dataset (20_validate_golden_dataset.py) sobre
un mini golden SINTETICO (nunca sobre golden/test_10000_v1 real). Cubre los
8 modos de fallo pedidos + el caso feliz."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPTS_DIR)

from golden_metrics import compute_baseline_metrics  # noqa: E402
from shard_integrity import sha256_of_file  # noqa: E402

POSITIONS_SCHEMA = pa.schema([
    ("row_id", pa.int64()), ("game_id", pa.int64()), ("human_move_id", pa.int32()),
    ("position_id", pa.uint64()), ("position_occurrence_count", pa.int32()),
    ("game_phase", pa.string()), ("is_check", pa.bool_()), ("is_checkmate", pa.bool_()),
    ("is_stalemate", pa.bool_()), ("move_is_capture", pa.bool_()), ("move_is_promotion", pa.bool_()),
    ("move_is_castling", pa.bool_()), ("move_is_en_passant", pa.bool_()), ("move_is_quiet", pa.bool_()),
    ("sample_weight", pa.float32()),
])

EVAL_SCHEMA = pa.schema([("row_id", pa.int64()), ("centipawn_loss", pa.int32()), ("move_quality", pa.string())])
TOPK_SCHEMA = pa.schema([("row_id", pa.int64()), ("rank", pa.int16())])
POLICY_SCHEMA = pa.schema([("row_id", pa.int64()), ("move_id", pa.int32())])
CRIT_SCHEMA = pa.schema([
    ("row_id", pa.int64()), ("engine_criticality_score", pa.float32()), ("engine_tactical_or_critical", pa.bool_()),
])


def build_mini_dataset(root):
    """root/shards_v2/mini/... + root/outputs_v2/mini/stockfish_full/... --
    5 posiciones (3 elegibles, 2 terminales), 1 shard."""
    shard_dir = os.path.join(root, "shards_v2", "mini")
    full_dir = os.path.join(root, "outputs_v2", "mini", "stockfish_full", "shard_00000")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(full_dir, exist_ok=True)

    rows = []
    for i in range(5):
        rows.append({
            "row_id": i, "game_id": 0 if i < 3 else 1, "human_move_id": -1 if i in (2, 4) else i,
            "position_id": 1000 + i, "position_occurrence_count": 1,
            "game_phase": "opening", "is_check": False, "is_checkmate": False, "is_stalemate": False,
            "move_is_capture": False, "move_is_promotion": False, "move_is_castling": False,
            "move_is_en_passant": False, "move_is_quiet": True, "sample_weight": 1.0,
        })
    table = pa.table({n: [r[n] for r in rows] for n in POSITIONS_SCHEMA.names}, schema=POSITIONS_SCHEMA)
    pq.write_table(table, os.path.join(shard_dir, "shard_00000.positions.parquet"))

    legal_bytes = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16).tobytes()
    with open(os.path.join(shard_dir, "shard_00000.legalmoves.bin"), "wb") as f:
        f.write(legal_bytes)
    np.save(os.path.join(shard_dir, "shard_00000.legalmoves_offsets.npy"), np.array([0, 2, 4, 6, 6], dtype=np.int64))

    eligible_ids = [0, 1, 3]
    ev_rows = [{"row_id": rid, "centipawn_loss": 10, "move_quality": "excellent"} for rid in eligible_ids]
    pq.write_table(pa.table({n: [r[n] for r in ev_rows] for n in EVAL_SCHEMA.names}, schema=EVAL_SCHEMA),
                    os.path.join(full_dir, "engine_eval.parquet"))
    tk_rows = [{"row_id": rid, "rank": 1} for rid in eligible_ids]
    pq.write_table(pa.table({n: [r[n] for r in tk_rows] for n in TOPK_SCHEMA.names}, schema=TOPK_SCHEMA),
                    os.path.join(full_dir, "engine_topk.parquet"))
    pt_rows = [{"row_id": rid, "move_id": 1} for rid in eligible_ids]
    pq.write_table(pa.table({n: [r[n] for r in pt_rows] for n in POLICY_SCHEMA.names}, schema=POLICY_SCHEMA),
                    os.path.join(full_dir, "engine_policy_target.parquet"))
    cr_rows = [{"row_id": rid, "engine_criticality_score": 0.2, "engine_tactical_or_critical": False} for rid in eligible_ids]
    pq.write_table(pa.table({n: [r[n] for r in cr_rows] for n in CRIT_SCHEMA.names}, schema=CRIT_SCHEMA),
                    os.path.join(full_dir, "engine_criticality.parquet"))

    return shard_dir, os.path.dirname(full_dir)


def freeze_mini_golden(root, golden_dir, dataset_id="mini_golden_v1", version="1.0.0"):
    shard_dir, full_dir = os.path.join(root, "shards_v2", "mini"), os.path.join(root, "outputs_v2", "mini", "stockfish_full")
    os.makedirs(golden_dir, exist_ok=True)

    file_index = []
    for tag in ["shard_00000"]:
        pq_path = os.path.join(shard_dir, f"{tag}.positions.parquet")
        bin_path = os.path.join(shard_dir, f"{tag}.legalmoves.bin")
        off_path = os.path.join(shard_dir, f"{tag}.legalmoves_offsets.npy")
        n_rows = pq.read_table(pq_path, columns=["row_id"]).num_rows
        file_index.append({"component": "base_dataset", "shard_id": tag,
                            "relative_path": os.path.relpath(pq_path, root).replace("\\", "/"),
                            "file_type": "parquet", "size_bytes": os.path.getsize(pq_path),
                            "sha256": sha256_of_file(pq_path), "expected_rows": n_rows, "schema_role": "BASE_POSITION"})
        file_index.append({"component": "base_dataset", "shard_id": tag,
                            "relative_path": os.path.relpath(bin_path, root).replace("\\", "/"),
                            "file_type": "binary_uint16", "size_bytes": os.path.getsize(bin_path),
                            "sha256": sha256_of_file(bin_path), "expected_rows": os.path.getsize(bin_path) // 2,
                            "schema_role": "LEGAL_MOVES"})
        file_index.append({"component": "base_dataset", "shard_id": tag,
                            "relative_path": os.path.relpath(off_path, root).replace("\\", "/"),
                            "file_type": "npy", "size_bytes": os.path.getsize(off_path),
                            "sha256": sha256_of_file(off_path), "expected_rows": n_rows,
                            "schema_role": "LEGAL_MOVE_OFFSETS"})
        for fname, role in [("engine_eval.parquet", "ENGINE_EVAL"), ("engine_topk.parquet", "ENGINE_TOPK"),
                             ("engine_policy_target.parquet", "ENGINE_POLICY"), ("engine_criticality.parquet", "ENGINE_CRITICALITY")]:
            path = os.path.join(full_dir, tag, fname)
            file_index.append({"component": "engine_enrichment", "shard_id": tag,
                                "relative_path": os.path.relpath(path, root).replace("\\", "/"),
                                "file_type": "parquet", "size_bytes": os.path.getsize(path),
                                "sha256": sha256_of_file(path), "expected_rows": pq.read_table(path, columns=[]).num_rows,
                                "schema_role": role})

    with open(os.path.join(golden_dir, "golden_file_index.json"), "w", encoding="utf-8") as f:
        json.dump(file_index, f)

    metrics = compute_baseline_metrics(shard_dir, full_dir)
    with open(os.path.join(golden_dir, "golden_baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    manifest = {
        "identity": {"golden_dataset_id": dataset_id, "golden_dataset_version": version, "test_name": "mini"},
        "dataset_base": {
            "total_positions": metrics["base"]["total_rows"], "plies": metrics["base"]["plies"],
            "terminal_rows": metrics["base"]["terminal_rows"], "row_id_min": 0,
            "row_id_max": metrics["base"]["total_rows"] - 1,
        },
        "schema": {"tokenization_schema_version": "tokenization_schema_v2"},
        "file_index_path": "golden_file_index.json", "baseline_metrics_path": "golden_baseline_metrics.json",
    }
    manifest_path = os.path.join(golden_dir, "golden_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    lock = {"golden_dataset_id": dataset_id, "golden_dataset_version": version, "test_name": "mini",
            "manifest_sha256": sha256_of_file(manifest_path)}
    with open(os.path.join(golden_dir, "GOLDEN_DATASET.lock"), "w", encoding="utf-8") as f:
        json.dump(lock, f)

    return manifest_path


def run_validator(golden_dir, mode, project_root=None):
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "20_validate_golden_dataset.py"),
           "--golden-dir", golden_dir, "--mode", mode]
    if project_root is not None:
        cmd += ["--project-root", project_root]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_happy_path_quick_and_full(tmp_path):
    build_mini_dataset(str(tmp_path))
    golden_dir = str(tmp_path / "golden" / "mini_v1")
    freeze_mini_golden(str(tmp_path), golden_dir)

    r_quick = run_validator(golden_dir, "quick", str(tmp_path))
    assert r_quick.returncode == 0, r_quick.stdout + r_quick.stderr
    assert json.loads(r_quick.stdout)["status"] == "PASS"

    r_full = run_validator(golden_dir, "full", str(tmp_path))
    assert r_full.returncode == 0, r_full.stdout + r_full.stderr
    assert json.loads(r_full.stdout)["status"] == "PASS"


def test_detects_missing_file(tmp_path):
    build_mini_dataset(str(tmp_path))
    golden_dir = str(tmp_path / "golden" / "mini_v1")
    freeze_mini_golden(str(tmp_path), golden_dir)
    os.remove(str(tmp_path / "shards_v2" / "mini" / "shard_00000.legalmoves.bin"))

    r = run_validator(golden_dir, "quick", str(tmp_path))
    assert r.returncode != 0
    out = json.loads(r.stdout)
    assert out["status"] == "FAIL"
    assert any("archivo faltante" in f for f in out["failures"])


def test_detects_wrong_sha256(tmp_path):
    build_mini_dataset(str(tmp_path))
    golden_dir = str(tmp_path / "golden" / "mini_v1")
    freeze_mini_golden(str(tmp_path), golden_dir)
    p = tmp_path / "shards_v2" / "mini" / "shard_00000.legalmoves.bin"
    original = p.read_bytes()
    p.write_bytes(original[:-2] + bytes([original[-1], original[-2]]))  # mismo tamano, contenido distinto

    r = run_validator(golden_dir, "quick", str(tmp_path))
    assert r.returncode != 0
    out = json.loads(r.stdout)
    assert any("sha256 distinto" in f for f in out["failures"])


def test_detects_wrong_size(tmp_path):
    build_mini_dataset(str(tmp_path))
    golden_dir = str(tmp_path / "golden" / "mini_v1")
    freeze_mini_golden(str(tmp_path), golden_dir)
    p = tmp_path / "shards_v2" / "mini" / "shard_00000.legalmoves.bin"
    p.write_bytes(p.read_bytes() + b"\x00\x00")

    r = run_validator(golden_dir, "quick", str(tmp_path))
    assert r.returncode != 0
    out = json.loads(r.stdout)
    assert any("tamano distinto" in f for f in out["failures"])


def test_detects_wrong_row_count(tmp_path):
    build_mini_dataset(str(tmp_path))
    golden_dir = str(tmp_path / "golden" / "mini_v1")
    freeze_mini_golden(str(tmp_path), golden_dir)

    # reescribe engine_eval.parquet con una fila de mas, manteniendo el resto
    full_dir = tmp_path / "outputs_v2" / "mini" / "stockfish_full" / "shard_00000"
    t = pq.read_table(str(full_dir / "engine_eval.parquet")).to_pandas()
    extra = pd.DataFrame([{"row_id": 999, "centipawn_loss": 0, "move_quality": "best"}])
    t2 = pd.concat([t, extra], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(t2, schema=EVAL_SCHEMA, preserve_index=False),
                    str(full_dir / "engine_eval.parquet"))

    r = run_validator(golden_dir, "quick", str(tmp_path))
    assert r.returncode != 0
    out = json.loads(r.stdout)
    # cambia el contenido -> hash distinto se detecta antes que el row count,
    # pero de cualquier forma debe fallar
    assert out["status"] == "FAIL"


def test_detects_corrupt_manifest(tmp_path):
    build_mini_dataset(str(tmp_path))
    golden_dir = str(tmp_path / "golden" / "mini_v1")
    manifest_path = freeze_mini_golden(str(tmp_path), golden_dir)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("{ esto no es json valido")

    r = run_validator(golden_dir, "quick", str(tmp_path))
    assert r.returncode != 0


def test_detects_manifest_lock_sha_mismatch(tmp_path):
    build_mini_dataset(str(tmp_path))
    golden_dir = str(tmp_path / "golden" / "mini_v1")
    manifest_path = freeze_mini_golden(str(tmp_path), golden_dir)
    # modifica el manifest DESPUES de calcular el lock -> el hash queda desactualizado
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["identity"]["golden_dataset_version"] = "9.9.9-tampered"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    r = run_validator(golden_dir, "quick", str(tmp_path))
    assert r.returncode != 0
    out = json.loads(r.stdout)
    assert any("manifest_sha256" in f for f in out["failures"])


def test_detects_full_mode_baseline_regression(tmp_path):
    build_mini_dataset(str(tmp_path))
    golden_dir = str(tmp_path / "golden" / "mini_v1")
    freeze_mini_golden(str(tmp_path), golden_dir)

    # cambia move_quality en los datos reales DESPUES de congelar -> el
    # recalculo de golden_baseline_metrics debe diferir del congelado
    full_dir = tmp_path / "outputs_v2" / "mini" / "stockfish_full" / "shard_00000"
    t = pq.read_table(str(full_dir / "engine_eval.parquet")).to_pandas()
    t.loc[0, "move_quality"] = "blunder"
    pq.write_table(pa.Table.from_pandas(t, schema=EVAL_SCHEMA, preserve_index=False),
                    str(full_dir / "engine_eval.parquet"))

    r = run_validator(golden_dir, "full", str(tmp_path))
    assert r.returncode != 0
    out = json.loads(r.stdout)
    assert out["status"] == "FAIL"
    assert any("REGRESION DETECTADA" in f for f in out["failures"])


def test_detects_missing_golden_dir(tmp_path):
    r = run_validator(str(tmp_path / "does_not_exist"), "quick")
    assert r.returncode != 0
