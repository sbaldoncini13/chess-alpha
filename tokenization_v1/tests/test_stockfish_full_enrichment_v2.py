"""Smoke test end-to-end de 16_stockfish_full_enrichment.py sobre un
piloto SINTETICO muy chico (nunca sobre shards_v2/test_10000 reales) --
valida sharding/checkpoint/resume/consolidacion antes de confiar el script
para la corrida real de 808611 filas."""
import importlib.util
import json
import os
import subprocess
import sys

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPTS_DIR)

from tokenizer import move_to_action_id  # noqa: E402

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "")
requires_stockfish = pytest.mark.skipif(not STOCKFISH_PATH, reason="STOCKFISH_PATH no configurado")

MINI_SCHEMA = pa.schema([
    ("row_id", pa.int64()), ("game_id", pa.int64()), ("ply", pa.int32()),
    ("fen", pa.string()), ("human_move_uci", pa.string()), ("human_move_id", pa.int32()),
    ("game_phase", pa.string()), ("is_check", pa.bool_()), ("tactical_or_critical", pa.bool_()),
])

GAMES = [
    ["e4", "e5", "Nf3", "Nc6"],
    ["d4", "d5", "c4"],
]


def build_mini_shards(shard_dir):
    # shard_dir YA incluye el test_name (p.ej. .../shards_v2/mini) -- el
    # pipeline real espera config["shard_dir"] = raiz SIN test_name y le
    # agrega args.test_name el solo, igual que 06_enrich_positions.py.
    os.makedirs(shard_dir, exist_ok=True)
    row_id = 0
    rows = []
    for gid, sans in enumerate(GAMES):
        board = chess.Board()
        for ply, san in enumerate(sans):
            move = board.parse_san(san)
            rows.append({
                "row_id": row_id, "game_id": gid, "ply": ply, "fen": board.fen(),
                "human_move_uci": move.uci(), "human_move_id": move_to_action_id(move),
                "game_phase": "opening", "is_check": False, "tactical_or_critical": False,
            })
            row_id += 1
            board.push(move)
        # fila terminal (sin movimiento humano)
        rows.append({
            "row_id": row_id, "game_id": gid, "ply": len(sans), "fen": board.fen(),
            "human_move_uci": "", "human_move_id": -1,
            "game_phase": "opening", "is_check": False, "tactical_or_critical": False,
        })
        row_id += 1

    table = pa.table({name: [r[name] for r in rows] for name in MINI_SCHEMA.names}, schema=MINI_SCHEMA)
    pq.write_table(table, os.path.join(shard_dir, "shard_00000.positions.parquet"))
    return len([r for r in rows if r["human_move_id"] != -1])


def write_config(tmp_path, shard_dir):
    # shard_dir = .../shards_v2/mini -> config.shard_dir debe ser la raiz
    # (.../shards_v2), porque el pipeline le agrega args.test_name el solo.
    shard_root = os.path.dirname(shard_dir)
    config = {
        "schema_version": "tokenization_schema_v2", "pipeline_version": "tokv2.0",
        "output_dir": "outputs_v2", "shard_dir": os.path.relpath(shard_root, tmp_path),
        "logs_dir": "logs", "reports_dir": "reports", "golden_dir": "golden",
        "stockfish_path_env_var": "STOCKFISH_PATH",
        "stockfish_depth": 6, "stockfish_multipv": 3,
        "stockfish_soft_policy_temperature_cp": 150.0,
        "quality_thresholds_cp": {"best": 5, "excellent": 20, "good": 60, "inaccuracy": 120, "mistake": 300},
        "engine_criticality": {"gap_saturation_cp": 200, "w_gap": 0.6, "w_policy_concentration": 0.4, "threshold": 0.5},
        "engine_schema_extension_version": "tokenization_schema_v2.1-engine",
        "stockfish_full": {
            "output_subdir": "stockfish_full", "child_eval_multipv": 1, "threads": 1, "hash_mb": 16,
            "batch_size": 2, "max_retries_per_position": 1, "cache_db_filename": "engine_cache.sqlite",
            "cache_enabled": False, "cache_disabled_reason": "test", "n_workers": 1, "n_workers_note": "test",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


def run_pipeline(config_path, test_name, resume=False, cwd=None, allow_golden_overwrite=False):
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "16_stockfish_full_enrichment.py"),
           "--config", config_path, "--test-name", test_name]
    if resume:
        cmd.append("--resume")
    if allow_golden_overwrite:
        cmd.append("--allow-golden-overwrite")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result


@requires_stockfish
def test_full_enrichment_end_to_end_small(tmp_path):
    shard_dir = str(tmp_path / "shards_v2" / "mini")
    n_eligible = build_mini_shards(shard_dir)
    config_path = write_config(tmp_path, shard_dir)

    result = run_pipeline(config_path, "mini", cwd=str(tmp_path))
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    out_dir = tmp_path / "outputs_v2" / "mini" / "stockfish_full" / "shard_00000"
    ev = pq.read_table(str(out_dir / "engine_eval.parquet")).to_pandas()
    assert len(ev) == n_eligible
    assert ev["row_id"].is_unique

    topk = pq.read_table(str(out_dir / "engine_topk.parquet")).to_pandas()
    policy = pq.read_table(str(out_dir / "engine_policy_target.parquet")).to_pandas()
    crit = pq.read_table(str(out_dir / "engine_criticality.parquet")).to_pandas()
    assert len(topk) > 0
    assert len(policy) > 0
    assert len(crit) == n_eligible

    ckpt = json.loads((tmp_path / "outputs_v2" / "mini" / "stockfish_full" / "checkpoint.json").read_text())
    assert ckpt["shards"]["shard_00000"]["status"] == "completed"


@requires_stockfish
def test_full_enrichment_resume_skips_completed_blocks(tmp_path):
    shard_dir = str(tmp_path / "shards_v2" / "mini")
    n_eligible = build_mini_shards(shard_dir)
    config_path = write_config(tmp_path, shard_dir)

    # primera corrida completa (linea base)
    r1 = run_pipeline(config_path, "mini", cwd=str(tmp_path))
    assert r1.returncode == 0

    blocks_dir = tmp_path / "outputs_v2" / "mini" / "stockfish_full" / "shard_00000" / "_blocks"
    n_blocks_before = len(list(blocks_dir.glob("*.engine_eval.parquet")))
    assert n_blocks_before > 0

    # segunda corrida con --resume sobre un run YA completo: no debe re-procesar
    r2 = run_pipeline(config_path, "mini", resume=True, cwd=str(tmp_path))
    assert r2.returncode == 0, f"stdout={r2.stdout}\nstderr={r2.stderr}"
    # el shard ya estaba completo -> se salta sin re-analizar
    ev = pq.read_table(str(tmp_path / "outputs_v2" / "mini" / "stockfish_full" / "shard_00000" / "engine_eval.parquet")).to_pandas()
    assert len(ev) == n_eligible


@requires_stockfish
def test_full_enrichment_rejects_resume_without_checkpoint(tmp_path):
    shard_dir = str(tmp_path / "shards_v2" / "mini")
    build_mini_shards(shard_dir)
    config_path = write_config(tmp_path, shard_dir)
    result = run_pipeline(config_path, "mini", resume=True, cwd=str(tmp_path))
    assert result.returncode != 0


def _import_enrich_module():
    spec = importlib.util.spec_from_file_location(
        "stockfish_full_enrich", os.path.join(SCRIPTS_DIR, "16_stockfish_full_enrichment.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@requires_stockfish
def test_resume_works_when_interrupted_before_any_shard_completed(tmp_path):
    """Regresion del bug real encontrado en produccion: un PermissionError
    transitorio de Windows interrumpio la corrida real de 808611 filas a
    mitad del PRIMER shard -- checkpoint.json todavia no existia (solo se
    escribia tras completar un shard entero), asi que --resume fallaba con
    'no existe checkpoint' pese a haber ~68000 posiciones ya validas en
    block files. Este test recrea exactamente ese estado (checkpoint
    inicial ya escrito, un batch ya en disco, ningun shard completo) y
    confirma que --resume ahora si continua correctamente."""
    shard_dir = str(tmp_path / "shards_v2" / "mini")
    n_eligible = build_mini_shards(shard_dir)
    config_path = write_config(tmp_path, shard_dir)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    mod = _import_enrich_module()
    shard_root = os.path.join(tmp_path, config["shard_dir"], "mini")
    eligible_by_shard = mod.load_eligible_by_shard(shard_root)
    out_dir = os.path.join(tmp_path, config["output_dir"], "mini", config["stockfish_full"]["output_subdir"])
    os.makedirs(out_dir, exist_ok=True)

    class DummyLogger:
        def log(self, msg):
            pass

    logger = DummyLogger()
    engine = mod.RestartableEngine(
        STOCKFISH_PATH, config["stockfish_depth"], config["stockfish_multipv"],
        config["stockfish_full"]["max_retries_per_position"], logger,
    )
    # procesa SOLO el primer batch a mano (simula "se corto justo despues
    # del primer batch, antes de terminar el shard")
    df = eligible_by_shard["shard_00000"]
    blocks_dir = os.path.join(out_dir, "shard_00000", "_blocks")
    os.makedirs(blocks_dir, exist_ok=True)
    batch = df.iloc[:2]
    eval_rows, topk_rows, policy_rows, crit_rows = [], [], [], []
    for _, r in batch.iterrows():
        result, exc = engine.evaluate(r["fen"], r["human_move_uci"], config["quality_thresholds_cp"])
        assert result is not None, exc
        eval_rows.append(mod.eval_row_from_result(int(r["row_id"]), result, config["stockfish_depth"], config["stockfish_multipv"]))
        t_rows = mod.topk_rows_from_analysis(int(r["row_id"]), result, config["stockfish_depth"])
        topk_rows.extend(t_rows)
        p_rows = mod.policy_rows_from_topk(int(r["row_id"]), result["topk"], config["stockfish_soft_policy_temperature_cp"])
        policy_rows.extend(p_rows)
        crit_rows.append(mod.engine_criticality_row(int(r["row_id"]), t_rows, p_rows, config["engine_criticality"]))
    engine.close()

    paths = mod.block_paths(blocks_dir, 0)
    mod.atomic_write_parquet(paths["eval"], mod.table_from_rows(eval_rows, mod.EVAL_SCHEMA))
    mod.atomic_write_parquet(paths["topk"], mod.table_from_rows(topk_rows, mod.TOPK_SCHEMA))
    mod.atomic_write_parquet(paths["policy"], mod.table_from_rows(policy_rows, mod.POLICY_SCHEMA))
    mod.atomic_write_parquet(paths["crit"], mod.table_from_rows(crit_rows, mod.CRITICALITY_SCHEMA))

    # checkpoint INICIAL (formato nuevo: shards={} , ningun shard completo todavia)
    import hashlib
    engine_binary_sha256 = mod.sha256_of_file(STOCKFISH_PATH)
    config_hash_payload = {
        "depth": config["stockfish_depth"], "multipv": config["stockfish_multipv"],
        "quality_thresholds_cp": config["quality_thresholds_cp"],
        "engine_criticality": config["engine_criticality"],
        "soft_policy_temperature_cp": config["stockfish_soft_policy_temperature_cp"],
        "engine_binary_sha256": engine_binary_sha256,
        "engine_schema_extension_version": config["engine_schema_extension_version"],
        "cache_enabled": config["stockfish_full"]["cache_enabled"],
    }
    config_hash = hashlib.sha256(json.dumps(config_hash_payload, sort_keys=True).encode()).hexdigest()
    ckpt_path = os.path.join(out_dir, "checkpoint.json")
    mod.atomic_write_json(ckpt_path, {
        "run_id": "test-run", "config_hash": config_hash, "test_name": "mini",
        "engine_schema_extension_version": config["engine_schema_extension_version"],
        "cache_enabled": config["stockfish_full"]["cache_enabled"],
        "total_eligible_rows": None, "shards": {}, "elapsed_seconds_total": 0.0,
        "last_checkpoint_utc": "2026-01-01T00:00:00Z", "status": "in_progress",
    })

    # ANTES del fix esto fallaba con "ERROR: --resume pedido pero no existe
    # checkpoint" era imposible llegar aca sin checkpoint -- ahora si existe
    # (formato inicial) y el resume debe completar el shard sin duplicar
    # las filas del batch ya escrito a mano.
    result = run_pipeline(config_path, "mini", resume=True, cwd=str(tmp_path))
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    ev = pq.read_table(os.path.join(out_dir, "shard_00000", "engine_eval.parquet")).to_pandas()
    assert len(ev) == n_eligible
    assert ev["row_id"].is_unique
    assert sorted(ev["row_id"].tolist()) == sorted(df["row_id"].tolist())


@requires_stockfish
def test_golden_lock_blocks_full_enrichment_and_override_unblocks_it(tmp_path):
    shard_dir = str(tmp_path / "shards_v2" / "mini")
    build_mini_shards(shard_dir)
    config_path = write_config(tmp_path, shard_dir)

    golden_lock_dir = tmp_path / "golden" / "mini_v1"
    golden_lock_dir.mkdir(parents=True)
    (golden_lock_dir / "GOLDEN_DATASET.lock").write_text(json.dumps({
        "golden_dataset_id": "mini_golden", "golden_dataset_version": "1.0.0",
        "test_name": "mini", "manifest_sha256": "irrelevant-for-this-test",
    }), encoding="utf-8")

    blocked = run_pipeline(config_path, "mini", cwd=str(tmp_path))
    assert blocked.returncode != 0
    out_dir = tmp_path / "outputs_v2" / "mini" / "stockfish_full"
    assert not out_dir.exists() or len(list(out_dir.glob("shard_*"))) == 0

    allowed = run_pipeline(config_path, "mini", cwd=str(tmp_path), allow_golden_overwrite=True)
    assert allowed.returncode == 0, f"stdout={allowed.stdout}\nstderr={allowed.stderr}"
    assert (out_dir / "shard_00000" / "engine_eval.parquet").exists()
