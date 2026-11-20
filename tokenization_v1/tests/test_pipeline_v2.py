"""PASO: serializacion/deserializacion y resume de procesamiento, sobre el
pipeline real 06_enrich_positions.py (ejecutado como subprocess, igual que
en produccion) y sobre un dataset base v1 SINTETICO construido aca (nunca se
toca test_10000/full_1000000 reales para estas pruebas)."""
import importlib.util
import json
import os
import subprocess
import sys

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPTS_DIR)

from tokenizer import move_to_action_id  # noqa: E402

META_SCHEMA = pa.schema([
    ("sample_id", pa.int64()), ("game_index", pa.int64()), ("draw_rank", pa.int64()),
    ("byte_start", pa.int64()), ("byte_end", pa.int64()), ("ply_count", pa.int32()),
    ("result_code", pa.int8()), ("white_elo", pa.int32()), ("black_elo", pa.int32()),
    ("eco", pa.string()), ("event", pa.string()), ("date", pa.string()),
    ("initial_fen", pa.string()), ("variant", pa.string()), ("identity_hash", pa.string()),
])

RESULT_CODES = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

GAMES = [
    (["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"], "1-0"),
    (["d4", "d5", "c4", "e6"], "1/2-1/2"),
    (["f3", "e5", "g4", "Qh4#"], "0-1"),  # jaque mate
    (["e4", "c5", "Nf3", "d6", "d4", "cxd4"], "1-0"),
]


def _import_enrich_module():
    spec = importlib.util.spec_from_file_location(
        "enrich_v2", os.path.join(SCRIPTS_DIR, "06_enrich_positions.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_synthetic_v1_outputs(base_dir: str, games=GAMES):
    """Construye outputs/<test>/ de tokv1.0 SINTETICO (mismo formato exacto
    que 04_run_tokenization.py produce) sin invocar el pipeline v1 real."""
    all_action_ids = []
    game_offsets = []
    game_lengths = []
    results = []
    meta_rows = []

    offset = 0
    for gid, (sans, result_str) in enumerate(games):
        board = chess.Board()
        ids = []
        for san in sans:
            move = board.parse_san(san)
            ids.append(move_to_action_id(move))
            board.push(move)
        all_action_ids.extend(ids)
        game_offsets.append(offset)
        game_lengths.append(len(ids))
        offset += len(ids)
        results.append(RESULT_CODES[result_str])
        meta_rows.append({
            "sample_id": gid, "game_index": gid, "draw_rank": gid,
            "byte_start": 0, "byte_end": 0, "ply_count": len(ids),
            "result_code": RESULT_CODES[result_str], "white_elo": 2000 + gid, "black_elo": 1900 + gid,
            "eco": "C00", "event": "synthetic", "date": "2026.01.01",
            "initial_fen": chess.STARTING_FEN, "variant": "standard",
            "identity_hash": f"synthetic-{gid}",
        })

    os.makedirs(base_dir, exist_ok=True)
    np.array(all_action_ids, dtype=np.uint16).tofile(os.path.join(base_dir, "moves_uint16.bin"))
    np.save(os.path.join(base_dir, "game_offsets.npy"), np.array(game_offsets, dtype=np.int64))
    np.save(os.path.join(base_dir, "game_lengths.npy"), np.array(game_lengths, dtype=np.uint16))
    np.save(os.path.join(base_dir, "results.npy"), np.array(results, dtype=np.int8))
    table = pa.table({name: [row[name] for row in meta_rows] for name in META_SCHEMA.names}, schema=META_SCHEMA)
    pq.write_table(table, os.path.join(base_dir, "metadata.parquet"))

    return sum(game_lengths) + len(games)  # total de posiciones esperadas (ply_count+1 por partida)


def write_config(tmp_path, v1_outputs_root):
    config = {
        "schema_version": "tokenization_schema_v2", "pipeline_version": "tokv2.0", "seed": 1,
        "base_dataset": {"v1_outputs_dir": os.path.relpath(v1_outputs_root, tmp_path)},
        "output_dir": "outputs_v2", "shard_dir": "shards_v2", "checkpoint_dir": "checkpoints_v2",
        "reports_dir": "reports", "logs_dir": "logs", "golden_dir": "golden",
        "shard_size_positions": 3,  # deliberadamente chico: fuerza varios shards con pocas partidas
        "difficulty_weights": {
            "w_inverse_legal_moves": 0.35, "w_is_check": 0.15, "w_forced_move": 0.30, "w_phase_endgame_bonus": 0.20,
        },
        "criticality_thresholds": {
            "forced_move_max_legal_moves": 1, "low_branching_max_legal_moves": 3, "big_eval_swing_cp": 150,
        },
        "log_every_n_games": 1,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


def run_pipeline(config_path, test_name, resume=False, cwd=None, allow_golden_overwrite=False):
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "06_enrich_positions.py"),
           "--config", config_path, "--test-name", test_name]
    if resume:
        cmd.append("--resume")
    if allow_golden_overwrite:
        cmd.append("--allow-golden-overwrite")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    return result


def read_all_shards(shard_dir, test_name):
    d = os.path.join(shard_dir, test_name)
    files = sorted(f for f in os.listdir(d) if f.endswith(".positions.parquet"))
    tables = [pq.read_table(os.path.join(d, f)) for f in files]
    return pa.concat_tables(tables).to_pandas(), files


def test_serialization_round_trip(tmp_path):
    v1_dir = str(tmp_path / "outputs" / "test_synth")
    expected_total = build_synthetic_v1_outputs(v1_dir)
    config_path = write_config(tmp_path, str(tmp_path / "outputs"))

    run_pipeline(config_path, "test_synth", cwd=str(tmp_path))

    df, files = read_all_shards(str(tmp_path / "shards_v2"), "test_synth")
    assert len(df) == expected_total
    assert sorted(df["row_id"].tolist()) == list(range(expected_total))

    # round-trip: reconstruir el board desde el FEN guardado y confirmar que
    # coincide con el estado real de la partida en ese ply
    for gid, (sans, result_str) in enumerate(GAMES):
        board = chess.Board()
        rows_g = df[df.game_id == gid].sort_values("ply")
        assert len(rows_g) == len(sans) + 1
        for i, san in enumerate(sans):
            row = rows_g.iloc[i]
            assert row["fen"] == board.fen()
            move = board.parse_san(san)
            assert row["human_move_id"] == move_to_action_id(move)
            board.push(move)
        final_row = rows_g.iloc[-1]
        assert final_row["fen"] == board.fen()
        assert final_row["human_move_id"] == -1

    # partida 2 termina en jaque mate: la fila final debe reflejarlo
    mate_rows = df[df.game_id == 2].sort_values("ply")
    assert bool(mate_rows.iloc[-1]["is_checkmate"]) is True


def test_resume_processes_remaining_games_without_duplicating(tmp_path):
    v1_dir = str(tmp_path / "outputs" / "test_synth")
    expected_total = build_synthetic_v1_outputs(v1_dir)

    # Corrida completa de referencia (sin interrupcion), en un directorio aparte
    (tmp_path / "baseline_run").mkdir()
    baseline_cfg_path = write_config(tmp_path / "baseline_run", str(tmp_path / "outputs"))
    run_pipeline(baseline_cfg_path, "test_synth", cwd=str(tmp_path / "baseline_run"))
    df_baseline, _ = read_all_shards(str(tmp_path / "baseline_run" / "shards_v2"), "test_synth")
    assert len(df_baseline) == expected_total

    # Corrida "interrumpida": procesamos manualmente solo la partida 0,
    # escribimos su shard + checkpoint a mano (simula una interrupcion real
    # despues del primer shard consolidado), despues resumimos con --resume.
    resume_root = tmp_path / "resume_run"
    resume_root.mkdir()
    resume_cfg_path = write_config(resume_root, str(tmp_path / "outputs"))

    mod = _import_enrich_module()
    out_dir = resume_root / "outputs_v2" / "test_synth"
    shard_dir = resume_root / "shards_v2" / "test_synth"
    ckpt_dir = resume_root / "checkpoints_v2" / "test_synth"
    for d in (out_dir, shard_dir, ckpt_dir):
        d.mkdir(parents=True)

    class DummyLogger:
        def log(self, msg):
            pass

    logger = DummyLogger()
    dataset = mod.V1BaseDataset(v1_dir)
    counter = mod.pass_a_global_occurrence(dataset, logger)
    rows, legal_ids = mod.build_rows_for_game(0, dataset, counter, 0, {
        "w_inverse_legal_moves": 0.35, "w_is_check": 0.15, "w_forced_move": 0.30, "w_phase_endgame_bonus": 0.20,
    }, {"forced_move_max_legal_moves": 1, "low_branching_max_legal_moves": 3, "big_eval_swing_cp": 150})
    mod.flush_shard(str(shard_dir), 0, rows, legal_ids, logger)

    checkpoint = {
        "test_name": "test_synth", "schema_version": "tokenization_schema_v2", "pipeline_version": "tokv2.0",
        "v1_source_test_name": "test_synth", "next_game_id": 1, "next_row_id": len(rows),
        "next_shard_index": 1, "n_positions_consolidated": len(rows), "n_games_consolidated": 1,
        "elapsed_seconds_total": 1.0, "first_started_utc": "2026-01-01T00:00:00Z",
        "last_checkpoint_utc": "2026-01-01T00:00:01Z", "ram_max_mb": 10.0,
        "status": "stopped_safely", "stop_reason": "solicitud_usuario",
    }
    with open(ckpt_dir / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)

    shard_before = (shard_dir / "shard_00000.positions.parquet").read_bytes()

    run_pipeline(resume_cfg_path, "test_synth", resume=True, cwd=str(resume_root))

    # shard_00000 (ya consolidado antes del resume) no fue reescrito
    shard_after = (shard_dir / "shard_00000.positions.parquet").read_bytes()
    assert shard_before == shard_after

    df_resumed, files = read_all_shards(str(resume_root / "shards_v2"), "test_synth")
    assert len(df_resumed) == expected_total
    assert sorted(df_resumed["row_id"].tolist()) == list(range(expected_total))
    assert len(files) > 1  # confirma que efectivamente se escribieron shards nuevos tras el resume

    # el contenido final coincide con la corrida completa de referencia (mismos human_move_id por (game_id, ply))
    merged = df_baseline.merge(df_resumed, on=["game_id", "ply"], suffixes=("_base", "_resumed"))
    assert (merged["human_move_id_base"] == merged["human_move_id_resumed"]).all()
    assert (merged["fen_base"] == merged["fen_resumed"]).all()


def test_cannot_restart_without_resume_flag_when_checkpoint_exists(tmp_path):
    v1_dir = str(tmp_path / "outputs" / "test_synth")
    build_synthetic_v1_outputs(v1_dir)
    config_path = write_config(tmp_path, str(tmp_path / "outputs"))
    run_pipeline(config_path, "test_synth", cwd=str(tmp_path))

    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "06_enrich_positions.py"),
           "--config", config_path, "--test-name", "test_synth"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode != 0


def test_golden_lock_blocks_pipeline_and_override_unblocks_it(tmp_path):
    v1_dir = str(tmp_path / "outputs" / "test_synth")
    build_synthetic_v1_outputs(v1_dir)
    config_path = write_config(tmp_path, str(tmp_path / "outputs"))

    golden_lock_dir = tmp_path / "golden" / "test_synth_v1"
    golden_lock_dir.mkdir(parents=True)
    (golden_lock_dir / "GOLDEN_DATASET.lock").write_text(json.dumps({
        "golden_dataset_id": "test_synth_golden", "golden_dataset_version": "1.0.0",
        "test_name": "test_synth", "manifest_sha256": "irrelevant-for-this-test",
    }), encoding="utf-8")

    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "06_enrich_positions.py"),
           "--config", config_path, "--test-name", "test_synth"]
    blocked = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path))
    assert blocked.returncode != 0
    assert not (tmp_path / "shards_v2" / "test_synth").exists() or \
        len(list((tmp_path / "shards_v2" / "test_synth").glob("*"))) == 0

    run_pipeline(config_path, "test_synth", cwd=str(tmp_path), allow_golden_overwrite=True)
    assert (tmp_path / "shards_v2" / "test_synth").exists()
    assert len(list((tmp_path / "shards_v2" / "test_synth").glob("*.positions.parquet"))) > 0
