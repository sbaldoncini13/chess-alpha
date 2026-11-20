"""Validacion de consistencia post-procesamiento sobre outputs_v2 ya escritos.
Solo lee shards_v2/<test_name>/ y outputs/<test_name>/ (v1, solo lectura).
No reescribe nada -- si encuentra una violacion, la reporta y falla, no
intenta reparar nada automaticamente.
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legal_moves import VOCAB_SIZE  # noqa: E402


def check(condition, message, failures):
    if not condition:
        failures.append(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    base = os.path.dirname(args.config)
    shard_dir = os.path.abspath(os.path.join(base, config["shard_dir"], args.test_name))
    v1_dir = os.path.abspath(os.path.join(base, config["base_dataset"]["v1_outputs_dir"], args.test_name))

    failures = []

    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    check(len(shard_files) > 0, f"no hay shards en {shard_dir}", failures)

    tables = []
    for sf in shard_files:
        path = os.path.join(shard_dir, sf)
        try:
            t = pq.read_table(path)
        except Exception as exc:
            failures.append(f"shard ilegible: {sf} ({exc})")
            continue
        tables.append(t)

    full = pa.concat_tables(tables).to_pandas()
    n = len(full)

    # 1. row_id contiguo 0..N-1
    check(sorted(full["row_id"].tolist()) == list(range(n)), "row_id no es contiguo 0..N-1", failures)

    # 2. shapes/tipos constantes (todas las columnas presentes en cada shard, mismo schema)
    schemas = {t.schema for t in tables}
    check(len(schemas) == 1, "shards con schema inconsistente entre bloques", failures)

    # 3. sin NaN inesperados en columnas no-nullable por diseno
    non_nullable_cols = ["row_id", "game_id", "ply", "fen", "position_id",
                          "value_target_stm", "legal_move_count", "material_white", "material_black"]
    for col in non_nullable_cols:
        check(full[col].isna().sum() == 0, f"NaN inesperado en columna '{col}'", failures)

    # 4. valores de resultado validos
    check(full["value_target_stm"].isin([-1, 0, 1]).all(), "value_target_stm fuera de {-1,0,1}", failures)
    check(full["game_result_white_pov"].isin([-1, 0, 1]).all(), "game_result_white_pov fuera de {-1,0,1}", failures)

    # 5. side_to_move consistente con el signo de fen (columna 2 de la FEN)
    fen_stm = full["fen"].str.split().str[1]
    expected = full["side_to_move"].map({True: "w", False: "b"})
    check((fen_stm == expected).all(), "side_to_move inconsistente con el campo de turno de la FEN", failures)

    # 6. numero de posiciones compatible con los plies de v1 (ply_count+1 por partida)
    v1_meta = pq.read_table(os.path.join(v1_dir, "metadata.parquet"), columns=["sample_id", "ply_count"]).to_pandas()
    expected_total = int(v1_meta["ply_count"].sum()) + len(v1_meta)
    check(n == expected_total, f"total de posiciones ({n}) != sum(ply_count)+n_games de v1 ({expected_total})", failures)

    # 7. game_id valido (existe en v1) y cantidad de filas por partida == ply_count+1
    rows_per_game = full.groupby("game_id").size()
    check(set(rows_per_game.index) == set(v1_meta["sample_id"]), "game_id en v2 no coincide con sample_id de v1", failures)
    merged = v1_meta.set_index("sample_id").join(rows_per_game.rename("n_rows_v2"))
    check((merged["n_rows_v2"] == merged["ply_count"] + 1).all(), "cantidad de filas por partida != ply_count+1", failures)

    # 8. human_move_id valido: -1 solo en la ultima fila de cada partida, rango correcto en las demas
    non_terminal = full.groupby("game_id").apply(lambda g: (g.sort_values("ply")["human_move_id"].iloc[:-1] != -1).all())
    check(bool(non_terminal.all()), "hay human_move_id == -1 en una fila NO terminal", failures)
    terminal_is_neg1 = full.groupby("game_id").apply(lambda g: g.sort_values("ply")["human_move_id"].iloc[-1] == -1)
    check(bool(terminal_is_neg1.all()), "la fila terminal de alguna partida tiene human_move_id != -1", failures)
    non_terminal_ids = full[full["human_move_id"] != -1]["human_move_id"]
    check(non_terminal_ids.between(0, VOCAB_SIZE - 1).all(), "human_move_id fuera de rango del vocabulario", failures)

    # 9. human_move_id siempre dentro de su propia legal_moves list
    off_by_shard = {}
    bin_by_shard = {}
    for sf in shard_files:
        tag = sf.split(".")[0]
        off_by_shard[tag] = np.load(os.path.join(shard_dir, f"{tag}.legalmoves_offsets.npy"))
        bin_by_shard[tag] = np.fromfile(os.path.join(shard_dir, f"{tag}.legalmoves.bin"), dtype=np.uint16)

    violations = 0
    sample = full[full["human_move_id"] != -1].sample(min(2000, (full["human_move_id"] != -1).sum()), random_state=0)
    row_to_shard = {}
    cursor = 0
    for sf in shard_files:
        tag = sf.split(".")[0]
        t = next(t for t, name in zip(tables, shard_files) if name == sf)
        n_rows = t.num_rows
        for local_i, rid in enumerate(t.column("row_id").to_pylist()):
            row_to_shard[rid] = (tag, local_i)
        cursor += n_rows

    for _, row in sample.iterrows():
        tag, local_i = row_to_shard[row["row_id"]]
        offs = off_by_shard[tag]
        arr = bin_by_shard[tag]
        start = offs[local_i]
        cnt = row["legal_move_count"]
        legal = arr[start:start + cnt]
        if row["human_move_id"] not in legal:
            violations += 1
    check(violations == 0, f"human_move_id fuera de su legal_moves list en {violations} filas muestreadas", failures)

    # 10. manifest coherente (si existe)
    manifest_path = os.path.join(os.path.dirname(shard_dir).replace("shards_v2", "outputs_v2"), args.test_name, "manifest.json")

    report = {
        "test_name": args.test_name, "n_positions_validadas": n, "n_shards": len(shard_files),
        "checks_ejecutados": 9, "violaciones": failures, "status": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
