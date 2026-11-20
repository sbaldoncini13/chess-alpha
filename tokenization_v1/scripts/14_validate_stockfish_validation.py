"""Validaciones post-Stockfish sobre stockfish_validation_1000/ (15 checks
pedidos en la auditoria). Solo lectura -- no vuelve a invocar Stockfish, no
modifica nada. Tambien re-verifica la integridad de los shards base
(check 15) contra el integrity_manifest.json calculado ANTES de la corrida.
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
from shard_integrity import load_integrity_manifest, verify_shard_integrity  # noqa: E402


def check(cond, msg, failures):
    if not cond:
        failures.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    base = os.path.dirname(args.config)
    shard_dir = os.path.abspath(os.path.join(base, config["shard_dir"], args.test_name))
    out_dir = os.path.abspath(os.path.join(base, config["output_dir"], args.test_name))
    subdir = os.path.join(out_dir, config["stockfish_validation"]["output_subdir"])

    failures = []

    run_manifest = json.load(open(os.path.join(subdir, "run_manifest.json"), encoding="utf-8"))
    eval_t = pq.read_table(os.path.join(subdir, "engine_eval.parquet")).to_pandas()
    topk_t = pq.read_table(os.path.join(subdir, "engine_topk.parquet")).to_pandas()
    policy_t = pq.read_table(os.path.join(subdir, "engine_policy_target.parquet")).to_pandas()
    crit_t = pq.read_table(os.path.join(subdir, "engine_criticality.parquet")).to_pandas()

    n_ok = run_manifest["sample_size_processed_ok"]

    # 1. 1000 posiciones seleccionadas
    check(run_manifest["sample_size_selected"] == config["stockfish_validation"]["sample_size"],
          f"sample_size_selected ({run_manifest['sample_size_selected']}) != "
          f"{config['stockfish_validation']['sample_size']} solicitado", failures)

    # 2. engine evaluation disponible cuando corresponde
    check(len(eval_t) == n_ok, "engine_eval.parquet no tiene una fila por posicion procesada", failures)
    check(eval_t["best_eval_unified_cp_equivalent"].notna().all() or len(eval_t) == 0,
          "hay best_eval_unified_cp_equivalent nulo en engine_eval", failures)

    # 3. best move disponible
    check(eval_t["best_move_uci"].notna().all(), "best_move_uci ausente en alguna fila", failures)

    # 4. rank 1 == best move
    rank1 = topk_t[topk_t["rank"] == 1][["row_id", "move_uci"]].set_index("row_id")["move_uci"]
    best = eval_t.set_index("row_id")["best_move_uci"]
    joined = rank1.to_frame("rank1_uci").join(best.to_frame("best_uci"), how="inner")
    check((joined["rank1_uci"] == joined["best_uci"]).all(),
          "rank=1 en engine_topk no coincide con best_move_uci de engine_eval en todas las filas", failures)
    check(len(joined) == n_ok, "no todas las posiciones tienen un candidato rank=1 en engine_topk", failures)

    # 5. Top-K persistido realmente (no solo tamano)
    check(len(topk_t) > 0, "engine_topk.parquet vacio", failures)
    check(set(["row_id", "rank", "move_id", "move_uci", "eval_type", "eval_unified_cp_equivalent"]).issubset(topk_t.columns),
          "engine_topk.parquet no tiene las columnas de detalle esperadas", failures)
    ranks_per_row = topk_t.groupby("row_id")["rank"].apply(lambda s: list(s.sort_values()))
    for row_id, ranks in ranks_per_row.items():
        if ranks[0] != 1 or ranks != sorted(set(ranks)) or len(ranks) != len(set(ranks)):
            failures.append(f"row_id={row_id}: ranks invalidos {ranks}")
            break
    check((topk_t.groupby("row_id").size() <= config["stockfish_multipv"]).all(),
          "alguna posicion tiene mas candidatos que multipv configurado", failures)

    # 6. Soft Policy persistida realmente
    check(len(policy_t) > 0, "engine_policy_target.parquet vacio", failures)
    check(set(["row_id", "move_id", "probability"]).issubset(policy_t.columns),
          "engine_policy_target.parquet no tiene columnas de detalle esperadas", failures)

    # 7. sum(policy) ~= 1 por row_id
    sums = policy_t.groupby("row_id")["probability"].sum()
    bad_sums = sums[(sums - 1.0).abs() > 1e-4]
    check(len(bad_sums) == 0, f"sum(probability) != 1 en {len(bad_sums)} posiciones", failures)
    check(len(sums) == n_ok, "no todas las posiciones tienen policy target", failures)

    # 8. todos los move_id de policy son legales
    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    off_by_shard, bin_by_shard, rowid_to_local = {}, {}, {}
    for f in shard_files:
        tag = f.split(".")[0]
        off_by_shard[tag] = np.load(os.path.join(shard_dir, f"{tag}.legalmoves_offsets.npy"))
        bin_by_shard[tag] = np.fromfile(os.path.join(shard_dir, f"{tag}.legalmoves.bin"), dtype=np.uint16)
        t = pq.read_table(os.path.join(shard_dir, f), columns=["row_id", "legal_move_count"])
        for local_i, rid in enumerate(t.column("row_id").to_pylist()):
            rowid_to_local[rid] = (tag, local_i, t.column("legal_move_count")[local_i].as_py())

    illegal = 0
    for row_id, grp in policy_t.groupby("row_id"):
        if row_id not in rowid_to_local:
            illegal += 1
            continue
        tag, local_i, cnt = rowid_to_local[row_id]
        start = off_by_shard[tag][local_i]
        legal = set(bin_by_shard[tag][start:start + cnt].tolist())
        for mid in grp["move_id"]:
            if mid not in legal:
                illegal += 1
    check(illegal == 0, f"{illegal} move_id de policy NO legales en su posicion", failures)
    check(not policy_t["probability"].isna().any(), "NaN en probability", failures)
    check((policy_t["probability"] >= 0).all(), "probability negativa", failures)
    check(np.isfinite(policy_t["probability"]).all(), "probability infinita", failures)

    # 9. centipawn_loss valido cuando corresponde
    check((eval_t["centipawn_loss"].dropna() >= 0).all(), "centipawn_loss negativo", failures)

    # 10. move_quality disponible cuando corresponde
    valid_quality = {"best", "excellent", "good", "inaccuracy", "mistake", "blunder"}
    check(eval_t["move_quality"].dropna().isin(valid_quality).all(), "move_quality con valor no reconocido", failures)

    # 11 y 12. engine_criticality_score disponible y en rango
    check(len(crit_t) == n_ok, "engine_criticality.parquet no tiene una fila por posicion", failures)
    scores = crit_t["engine_criticality_score"].dropna()
    check(len(scores) == len(crit_t), "hay engine_criticality_score nulo inesperado", failures)
    check(((scores >= 0.0) & (scores <= 1.0)).all(), "engine_criticality_score fuera de [0,1]", failures)

    # 13. sin duplicacion inesperada de row_id
    check(eval_t["row_id"].is_unique, "row_id duplicado en engine_eval", failures)
    check(crit_t["row_id"].is_unique, "row_id duplicado en engine_criticality", failures)

    # 14. joins con shards base correctos
    base_cols = pa.concat_tables([
        pq.read_table(os.path.join(shard_dir, f), columns=["row_id", "fen", "human_move_uci"]) for f in shard_files
    ]).to_pandas().set_index("row_id")
    sampled_base = base_cols.loc[eval_t["row_id"]]
    check((sampled_base["human_move_uci"].to_numpy() == eval_t["human_move_uci"].to_numpy()).all(),
          "human_move_uci de engine_eval no coincide con el shard base (join incorrecto)", failures)

    # 15. ningun shard base fue modificado (re-verifica integridad calculada ANTES de correr Stockfish)
    integrity_path = os.path.join(out_dir, "integrity_manifest.json")
    if os.path.isfile(integrity_path):
        manifest = load_integrity_manifest(integrity_path)
        integrity_result = verify_shard_integrity(shard_dir, manifest)
        check(integrity_result["status"] == "PASS",
              f"integridad de shards base FALLO tras la corrida Stockfish: {integrity_result['failures']}", failures)
    else:
        failures.append("no existe integrity_manifest.json -- no se pudo re-verificar shards base")

    report = {
        "test_name": args.test_name, "n_checks": 15, "n_positions_validated": n_ok,
        "violaciones": failures, "status": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
