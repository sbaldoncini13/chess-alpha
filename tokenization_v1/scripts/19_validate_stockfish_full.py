"""Fase 7: validacion post-run del full enrichment (808611 filas). Solo
lectura -- no vuelve a invocar Stockfish, no modifica nada. Tambien calcula
checksums SHA-256 de los 36 archivos finales de motor
(stockfish_full_integrity_manifest.json) y re-verifica que los shards base
siguen intactos (integrity_manifest.json ya calculado antes del run)."""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils_v2 import atomic_write_json  # noqa: E402
from legal_moves import VOCAB_SIZE  # noqa: E402
from shard_integrity import build_integrity_manifest, load_integrity_manifest, sha256_of_file, verify_shard_integrity  # noqa: E402

VALID_QUALITY = {"best", "excellent", "good", "inaccuracy", "mistake", "blunder"}


def check(cond, msg, failures):
    if not cond:
        failures.append(msg)


def percentile_stats(series):
    s = series.dropna().sort_values().to_numpy()
    if len(s) == 0:
        return {}
    def p(q):
        idx = min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))
        return float(s[idx])
    return {
        "min": float(s[0]), "p1": p(1), "p5": p(5), "median": p(50),
        "p90": p(90), "p95": p(95), "p99": p(99), "max": float(s[-1]),
        "mean": float(s.mean()),
    }


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
    full_dir = os.path.join(out_dir, config["stockfish_full"]["output_subdir"])
    reports_dir = os.path.abspath(os.path.join(base, config["reports_dir"]))

    shard_tags = sorted(
        d for d in os.listdir(full_dir) if os.path.isdir(os.path.join(full_dir, d)) and d.startswith("shard_")
    )

    failures = []

    eval_tables, topk_tables, policy_tables, crit_tables = [], [], [], []
    engine_files = {}
    for tag in shard_tags:
        sdir = os.path.join(full_dir, tag)
        for name in ("engine_eval", "engine_topk", "engine_policy_target", "engine_criticality"):
            fpath = os.path.join(sdir, f"{name}.parquet")
            check(os.path.isfile(fpath), f"falta {fpath}", failures)
            engine_files[f"{tag}/{name}.parquet"] = fpath
        eval_tables.append(pq.read_table(os.path.join(sdir, "engine_eval.parquet")))
        topk_tables.append(pq.read_table(os.path.join(sdir, "engine_topk.parquet")))
        policy_tables.append(pq.read_table(os.path.join(sdir, "engine_policy_target.parquet")))
        crit_tables.append(pq.read_table(os.path.join(sdir, "engine_criticality.parquet")))

    ev = pa.concat_tables(eval_tables).to_pandas()
    tk = pa.concat_tables(topk_tables).to_pandas()
    pt = pa.concat_tables(policy_tables).to_pandas()
    cr = pa.concat_tables(crit_tables).to_pandas()

    n_eligible_expected = 808611

    # 1. engine_eval: fila unica por row_id, cobertura completa
    check(len(ev) == n_eligible_expected, f"engine_eval tiene {len(ev)} filas, esperadas {n_eligible_expected}", failures)
    check(ev["row_id"].is_unique, "row_id duplicado en engine_eval", failures)
    check(ev["best_move_uci"].notna().all(), "best_move_uci ausente en alguna fila", failures)
    check((ev["depth"] == config["stockfish_depth"]).all(), "depth distinto del configurado en alguna fila", failures)

    # eval_type/cp/mate consistency: nunca ambos ni ninguno
    for prefix in ("best", "human"):
        type_col, cp_col, mate_col = f"{prefix}_eval_type", f"{prefix}_eval_cp", f"{prefix}_eval_mate_in"
        is_cp = ev[type_col] == "cp"
        is_mate = ev[type_col] == "mate"
        check(bool((is_cp == ev[cp_col].notna()).all()), f"{cp_col} inconsistente con {type_col}=='cp'", failures)
        check(bool((is_mate == ev[mate_col].notna()).all()), f"{mate_col} inconsistente con {type_col}=='mate'", failures)

    # 2. Top-K: rank invariantes
    ranks_per_row = tk.groupby("row_id")["rank"].apply(lambda s: s.sort_values().tolist())
    bad_rank_rows = 0
    for row_id, ranks in ranks_per_row.items():
        if ranks[0] != 1 or ranks != sorted(set(ranks)) or len(ranks) != len(set(ranks)):
            bad_rank_rows += 1
    check(bad_rank_rows == 0, f"{bad_rank_rows} filas con ranks invalidos en engine_topk", failures)
    counts_per_row = tk.groupby("row_id").size()
    check((counts_per_row <= config["stockfish_multipv"]).all(), "alguna posicion excede multipv configurado en engine_topk", failures)
    check(counts_per_row.reindex(ev["row_id"]).notna().all(), "alguna fila de engine_eval no tiene ningun candidato en engine_topk", failures)

    # rank=1 == best_move_uci
    rank1 = tk[tk["rank"] == 1].set_index("row_id")["move_uci"]
    best = ev.set_index("row_id")["best_move_uci"]
    joined = rank1.to_frame("rank1_uci").join(best.to_frame("best_uci"), how="inner")
    check(len(joined) == len(ev), "no todas las filas tienen rank=1 en engine_topk", failures)
    check(bool((joined["rank1_uci"] == joined["best_uci"]).all()), "rank=1 != best_move_uci en alguna fila", failures)

    # 3. Soft Policy
    check(len(pt) > 0, "engine_policy_target vacio", failures)
    check(not pt["probability"].isna().any(), "NaN en probability", failures)
    check(bool((pt["probability"] >= 0).all()), "probability negativa", failures)
    check(bool(np.isfinite(pt["probability"]).all()), "probability infinita", failures)
    sums = pt.groupby("row_id")["probability"].sum()
    bad_sums = sums[(sums - 1.0).abs() > 1e-3]
    check(len(bad_sums) == 0, f"sum(probability) != 1 (tolerancia 1e-3) en {len(bad_sums)} posiciones", failures)
    check(len(sums) == len(ev), "no todas las filas tienen policy target", failures)

    # 4. CPL / quality
    cpl_stats = percentile_stats(ev["centipawn_loss"])
    check(bool((ev["centipawn_loss"].dropna() >= 0).all()), "centipawn_loss negativo", failures)
    check(bool(ev["move_quality"].dropna().isin(VALID_QUALITY).all()), "move_quality con valor no reconocido", failures)

    # 5. engine_criticality
    check(len(cr) == n_eligible_expected, f"engine_criticality tiene {len(cr)} filas, esperadas {n_eligible_expected}", failures)
    check(cr["row_id"].is_unique, "row_id duplicado en engine_criticality", failures)
    crit_stats = percentile_stats(cr["engine_criticality_score"])
    check(bool(cr["engine_criticality_score"].between(0.0, 1.0).all()), "engine_criticality_score fuera de [0,1]", failures)
    check(not cr["engine_criticality_score"].isna().any(), "engine_criticality_score nulo inesperado", failures)
    pct_critical = float(cr["engine_tactical_or_critical"].mean() * 100)

    # 6. joins con shards base (human_move_uci)
    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    base_cols = pa.concat_tables([
        pq.read_table(os.path.join(shard_dir, f), columns=["row_id", "human_move_uci", "human_move_id"])
        for f in shard_files
    ]).to_pandas().set_index("row_id")
    base_eligible = base_cols[base_cols["human_move_id"] != -1]
    check(set(ev["row_id"]) == set(base_eligible.index), "row_id de engine_eval no coincide exactamente con elegibles de v2", failures)
    merged = base_eligible.join(ev.set_index("row_id")[["human_move_uci"]], rsuffix="_engine")
    check(bool((merged["human_move_uci"] == merged["human_move_uci_engine"]).all()), "human_move_uci no coincide con el shard base (join incorrecto)", failures)

    # 7. legalidad muestreada (5000 filas) contra la legal mask real de los shards base
    rng = np.random.default_rng(20260812)
    sample_row_ids = rng.choice(pt["row_id"].unique(), size=min(5000, pt["row_id"].nunique()), replace=False)
    off_by_shard, bin_by_shard, rowid_to_local = {}, {}, {}
    for f in shard_files:
        tag = f.split(".")[0]
        off_by_shard[tag] = np.load(os.path.join(shard_dir, f"{tag}.legalmoves_offsets.npy"))
        bin_by_shard[tag] = np.fromfile(os.path.join(shard_dir, f"{tag}.legalmoves.bin"), dtype=np.uint16)
        t = pq.read_table(os.path.join(shard_dir, f), columns=["row_id", "legal_move_count"])
        for local_i, rid in enumerate(t.column("row_id").to_pylist()):
            rowid_to_local[rid] = (tag, local_i, t.column("legal_move_count")[local_i].as_py())

    illegal = 0
    pt_by_row = pt[pt["row_id"].isin(sample_row_ids)].groupby("row_id")
    for row_id, grp in pt_by_row:
        tag, local_i, cnt = rowid_to_local[row_id]
        start = off_by_shard[tag][local_i]
        legal = set(bin_by_shard[tag][start:start + cnt].tolist())
        for mid in grp["move_id"]:
            if int(mid) not in legal:
                illegal += 1
    check(illegal == 0, f"{illegal} move_id de policy NO legales en la muestra de {len(sample_row_ids)}", failures)
    check(all(int(mid) <= VOCAB_SIZE - 1 for mid in ev["best_move_id"].dropna()), "best_move_id fuera de rango de vocabulario", failures)

    # 8. checksums de outputs de motor (36 archivos)
    engine_manifest = {"algorithm": "sha256", "n_files": len(engine_files), "files": {}}
    for rel, path in engine_files.items():
        engine_manifest["files"][rel] = {"size_bytes": os.path.getsize(path), "sha256": sha256_of_file(path)}
    engine_manifest_path = os.path.join(full_dir, "stockfish_full_integrity_manifest.json")
    atomic_write_json(engine_manifest_path, engine_manifest)

    # 9. re-verificar integridad de shards base (contra el manifest calculado ANTES del run)
    base_integrity_path = os.path.join(out_dir, "integrity_manifest.json")
    base_integrity_status = None
    if os.path.isfile(base_integrity_path):
        base_manifest = load_integrity_manifest(base_integrity_path)
        base_result = verify_shard_integrity(shard_dir, base_manifest)
        base_integrity_status = base_result["status"]
        check(base_integrity_status == "PASS", f"integridad de shards base FALLO tras el full run: {base_result['failures']}", failures)
    else:
        failures.append("no existe integrity_manifest.json de shards base para re-verificar")

    report = {
        "test_name": args.test_name,
        "n_positions_eligible": n_eligible_expected,
        "n_engine_eval_rows": len(ev), "n_topk_rows": len(tk), "n_policy_rows": len(pt), "n_criticality_rows": len(cr),
        "avg_candidates_per_position": round(len(tk) / len(ev), 4) if len(ev) else None,
        "move_quality_distribution": ev["move_quality"].value_counts().to_dict(),
        "centipawn_loss_stats": cpl_stats,
        "engine_criticality_score_stats": crit_stats,
        "pct_engine_tactical_or_critical": round(pct_critical, 4),
        "best_eval_type_distribution": ev["best_eval_type"].value_counts().to_dict(),
        "human_eval_type_distribution": ev["human_eval_type"].value_counts().to_dict(),
        "legality_sample_size": len(sample_row_ids), "legality_sample_illegal_found": illegal,
        "base_shard_integrity_status_post_run": base_integrity_status,
        "engine_output_integrity_manifest_path": engine_manifest_path,
        "engine_output_total_bytes": sum(v["size_bytes"] for v in engine_manifest["files"].values()),
        "n_checks": 15, "violaciones": failures, "status": "PASS" if not failures else "FAIL",
    }

    out_path = os.path.join(reports_dir, f"{args.test_name}_stockfish_full_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nEscrito en {out_path}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
