"""PASO 49 (pre-manifest) -- validacion post-run sobre el dataset V3
COMPLETO (818.611 posiciones / 9 shards), mismo criterio que el gate de
la piloto 10k pero a escala completa: join de row_id 100%, sin NaN
inesperados, rangos de cordura, y dtype consistente de los bitboards
(regresion directa del bug real de shard_00001, ver attack_maps.py)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

CONFIG_PATH = os.path.join(SCRIPTS_DIR, "..", "config", "enrichment_v3_config.json")
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)
CONFIG_DIR = os.path.dirname(os.path.abspath(CONFIG_PATH))
OUT_DIR = os.path.abspath(os.path.join(CONFIG_DIR, CONFIG["output_dir"]))
GOLDEN_SHARDS_DIR = os.path.abspath(os.path.join(CONFIG_DIR, CONFIG["source_golden"]["golden_shards_dir"]))

POSITION_LEVEL_FAMILIES = ["position_tactical", "position_strategic", "pawn_structure", "king_safety", "sequences_meta"]
BITBOARD_COLS = ["squares_attacked_white_bitboard", "squares_attacked_black_bitboard"]


def shard_tags():
    return sorted({f.split(".")[0] for f in os.listdir(GOLDEN_SHARDS_DIR) if f.endswith(".positions.parquet")})


def golden_total_rows():
    total = 0
    for tag in shard_tags():
        t = pq.read_table(os.path.join(GOLDEN_SHARDS_DIR, f"{tag}.positions.parquet"), columns=["row_id"])
        total += t.num_rows
    return total


def main():
    report = {"shards": {}, "issues": [], "bitboard_dtype_by_shard": {}}
    tags = shard_tags()

    expected_total = golden_total_rows()
    report["golden_total_rows"] = expected_total

    all_row_ids_by_family = {fam: [] for fam in POSITION_LEVEL_FAMILIES}
    null_counts = {}
    sanity = {"mobility_white_min": None, "mobility_white_max": None,
              "king_exposure_min": None, "king_exposure_max": None,
              "see_best_see_min": None, "see_best_see_max": None}

    for tag in tags:
        shard_dir = os.path.join(OUT_DIR, tag)
        shard_report = {}

        for fam in POSITION_LEVEL_FAMILIES:
            p = os.path.join(shard_dir, f"{fam}.parquet")
            t = pq.read_table(p)
            df = t.to_pandas()
            shard_report[fam] = len(df)
            all_row_ids_by_family[fam].append(df["row_id"].to_numpy())
            if df["row_id"].duplicated().any():
                report["issues"].append(f"{tag}/{fam}: row_id duplicado")

            for col in df.columns:
                n_null = int(df[col].isna().sum())
                if n_null:
                    null_counts.setdefault(f"{fam}.{col}", 0)
                    null_counts[f"{fam}.{col}"] += n_null

            if fam == "position_tactical":
                dtypes = {c: str(df[c].dtype) for c in BITBOARD_COLS}
                report["bitboard_dtype_by_shard"].setdefault(tag, dtypes)
                for c in BITBOARD_COLS:
                    if df[c].dtype != np.uint64:
                        report["issues"].append(f"{tag}/position_tactical.{c}: dtype {df[c].dtype} != uint64")
                sanity["see_best_see_min"] = df["see_best_see"].min() if sanity["see_best_see_min"] is None else min(sanity["see_best_see_min"], df["see_best_see"].min())
                sanity["see_best_see_max"] = df["see_best_see"].max() if sanity["see_best_see_max"] is None else max(sanity["see_best_see_max"], df["see_best_see"].max())

            if fam == "position_strategic":
                sanity["mobility_white_min"] = df["mobility_white"].min() if sanity["mobility_white_min"] is None else min(sanity["mobility_white_min"], df["mobility_white"].min())
                sanity["mobility_white_max"] = df["mobility_white"].max() if sanity["mobility_white_max"] is None else max(sanity["mobility_white_max"], df["mobility_white"].max())

            if fam == "king_safety":
                sanity["king_exposure_min"] = df["white_king_exposure_score"].min() if sanity["king_exposure_min"] is None else min(sanity["king_exposure_min"], df["white_king_exposure_score"].min())
                sanity["king_exposure_max"] = df["white_king_exposure_score"].max() if sanity["king_exposure_max"] is None else max(sanity["king_exposure_max"], df["white_king_exposure_score"].max())

        ed = pq.read_table(os.path.join(shard_dir, "engine_derived.parquet")).to_pandas()
        shard_report["engine_derived"] = len(ed)
        hc = pq.read_table(os.path.join(shard_dir, "human_consensus.parquet")).to_pandas()
        shard_report["human_consensus"] = len(hc)
        sm = pq.read_table(os.path.join(shard_dir, "see_moves.parquet")).to_pandas()
        shard_report["see_moves"] = len(sm)

        report["shards"][tag] = shard_report

    for fam, arrs in all_row_ids_by_family.items():
        combined = np.concatenate(arrs)
        n_total = len(combined)
        n_unique = len(np.unique(combined))
        expected_ids = set(range(expected_total))
        actual_ids = set(combined.tolist())
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        report.setdefault("row_id_join_check", {})[fam] = {
            "n_total_rows": int(n_total), "n_unique_row_ids": int(n_unique),
            "matches_golden_total": n_unique == expected_total == n_total,
            "n_missing": len(missing), "n_extra": len(extra),
        }
        if n_unique != expected_total or n_total != n_unique:
            report["issues"].append(f"{fam}: row_id join incompleto (unique={n_unique}, total={n_total}, esperado={expected_total})")

    report["null_counts_nonzero"] = null_counts
    report["sanity_ranges"] = {k: (float(v) if v is not None else None) for k, v in sanity.items()}

    if sanity["mobility_white_min"] is not None and sanity["mobility_white_min"] < 0:
        report["issues"].append("mobility_white negativa detectada")
    if sanity["king_exposure_min"] is not None and (sanity["king_exposure_min"] < 0 or sanity["king_exposure_max"] > 1):
        report["issues"].append("king_exposure_score fuera de [0,1]")

    report["status"] = "PASS" if not report["issues"] else "FAIL"

    reports_dir = os.path.abspath(os.path.join(CONFIG_DIR, CONFIG["reports_dir"]))
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "full_v3_validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(json.dumps({"status": report["status"], "issues": report["issues"],
                       "golden_total_rows": report["golden_total_rows"]}, indent=2, ensure_ascii=False))
    print(f"\nEscrito en {out_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
