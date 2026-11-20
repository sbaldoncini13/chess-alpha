"""PASO 36 -- piloto de 10.000 posiciones (muestra determinista y
estratificada del Golden, NUNCA las 818.611 todavia). Mide throughput,
RAM, storage, cobertura de features, tasas de nulos y distribuciones antes
de autorizar el full run (PASO 37, gate)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_reader import GoldenDataset  # noqa: E402
from position_features import compute_all_position_features  # noqa: E402


def select_stratified_10k(positions: pd.DataFrame, seed: int = 20260812) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eligible = positions[positions["human_move_id"] != -1]

    def sample(mask, k):
        pool = eligible[mask]
        k = min(k, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        return pool.iloc[idx]

    parts = [
        sample(eligible["game_phase"] == "opening", 3300),
        sample(eligible["game_phase"] == "middlegame", 3300),
        sample(eligible["game_phase"] == "endgame", 3400),
    ]
    combined = pd.concat(parts).drop_duplicates(subset=["row_id"])
    remaining = 10000 - len(combined)
    if remaining > 0:
        rest_pool = eligible[~eligible["row_id"].isin(combined["row_id"])]
        idx = rng.choice(len(rest_pool), size=min(remaining, len(rest_pool)), replace=False)
        combined = pd.concat([combined, rest_pool.iloc[idx]])
    return combined.sort_values("row_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    config_dir = os.path.dirname(os.path.abspath(args.config))

    gd = GoldenDataset(config, config_dir)
    positions = gd.positions()
    sample = select_stratified_10k(positions)
    n = len(sample)

    proc = psutil.Process()
    ram_before = proc.memory_info().rss / (1024 * 1024)

    errors = []
    null_counts = {}
    t0 = time.time()
    all_feats = []
    for r in sample[["row_id", "fen"]].itertuples(index=False):
        try:
            feats = compute_all_position_features(r.fen, config)
            feats["row_id"] = int(r.row_id)
            all_feats.append(feats)
        except Exception as exc:
            errors.append({"row_id": int(r.row_id), "error": str(exc)})
    elapsed = time.time() - t0
    ram_after = proc.memory_info().rss / (1024 * 1024)

    df = pd.DataFrame(all_feats)
    for col in df.columns:
        null_counts[col] = int(df[col].isna().sum())
    total_nulls = sum(null_counts.values())

    est_bytes_per_row = df.memory_usage(deep=True).sum() / max(1, len(df))

    report = {
        "sample_size": n, "errors": len(errors), "error_examples": errors[:5],
        "elapsed_seconds": round(elapsed, 2), "positions_per_second": round(n / elapsed, 2) if elapsed > 0 else None,
        "ram_before_mb": round(ram_before, 1), "ram_after_mb": round(ram_after, 1),
        "ram_delta_mb": round(ram_after - ram_before, 1),
        "n_features": len(df.columns) - 1,
        "total_null_cells": total_nulls,
        "columns_with_any_null": [c for c, v in null_counts.items() if v > 0],
        "estimated_bytes_per_row_in_memory": round(est_bytes_per_row, 1),
        "projected_full_818611_minutes": round(818611 / (n / elapsed) / 60, 1) if elapsed > 0 else None,
        "row_id_join_check": {
            "unique_row_ids_in_sample": int(df["row_id"].nunique()) if len(df) else 0,
            "matches_sample_size": int(df["row_id"].nunique()) == n if len(df) else False,
        },
        "sanity_ranges": {
            "mobility_white_min_max": [int(df["mobility_white"].min()), int(df["mobility_white"].max())] if len(df) else None,
            "king_exposure_white_min_max": [float(df["white_king_exposure_score"].min()), float(df["white_king_exposure_score"].max())] if len(df) else None,
        },
    }

    reports_dir = os.path.abspath(os.path.join(config_dir, config["reports_dir"]))
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "pilot_10k_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nEscrito en {out_path}")


if __name__ == "__main__":
    main()
