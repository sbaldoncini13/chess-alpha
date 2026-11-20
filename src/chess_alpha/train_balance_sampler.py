"""Sampler balanceado opcional para el bucket GENERAL de train (Punto 2/5
del plan de continuacion v1.1). Actua SOLO sobre train - nunca toca el
Dataset Epoch 3 original ni el validation set congelado (ver
`load_general_enriched_train_pool`, que excluye las filas de val usando el
pool unificado ya congelado, sin recalcular ningun split).

Jerarquia de balanceo, tal como la pidio el spec: fuente -> resultado ->
side_to_move -> game_id -> posicion. Con `max_positions_per_game` se evita
que una partida larga (mas plies) domine la muestra solo por tener mas
filas: se cappea cuantas posiciones de una misma partida pueden entrar.

El sampler "natural" (default) sigue siendo simplemente usar el pool tal
cual viene, sin este modulo - este archivo solo se activa si
`data_mix.general_sampler: balanced` en la config."""

import numpy as np
import pandas as pd

from chess_alpha.config import DATA_PROCESSED_DIR

EPOCH3_NEW_PATH = DATA_PROCESSED_DIR / "training_dataset_lichess_epoch3_new.parquet"
EPOCH3_MIDGAME_PATH = DATA_PROCESSED_DIR / "training_dataset_lichess_epoch3_midgame.parquet"
EPOCH3_REPLAY_PATH = DATA_PROCESSED_DIR / "training_dataset_lichess_epoch3_replay.parquet"

REPLAY_GAME_ID_BASE = 20_000_000  # mismo namespace que 36_build_validation_set.py

RESULT_BUCKET = {"1-0": "white_win", "0-1": "black_win", "1/2-1/2": "draw"}


def load_general_enriched_train_pool(unified_pool_path) -> pd.DataFrame:
    """Relee los 3 buckets originales de Epoch 3 (fen/turn/game_id/result
    intactos, solo lectura) y les agrega result_bucket + source, despues
    excluye las filas que estan en el validation set congelado (por FEN,
    seguro para el bucket general porque ahi el join real es por game_id -
    se usa el pool unificado ya calculado, no se recalcula el split)."""
    new_df = pd.read_parquet(EPOCH3_NEW_PATH)
    new_df["source"] = "new"
    midgame_df = pd.read_parquet(EPOCH3_MIDGAME_PATH)
    midgame_df["source"] = "midgame"
    replay_df = pd.read_parquet(EPOCH3_REPLAY_PATH).reset_index(drop=True)
    replay_df["game_id"] = REPLAY_GAME_ID_BASE + replay_df.index
    replay_df["source"] = "replay"

    cols = ["fen", "turn", "policy_target", "value_target", "game_id", "result", "source"]
    combined = pd.concat([new_df[cols], midgame_df[cols], replay_df[cols]], ignore_index=True)
    combined["result_bucket"] = combined["result"].map(RESULT_BUCKET).fillna("invalid")

    unified = pd.read_parquet(unified_pool_path, columns=["fen", "split", "row_source"])
    val_fens = set(unified.loc[(unified["split"] == "val") & unified["row_source"].str.startswith("general"), "fen"])
    train_mask = ~combined["fen"].isin(val_fens)
    return combined[train_mask].reset_index(drop=True)


def _cap_positions_per_game(df: pd.DataFrame, max_per_game: int, rng: np.random.Generator) -> pd.DataFrame:
    if max_per_game is None:
        return df
    return df.groupby("game_id", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max_per_game), random_state=rng.integers(0, 2**31 - 1))
    )


def _sample_with_repeat_cap(pool_idx: np.ndarray, n_target: int, oversampling_max: int, rng: np.random.Generator):
    """Como _sample_with_cap en train_dataset.py: arma n_target indices
    tomados de pool_idx repitiendo cada uno como maximo oversampling_max
    veces, en vez de un rng.choice(replace=True) sin tope (que puede hacer
    que UNA posicion, y por lo tanto UNA partida, aparezca decenas de veces
    en subgrupos chicos como "tablas" - poco frecuentes en new/midgame)."""
    n_available = len(pool_idx)
    if n_available == 0:
        return np.array([], dtype=np.int64), n_target
    max_possible = n_available * oversampling_max
    n_target_capped = min(n_target, max_possible)
    shortfall = n_target - n_target_capped

    full_rounds = n_target_capped // n_available
    remainder = n_target_capped % n_available
    chunks = []
    for _ in range(full_rounds):
        chunks.append(rng.permutation(pool_idx))
    if remainder:
        chunks.append(rng.permutation(pool_idx)[:remainder])
    result = np.concatenate(chunks) if chunks else np.array([], dtype=np.int64)
    return result, shortfall


def build_balanced_general_indices(enriched_df: pd.DataFrame, n_target: int, seed: int = 42,
                                    max_positions_per_game: int = 8, oversampling_max: int = 4) -> tuple:
    """Devuelve (indices_array, report). indices_array son posiciones
    (0-based, sobre `enriched_df.reset_index(drop=True)`) elegidas con la
    jerarquia fuente -> resultado -> side_to_move -> game_id -> posicion,
    apuntando a ~1/3 white_win/black_win/draw y ~50/50 white/black-to-move
    DENTRO de cada fuente, preservando la proporcion original entre
    fuentes (new/midgame/replay). `oversampling_max` limita cuantas veces
    puede repetirse una misma posicion (y por lo tanto una misma partida)
    cuando el subgrupo (p.ej. "tablas" en new/midgame, raras: ~4-6%) es
    demasiado chico para llegar al target sin repetir - sin este tope, una
    sola partida puede terminar apareciendo decenas de veces (visto en
    pruebas: 66 filas de una misma partida en una muestra de 100k)."""
    df = enriched_df.reset_index(drop=True)
    rng = np.random.default_rng(seed)

    source_props = df["source"].value_counts(normalize=True).to_dict()
    all_idx = []
    report = {"n_target": n_target, "by_source": {}}

    for source, source_frac in source_props.items():
        source_df = df[df["source"] == source]
        n_source_target = int(round(n_target * source_frac))
        per_result_target = n_source_target // 3
        source_report = {"target": n_source_target, "by_result": {}}

        for result in ("white_win", "black_win", "draw"):
            result_df = source_df[source_df["result_bucket"] == result]
            if len(result_df) == 0:
                source_report["by_result"][result] = {"target": per_result_target, "available": 0, "effective": 0}
                continue
            per_side_target = per_result_target // 2
            result_idx = []
            side_report = {}
            for side in ("white", "black"):
                side_df = result_df[result_df["turn"] == side]
                if len(side_df) == 0:
                    side_report[side] = {"target": per_side_target, "available": 0, "effective": 0}
                    continue
                capped = _cap_positions_per_game(side_df, max_positions_per_game, rng)
                pool_idx = capped.index.to_numpy()
                chosen, shortfall = _sample_with_repeat_cap(pool_idx, per_side_target, oversampling_max, rng)
                result_idx.append(chosen)
                side_report[side] = {
                    "target": per_side_target, "available": len(pool_idx), "effective": len(chosen),
                    "shortfall": int(shortfall), "max_repeats_per_position": int(oversampling_max),
                }
            source_report["by_result"][result] = {"target": per_result_target, "by_side": side_report}
            if result_idx:
                all_idx.append(np.concatenate(result_idx))

        report["by_source"][source] = source_report

    indices = np.concatenate(all_idx) if all_idx else np.array([], dtype=np.int64)
    rng.shuffle(indices)
    report["n_effective"] = len(indices)
    return indices, report
