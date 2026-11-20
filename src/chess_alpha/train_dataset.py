"""Dataset y mezcla de batches para el entrenamiento v1.1 (secciones 6, 8,
16 del spec). Carga las tres fuentes (general/correctivo/tactico), separa
train/val usando el validation set YA CONGELADO (no se recalcula el split
aca), y arma los indices de cada epoch de train respetando la mezcla
75/15/10 (general/correctivo/tactico) con oversampling limitado y logging
de repeticiones por ejemplo.

El 20% "mates/errores criticos" dentro del 15% correctivo (seccion 6) y el
10% "tactico especializado" (seccion 4) leen del MISMO tactical_dataset.parquet:
son la misma clase de contenido (mates/blunders/jaques ganadores), a los
que el spec les da dos pesos de muestreo distintos segun la seccion que se
lea. Se muestrean de forma independiente en cada slot, documentado aca y en
el reporte final."""

import chess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from chess_alpha.encoding import board_to_tensor, NUM_PLANES
from chess_alpha.train_targets import build_vocab_index, legal_move_mask


class MixSpec:
    """Cuantas filas de train sacar de cada fuente para un batch/epoch
    dado el tamano del bucket general y las proporciones de la config."""

    def __init__(self, n_general: int, data_mix_cfg):
        self.n_general = n_general
        pct_general = data_mix_cfg.pct_general
        if pct_general <= 0:
            raise ValueError("data_mix.pct_general debe ser > 0 para poder derivar el largo de la epoch.")
        epoch_total = int(round(n_general / pct_general))
        self.epoch_total = epoch_total
        self.n_corrective = int(round(epoch_total * data_mix_cfg.pct_corrective))
        self.n_tactical_slot = int(round(epoch_total * data_mix_cfg.pct_tactical))
        self.n_corrective_policy = int(round(self.n_corrective * data_mix_cfg.corrective_pct_policy))
        self.n_corrective_value = int(round(self.n_corrective * data_mix_cfg.corrective_pct_value))
        self.n_corrective_critical = self.n_corrective - self.n_corrective_policy - self.n_corrective_value


def _sample_with_cap(n_available: int, n_target: int, oversampling_max: int, rng: np.random.Generator):
    """Devuelve un array de indices locales [0, n_available) de longitud
    n_target, repitiendo cada indice como maximo `oversampling_max` veces.
    Si el pool es tan chico que ni con el cap se llega a n_target, se
    satura al maximo posible (n_available * oversampling_max) en vez de
    fallar, y se reporta el faltante."""
    if n_available == 0:
        return np.array([], dtype=np.int64), 0
    max_possible = n_available * oversampling_max
    n_target_capped = min(n_target, max_possible)
    shortfall = n_target - n_target_capped

    full_rounds = n_target_capped // n_available
    remainder = n_target_capped % n_available
    idx = []
    order = rng.permutation(n_available)
    for _ in range(full_rounds):
        idx.append(order)
        order = rng.permutation(n_available)
    if remainder:
        idx.append(order[:remainder])
    result = np.concatenate(idx) if idx else np.array([], dtype=np.int64)
    return result.astype(np.int64), shortfall


def build_epoch_indices(mix: MixSpec, n_corrective_policy_pool: int, n_corrective_value_pool: int,
                         n_tactical_pool: int, oversampling_max: int, seed: int):
    """Devuelve un dict source -> array de indices locales para UNA epoch de
    train, mas un reporte de repeticion efectiva por fuente."""
    rng = np.random.default_rng(seed)

    general_idx = rng.permutation(mix.n_general)

    cp_idx, cp_shortfall = _sample_with_cap(n_corrective_policy_pool, mix.n_corrective_policy, oversampling_max, rng)
    cv_idx, cv_shortfall = _sample_with_cap(n_corrective_value_pool, mix.n_corrective_value, oversampling_max, rng)
    cc_idx, cc_shortfall = _sample_with_cap(n_tactical_pool, mix.n_corrective_critical, oversampling_max, rng)
    tac_idx, tac_shortfall = _sample_with_cap(n_tactical_pool, mix.n_tactical_slot, oversampling_max, rng)

    def repeat_factor(target, pool):
        return (target / pool) if pool else 0.0

    def exposure_stats(idx: np.ndarray, pool_size: int) -> dict:
        """max/mean de repeticiones reales (no el promedio nominal target/pool)
        y diversidad efectiva (fraccion de posiciones DISTINTAS del pool que
        efectivamente entraron en la epoch) - pedido en la seccion 3/6 del
        plan de continuacion v1.1."""
        if len(idx) == 0 or pool_size == 0:
            return {"max_repeats": 0, "mean_repeats": 0.0, "n_unique": 0, "diversity_ratio": 0.0}
        counts = np.bincount(idx, minlength=pool_size)
        n_unique = int((counts > 0).sum())
        return {
            "max_repeats": int(counts.max()),
            "mean_repeats": float(counts[counts > 0].mean()),
            "n_unique": n_unique,
            "diversity_ratio": n_unique / pool_size,
        }

    report = {
        "epoch_total_target": mix.epoch_total,
        "n_general": int(mix.n_general),
        "n_corrective_policy": {"target": mix.n_corrective_policy, "pool": n_corrective_policy_pool,
                                 "effective": len(cp_idx), "avg_repeats": repeat_factor(len(cp_idx), n_corrective_policy_pool), "shortfall": int(cp_shortfall),
                                 **exposure_stats(cp_idx, n_corrective_policy_pool)},
        "n_corrective_value": {"target": mix.n_corrective_value, "pool": n_corrective_value_pool,
                                "effective": len(cv_idx), "avg_repeats": repeat_factor(len(cv_idx), n_corrective_value_pool), "shortfall": int(cv_shortfall),
                                **exposure_stats(cv_idx, n_corrective_value_pool)},
        "n_corrective_critical": {"target": mix.n_corrective_critical, "pool": n_tactical_pool,
                                   "effective": len(cc_idx), "avg_repeats": repeat_factor(len(cc_idx), n_tactical_pool), "shortfall": int(cc_shortfall),
                                   **exposure_stats(cc_idx, n_tactical_pool)},
        "n_tactical_slot": {"target": mix.n_tactical_slot, "pool": n_tactical_pool,
                             "effective": len(tac_idx), "avg_repeats": repeat_factor(len(tac_idx), n_tactical_pool), "shortfall": int(tac_shortfall),
                             **exposure_stats(tac_idx, n_tactical_pool)},
    }
    indices = {
        "general": general_idx,
        "corrective_policy": cp_idx,
        "corrective_value": cv_idx,
        "corrective_critical": cc_idx,
        "tactical": tac_idx,
    }
    return indices, report


class MixedChessDataset(Dataset):
    """Envuelve las filas ya seleccionadas para una epoch (train) o el pool
    completo (val) y devuelve tensores listos para el modelo. La mascara
    legal y el tensor de entrada se calculan on-the-fly con python-chess
    (no hay tensores 23x8x8 pre-cacheados para Epoch 3/correctivo/tactico:
    a esta escala el costo de encoding esta dominado por el forward pass
    del modelo, no por python-chess)."""

    def __init__(self, rows: pd.DataFrame, vocab: list):
        self.rows = rows.reset_index(drop=True)
        self.vocab = vocab
        self.vocab_index = build_vocab_index(vocab)
        self.vocab_size = len(vocab)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        fen = row["fen"]
        board = chess.Board(fen)
        x = torch.from_numpy(board_to_tensor(board))

        mask = torch.from_numpy(legal_move_mask(fen, self.vocab_index))
        hard_target = int(row["policy_target"])

        soft_idx = row.get("soft_policy_idx")
        soft_weight = row.get("soft_policy_weight")
        is_soft = soft_idx is not None and not (isinstance(soft_idx, float) and np.isnan(soft_idx)) and len(soft_idx) > 0

        soft_target = torch.zeros(self.vocab_size)
        if is_soft:
            idxs = torch.as_tensor(np.asarray(soft_idx, dtype=np.int64))
            weights = torch.as_tensor(np.asarray(soft_weight, dtype=np.float32))
            soft_target[idxs] = weights

        value_target = torch.tensor(float(row["value_target"]), dtype=torch.float32)

        return x, mask, hard_target, soft_target, is_soft, value_target


def collate_batch(batch):
    xs, masks, hard_targets, soft_targets, is_softs, value_targets = zip(*batch)
    return (
        torch.stack(xs).float(),
        torch.stack(masks),
        torch.tensor(hard_targets, dtype=torch.long),
        torch.stack(soft_targets),
        torch.tensor(is_softs, dtype=torch.bool),
        torch.stack(value_targets),
    )


def load_pools(paths_cfg):
    """Carga el pool unificado (36_build_validation_set.py) y separa train/val
    usando su columna 'split', ya calculada por game_id de forma congelada.

    IMPORTANTE: no reconstruir el split comparando FEN contra el validation
    set. Un mismo FEN (tipicamente una apertura comun) aparece legitimamente
    en muchas partidas distintas - unas en train, otras en val - asi que
    "fen in val_fens" marca como val muchisimas filas de train que solo
    comparten una posicion inicial con alguna partida de val (bug real
    encontrado y corregido: inflaba el val set del bucket general de ~10%
    a ~20%). El pool unificado ya tiene el split correcto por game_id."""
    pool = pd.read_parquet(paths_cfg.unified_pool_path)

    def subset(row_source_prefix, split):
        mask = pool["row_source"].str.startswith(row_source_prefix) & (pool["split"] == split)
        return pool[mask].reset_index(drop=True)

    pools = {
        "general_train": subset("general", "train"),
        "general_val": subset("general", "val"),
        "corrective_train": subset("corrective", "train"),
        "corrective_val": subset("corrective", "val"),
        "tactical_train": subset("tactical", "train"),
        "tactical_val": subset("tactical", "val"),
    }
    pools["corrective_policy_train"] = pools["corrective_train"][pools["corrective_train"]["row_source"] == "corrective_policy"].reset_index(drop=True)
    pools["corrective_value_train"] = pools["corrective_train"][pools["corrective_train"]["row_source"] == "corrective_value"].reset_index(drop=True)
    return pools
