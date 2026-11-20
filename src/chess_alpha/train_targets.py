"""Construccion de targets de policy (hard/soft) y value (mixto), y mascara
de jugadas legales para el entrenamiento v1.1. Ver secciones 7-9 del spec.

No cambia el vocabulario ni el formato de policy/value existente: los
targets siguen viviendo en el mismo espacio (indice de vocabulario UCI,
value en [0,1] desde la perspectiva de quien mueve)."""

import chess
import numpy as np
import torch


def build_vocab_index(vocab: list) -> dict:
    return {uci: idx for idx, uci in enumerate(vocab)}


def legal_move_mask(fen: str, vocab_index: dict) -> np.ndarray:
    """Mascara booleana (len(vocab),) marcando que jugadas legales de esta
    posicion existen en el vocabulario. Las jugadas legales fuera del
    vocabulario simplemente no se marcan (no hay indice para ellas)."""
    board = chess.Board(fen)
    mask = np.zeros(len(vocab_index), dtype=bool)
    for move in board.legal_moves:
        idx = vocab_index.get(move.uci())
        if idx is not None:
            mask[idx] = True
    return mask


def apply_legal_mask(logits: torch.Tensor, mask: torch.Tensor, fill_value: float = None) -> torch.Tensor:
    """logits: (B, vocab). mask: (B, vocab) booleana, True = jugada legal
    presente en el vocabulario. No permite que la red desperdicie masa en
    jugadas ilegales/fuera de vocabulario. Por defecto usa el minimo
    representable del dtype de `logits` (no un -1e9 fijo): bajo autocast/AMP
    los logits pueden ser float16, donde -1e9 desborda y tira RuntimeError."""
    if fill_value is None:
        fill_value = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, fill_value)


def build_soft_policy_target(candidates: list, vocab_index: dict, temperature_cp: float = 100.0):
    """candidates: lista de (move_uci, cpl) con cpl = perdida en centipawns
    respecto a la mejor jugada evaluada (0 para la mejor, siempre >= 0).
    Devuelve (idx_array int64, weight_array float32) normalizados via
    p(a) ∝ exp(-CPL(a)/temperature), o (None, None) si ninguna candidata
    esta en el vocabulario (la fila debe caer a target duro como fallback)."""
    idxs, cpls = [], []
    for uci, cpl in candidates:
        idx = vocab_index.get(uci)
        if idx is not None:
            idxs.append(idx)
            cpls.append(cpl)
    if not idxs:
        return None, None

    cpls = np.asarray(cpls, dtype=np.float64)
    logits = -cpls / temperature_cp
    logits -= logits.max()  # estabilidad numerica, no cambia el softmax
    weights = np.exp(logits)
    weights /= weights.sum()
    return np.asarray(idxs, dtype=np.int64), weights.astype(np.float32)


def cp_to_value(cp, cp_scale: float = 600.0):
    """Centipawns (perspectiva de quien mueve) -> [0,1] via tanh. Los mates
    ya vienen codificados como +-(100000 - mate_in) en engine_eval.py, asi
    que tanh(cp/cp_scale) satura naturalmente a +-1.0 sin necesidad de un
    caso especial: no usar +-100000 directamente como value, siempre pasar
    por esta funcion."""
    centered = np.tanh(np.asarray(cp, dtype=np.float64) / cp_scale)
    return (centered + 1.0) / 2.0


def value_target_mixed(result_target, engine_cp, result_weight: float = 0.7,
                        engine_weight: float = 0.3, cp_scale: float = 600.0) -> float:
    """result_target: 1.0/0.5/0.0 desde la perspectiva de quien mueve, o None
    si la partida no es terminal (o no aplica, p.ej. filas correctivas sin
    resultado de partida). engine_cp: centipawns desde la perspectiva de
    quien mueve, o None si no hay evaluacion de motor.

    Si faltan ambos, error (la fila no deberia haber llegado hasta aca). Si
    falta uno, se usa el otro con peso 1.0 en vez de dejar hueco."""
    has_result = result_target is not None
    has_engine = engine_cp is not None
    if not has_result and not has_engine:
        raise ValueError("value_target_mixed: no hay result_target ni engine_cp para esta fila.")
    if has_result and not has_engine:
        return float(result_target)
    if has_engine and not has_result:
        return float(cp_to_value(engine_cp, cp_scale=cp_scale))
    return float(result_weight * result_target + engine_weight * cp_to_value(engine_cp, cp_scale=cp_scale))
