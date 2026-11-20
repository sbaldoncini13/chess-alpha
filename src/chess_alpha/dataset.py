import json

import numpy as np

RESULT_TO_WHITE_SCORE = {
    "1-0": 1.0,
    "0-1": 0.0,
    "1/2-1/2": 0.5,
}


def value_target_for_side_to_move(result: str, turn: str) -> float:
    """Convierte el resultado final de la partida en un target de valor
    desde la perspectiva del jugador que le toca mover en esa posición
    (1.0 = termina ganando, 0.5 = tablas, 0.0 = termina perdiendo).
    Devuelve None si el resultado no es terminal (partida inconclusa, "*")."""
    white_score = RESULT_TO_WHITE_SCORE.get(result)
    if white_score is None:
        return None
    return white_score if turn == "white" else 1.0 - white_score


def build_policy_vocab(moves_uci) -> list:
    """Vocabulario de jugadas (en UCI) observadas realmente en los datos,
    ordenado para que el índice sea reproducible entre corridas."""
    return sorted(set(moves_uci))


def save_policy_vocab(vocab: list, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_policy_vocab(path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def game_id_train_val_split(game_ids, val_fraction: float = 0.1, seed: int = 42):
    """Split train/val por PARTIDA, no por posicion suelta. Si se separa al
    azar por fila, posiciones consecutivas de la misma partida (muy
    correlacionadas entre si) pueden terminar una en train y otra en val,
    inflando la accuracy de validacion sin que el modelo este generalizando
    de verdad. Devuelve (train_idx, val_idx): arrays de indices de fila
    (0-based, mismo orden que `game_ids`)."""
    game_ids = np.asarray(game_ids)
    unique_games = np.unique(game_ids)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_games)
    n_val_games = max(1, int(len(shuffled) * val_fraction))
    val_games = set(shuffled[:n_val_games].tolist())

    is_val = np.array([g in val_games for g in game_ids])
    val_idx = np.nonzero(is_val)[0]
    train_idx = np.nonzero(~is_val)[0]
    return train_idx, val_idx
