"""Genera el orden determinista de candidatos para el muestreo del piloto.

Metodo: permutacion completa de range(N) con numpy.random.default_rng(seed),
donde N = cantidad total de partidas indexadas por 02_build_game_index.py.
candidate_order[draw_rank] = game_index a intentar en esa posicion.

El pipeline (04_run_tokenization.py) recorre este orden y acepta las primeras
10.000 partidas que resulten validas y no duplicadas -- las que se rechazan
se documentan y se pasa al siguiente candidato determinista, sin volver a
tirar la moneda. No se materializa ningun parseo de partida aqui: esto solo
decide EN QUE ORDEN se van a intentar.
"""
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

SEED = 20260803
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "input"))
INDEX_PATH = os.path.join(INPUT_DIR, "game_index.parquet")
ORDER_PATH = os.path.join(INPUT_DIR, "candidate_order.npy")
MANIFEST_PATH = os.path.join(INPUT_DIR, "candidate_order_manifest.json")


def main():
    manifest_index = json.load(open(os.path.join(INPUT_DIR, "game_index_manifest.json"), encoding="utf-8"))
    n_games = manifest_index["n_games_indexed"]

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n_games).astype(np.int64)
    elapsed = time.time() - t0

    tmp_path = ORDER_PATH + ".tmp"
    np.save(tmp_path, order)
    # numpy agrega '.npy' si no estaba -- normalizamos el nombre resultante.
    if not tmp_path.endswith(".npy"):
        tmp_path = tmp_path + ".npy"
    reloaded = np.load(tmp_path)
    assert reloaded.shape[0] == n_games
    assert len(set(reloaded[:1000].tolist())) == len(reloaded[:1000])  # sin repetidos obvios
    os.replace(tmp_path, ORDER_PATH)

    manifest = {
        "seed": SEED,
        "method": "numpy.random.default_rng(seed).permutation(N)",
        "n_candidates": int(n_games),
        "source_index": INDEX_PATH,
        "elapsed_seconds": round(elapsed, 3),
        "note": (
            "candidate_order[draw_rank] = game_index. El pipeline recorre "
            "draw_rank = 0, 1, 2, ... y acepta las primeras 10000 partidas "
            "validas y unicas que encuentra, documentando cada rechazo."
        ),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Orden de candidatos generado: N={n_games}, seed={SEED}, {elapsed:.3f}s")
    print(f"Primeros 10 draw_rank -> game_index: {order[:10].tolist()}")
    print(f"Escrito en {ORDER_PATH}")


if __name__ == "__main__":
    main()
