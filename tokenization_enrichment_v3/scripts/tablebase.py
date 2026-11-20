"""#25 -- Syzygy tablebase, OPCIONAL. NUNCA descarga nada, NUNCA usa
internet. Si `SYZYGY_PATH` (config.tablebase.syzygy_path_env_var) no esta
configurado, tablebase_available=False para TODAS las filas y el modulo no
intenta abrir nada -- esto NO es una condicion de parada."""
import os

import chess
import chess.syzygy


def open_tablebase(config: dict):
    env_var = config["tablebase"]["syzygy_path_env_var"]
    path = os.environ.get(env_var, "")
    if not path or not os.path.isdir(path):
        return None
    try:
        return chess.syzygy.open_tablebase(path)
    except Exception:
        return None


def compute_tablebase_row(board: chess.Board, tb) -> dict:
    n_pieces = len(board.piece_map())
    if tb is None:
        return {"tablebase_available": False, "tablebase_hit": False, "wdl": None, "dtz": None, "n_pieces": n_pieces}
    try:
        wdl = tb.probe_wdl(board)
        try:
            dtz = tb.probe_dtz(board)
        except Exception:
            dtz = None
        return {"tablebase_available": True, "tablebase_hit": True, "wdl": wdl, "dtz": dtz, "n_pieces": n_pieces}
    except (chess.syzygy.MissingTableError, KeyError):
        return {"tablebase_available": True, "tablebase_hit": False, "wdl": None, "dtz": None, "n_pieces": n_pieces}
