"""Compone StockfishEnricher + EngineCache para producir un dict con
EXACTAMENTE la misma forma que StockfishEnricher.evaluate_human_move
(stockfish_enrichment.py, sin modificar), pero permitiendo que la parte
root (multipv) y la parte child (eval tras el movimiento humano) se
cacheen de forma INDEPENDIENTE.

No se reimplementa ninguna formula: analyse_position() se llama tal cual
esta en stockfish_enrichment.py en un cache-miss root; score_to_cp/
classify_move_quality se reutilizan sin cambios para el child y para
centipawn_loss/move_quality. Cache = solo evita repetir la llamada UCI
cuando la clave (FEN exacto + config) ya se vio antes en esta corrida.
"""
import chess
import chess.engine

from engine_cache import REQUEST_TYPE_CHILD, REQUEST_TYPE_ROOT, EngineCache, compute_request_key
from stockfish_enrichment import StockfishEnricher, classify_move_quality, score_to_cp


def analyse_human_move_cached(sf: StockfishEnricher, cache: EngineCache, fen: str, human_move_uci: str,
                               engine_version: str, engine_binary_sha256: str,
                               depth: int, multipv: int, child_multipv: int,
                               threads: int, hash_mb: int, quality_thresholds: dict) -> dict:
    root_key = compute_request_key(
        fen, engine_version, engine_binary_sha256, depth, multipv, threads, hash_mb, REQUEST_TYPE_ROOT
    )
    cached_root = cache.get_root(root_key)
    if cached_root is None:
        analysis = sf.analyse_position(fen)
        payload = {
            "best_move_id": analysis["best_move_id"], "best_move_uci": analysis["best_move_uci"],
            "topk": [
                {"rank": c["rank"], "move_id": c["move_id"], "move_uci": c["move_uci"],
                 "score_cp": c["score_cp"], "is_mate": c["is_mate"], "mate_in": c["mate_in"]}
                for c in analysis["topk"]
            ],
        }
        cache.put_root(root_key, fen, payload)
    else:
        payload = cached_root

    board = chess.Board(fen)
    board.push(chess.Move.from_uci(human_move_uci))
    child_fen = board.fen()

    child_key = compute_request_key(
        child_fen, engine_version, engine_binary_sha256, depth, child_multipv, threads, hash_mb, REQUEST_TYPE_CHILD
    )
    cached_child = cache.get_child(child_key)
    if cached_child is None:
        limit = chess.engine.Limit(depth=depth)
        info_after = sf.engine.analyse(board, limit)
        # misma inversion de perspectiva que evaluate_human_move original
        eval_after_stm = -score_to_cp(info_after["score"], board.turn)
        cache.put_child(child_key, child_fen, {"eval_after_stm": eval_after_stm})
    else:
        eval_after_stm = cached_child["eval_after_stm"]

    best_eval = payload["topk"][0]["score_cp"] if payload["topk"] else None
    centipawn_loss = max(0, best_eval - eval_after_stm) if best_eval is not None else None
    quality = classify_move_quality(centipawn_loss, quality_thresholds) if centipawn_loss is not None else None

    return {
        "topk": payload["topk"],
        "best_move_id": payload["best_move_id"], "best_move_uci": payload["best_move_uci"],
        "best_eval_cp": best_eval,
        "human_move_uci": human_move_uci, "human_eval_after_cp": eval_after_stm,
        "centipawn_loss": centipawn_loss, "move_quality": quality,
    }
