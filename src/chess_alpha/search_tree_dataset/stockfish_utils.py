"""Wrapper Stockfish MultiPV compartido - factoriza el patron repetido en 7
scripts del proyecto (22/40/43/44/benchmark_block3_grid/benchmark_policy_coverage/111)
que nunca se habia extraido a un modulo comun. Uso EXCLUSIVO de etiquetado
offline (Fase 13 del pedido de SEARCH TREE DATASET V1.1) - nunca para jugar
durante una partida real del motor."""

import chess
import chess.engine


def open_engine(stockfish_path: str):
    return chess.engine.SimpleEngine.popen_uci(stockfish_path)


def sf_analyse(engine, board: chess.Board, depth: int, multipv: int):
    n = min(multipv, board.legal_moves.count())
    if n == 0:
        return None
    return engine.analyse(board, chess.engine.Limit(depth=depth), multipv=n)


def score_cp_or_mate(info_entry, color):
    """Devuelve (score_cp, mate_distance) - exactamente uno de los dos es
    no-None. mate_distance > 0 = mate a favor de `color`, < 0 = en contra."""
    pov = info_entry["score"].pov(color)
    if pov.is_mate():
        return None, pov.mate()
    return pov.score(), None
