import chess
import chess.engine

CP_MATE_SCORE = 100_000  # score asignado a un mate forzado, para poder ordenar/restar


def _score_to_cp(score: chess.engine.PovScore, pov: bool) -> int:
    """Convierte un PovScore (desde la perspectiva de `pov`) a centipawns,
    mapeando los mates a un valor grande con signo."""
    relative = score.pov(pov)
    if relative.is_mate():
        mate_in = relative.mate()
        sign = 1 if mate_in > 0 else -1
        return sign * (CP_MATE_SCORE - abs(mate_in))
    return relative.score()


def classify_move_quality(centipawn_loss: int) -> str:
    if centipawn_loss <= 10:
        return "excellent"
    elif centipawn_loss <= 30:
        return "good"
    elif centipawn_loss <= 90:
        return "inaccuracy"
    elif centipawn_loss <= 200:
        return "mistake"
    else:
        return "blunder"


class StockfishEvaluator:
    """Wrapper delgado sobre chess.engine para evaluar una posición antes/después
    de una jugada y calcular la pérdida en centipawns respecto a la mejor jugada
    del motor. Requiere un binario de Stockfish instalado localmente."""

    def __init__(self, engine_path: str, depth: int = 12):
        if not engine_path:
            raise ValueError(
                "STOCKFISH_PATH no está configurado. Instalá Stockfish y "
                "seteá la variable de entorno STOCKFISH_PATH con la ruta al "
                "ejecutable (o editá chess_alpha/config.py)."
            )
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.depth = depth

    def evaluate_move(self, fen: str, played_move_uci: str) -> dict:
        board = chess.Board(fen)
        pov = board.turn

        limit = chess.engine.Limit(depth=self.depth)
        info_before = self.engine.analyse(board, limit)
        best_move = info_before["pv"][0]
        eval_before = _score_to_cp(info_before["score"], pov)

        played_move = chess.Move.from_uci(played_move_uci)
        board.push(played_move)
        info_after = self.engine.analyse(board, limit)
        # el eval "after" viene desde la perspectiva del rival; lo invertimos
        # para dejarlo desde la perspectiva de quien jugó `played_move`
        eval_after = -_score_to_cp(info_after["score"], board.turn)

        centipawn_loss = max(0, eval_before - eval_after)

        return {
            "engine_eval_before": eval_before,
            "engine_eval_after": eval_after,
            "centipawn_loss": centipawn_loss,
            "best_move": best_move.uci(),
            "move_quality": classify_move_quality(centipawn_loss),
        }

    def close(self):
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
