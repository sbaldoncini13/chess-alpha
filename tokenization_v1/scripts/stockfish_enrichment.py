"""PASO 21-27: enriquecimiento OPCIONAL y desacoplado via Stockfish.

Arquitectura (no negociable, ver encargo): TOKENIZATION BASE -> DATASET BASE
-> OPTIONAL STOCKFISH ENRICHMENT. Este modulo nunca se importa desde el
pipeline base (06_enrich_positions.py) salvo que enable_stockfish=true en la
config; si Stockfish no esta instalado/configurado, el resto del pipeline
sigue funcionando sin tocar este archivo.

Convencion de score/perspectiva y CP_MATE_SCORE mirror
src/chess_alpha/engine_eval.py (solo lectura, no importada -- reimplementada
aca para que tokenization_v1/ siga siendo un modulo independiente), con
thresholds configurables en vez de hardcodeados.
"""
import chess
import chess.engine

from tokenizer import move_to_action_id

CP_MATE_SCORE = 100_000


def score_to_cp(score: "chess.engine.PovScore", pov: bool) -> int:
    """Convierte un PovScore (relativo) a centipawns absolutos desde `pov`,
    mapeando mates a un valor grande con signo (nunca se resta un cp real
    contra un mate sin pasar por esta conversion primero)."""
    relative = score.pov(pov)
    if relative.is_mate():
        mate_in = relative.mate()
        sign = 1 if mate_in > 0 else -1
        return sign * (CP_MATE_SCORE - abs(mate_in))
    return relative.score()


def is_mate_score(score: "chess.engine.PovScore", pov: bool) -> bool:
    return score.pov(pov).is_mate()


def mate_in(score: "chess.engine.PovScore", pov: bool):
    rel = score.pov(pov)
    return rel.mate() if rel.is_mate() else None


def classify_move_quality(centipawn_loss: int, thresholds: dict) -> str:
    if centipawn_loss <= thresholds["best"]:
        return "best"
    elif centipawn_loss <= thresholds["excellent"]:
        return "excellent"
    elif centipawn_loss <= thresholds["good"]:
        return "good"
    elif centipawn_loss <= thresholds["inaccuracy"]:
        return "inaccuracy"
    elif centipawn_loss <= thresholds["mistake"]:
        return "mistake"
    return "blunder"


def softmax_policy_from_topk(topk: list, temperature_cp: float) -> list:
    """topk: lista de dicts {move_id, score_cp} YA en cp desde la
    perspectiva de quien mueve (ver score_to_cp). Softmax con temperatura
    configurable -- NO se inventan probabilidades: es una transformacion
    documentada y determinista de los scores del motor."""
    import math
    if not topk:
        return []
    scores = [c["score_cp"] / temperature_cp for c in topk]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [
        {"move_id": c["move_id"], "weight": e / total}
        for c, e in zip(topk, exps)
    ]


class StockfishEnricher:
    """Wrapper de chess.engine para evaluar una posicion con MultiPV y
    calcular la perdida en centipawns del movimiento humano respecto al
    mejor candidato del motor. Requiere un binario de Stockfish instalado
    localmente (STOCKFISH_PATH) -- si no esta disponible, este objeto
    simplemente no se instancia y el resto del pipeline no se ve afectado.
    """

    def __init__(self, engine_path: str, depth: int, multipv: int):
        if not engine_path:
            raise ValueError(
                "STOCKFISH_PATH no configurado. Este enriquecimiento es "
                "opcional -- el pipeline base no requiere Stockfish."
            )
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.depth = depth
        self.multipv = multipv

    def analyse_position(self, fen: str) -> dict:
        """Evalua la posicion (antes de cualquier movimiento) y devuelve
        top-k candidatos (PASO 23) + mejor jugada (PASO 22). No juega ningun
        movimiento humano todavia -- eso se hace en evaluate_human_move."""
        board = chess.Board(fen)
        pov = board.turn
        limit = chess.engine.Limit(depth=self.depth)
        infos = self.engine.analyse(board, limit, multipv=self.multipv)
        if isinstance(infos, dict):
            infos = [infos]

        topk = []
        for rank, info in enumerate(infos, start=1):
            pv = info.get("pv")
            if not pv:
                continue
            move = pv[0]
            cp = score_to_cp(info["score"], pov)
            topk.append({
                "rank": rank,
                "move": move,
                "move_id": move_to_action_id(move),
                "move_uci": move.uci(),
                "score_cp": cp,
                "is_mate": is_mate_score(info["score"], pov),
                "mate_in": mate_in(info["score"], pov),
            })

        return {
            "fen": fen,
            "depth": self.depth,
            "multipv": self.multipv,
            "topk": topk,
            "best_move": topk[0]["move"] if topk else None,
            "best_move_id": topk[0]["move_id"] if topk else None,
            "best_move_uci": topk[0]["move_uci"] if topk else None,
            "best_eval_cp": topk[0]["score_cp"] if topk else None,
        }

    def evaluate_human_move(self, fen: str, human_move_uci: str, quality_thresholds: dict) -> dict:
        """PASO 25-26: centipawn_loss y clasificacion del movimiento humano
        respecto al mejor candidato del motor EN ESTA posicion."""
        analysis = self.analyse_position(fen)
        best_eval = analysis["best_eval_cp"]

        board = chess.Board(fen)
        pov = board.turn
        played_move = chess.Move.from_uci(human_move_uci)
        board.push(played_move)
        limit = chess.engine.Limit(depth=self.depth)
        info_after = self.engine.analyse(board, limit)
        # eval "after" es desde la perspectiva del rival; se invierte para
        # dejarlo desde la perspectiva de quien jugo el movimiento humano
        eval_after_stm = -score_to_cp(info_after["score"], board.turn)

        centipawn_loss = max(0, best_eval - eval_after_stm) if best_eval is not None else None
        quality = classify_move_quality(centipawn_loss, quality_thresholds) if centipawn_loss is not None else None

        return {
            **analysis,
            "human_move_uci": human_move_uci,
            "human_eval_after_cp": eval_after_stm,
            "centipawn_loss": centipawn_loss,
            "move_quality": quality,
        }

    def close(self):
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
