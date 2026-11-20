"""Candidato experimental de busqueda: hybrid_epoch4_search_candidate_v1
(Seccion 13 del pedido). NO es operativo, NO se promueve, NO se toca
config.py ni selected_model_manifest. Cambio MINIMO respecto de
hybrid_25m_75v_depth2_topk16: agrega un prior logaritmico de Policy al score
final de cada candidato raiz (lambda_policy=0.15, elegido en train/validado
en test - ver search_variant_results.json). Mismo checkpoint, misma
legalidad, mismos terminales, mismo depth/top_k/material_weight/value_weight
de hoja - la UNICA diferencia es la linea de codigo que suma
lambda_policy*log(policy_prob) al score de cada jugada raiz antes de
rankear.

No incluye libro externo en la metrica de busqueda ni heuristicas manuales
de apertura - solo la propia Policy de la red, ya entrenada."""
from dataclasses import dataclass

import chess

from chess_alpha.search_experiments.variant_search import search_variant

CANDIDATE_NAME = "hybrid_epoch4_search_candidate_v1"
LAMBDA_POLICY = 0.15  # elegido en train (30 posiciones, 63.3% acc) y confirmado en test held-out (73 posiciones, 56.2% acc vs 27.4% del baseline depth2)


class SearchCandidateV1Engine:
    """Misma interfaz que arena.engines.FinalistEngine (get_move(board) ->
    (move, mover_score, stats)) para poder reusar directamente
    play_game/build_finalist_engine-style benchmarks sin duplicar logica de
    Arena/GUI. NO se registra en build_finalist_engine ni en ningun path
    operativo - se instancia solo desde scripts de auditoria."""

    def __init__(self, evaluator, exp_config, lambda_policy: float = LAMBDA_POLICY):
        self.evaluator = evaluator
        self.exp_config = exp_config
        self.lambda_policy = lambda_policy
        self.name = CANDIDATE_NAME
        self.model = evaluator.model

    def get_move(self, board: chess.Board):
        import time
        t0 = time.perf_counter()
        move = search_variant(board, self.evaluator, self.exp_config, "log_prior",
                               lambda_policy=self.lambda_policy)
        think_ms = (time.perf_counter() - t0) * 1000
        if move is None:
            return None, 0.0, {"think_ms": think_ms, "nodes_expanded": None}
        return move, None, {"think_ms": think_ms, "nodes_expanded": None}
