"""Variantes experimentales de busqueda (Secciones 7+8 del pedido) - NINGUNA
es operativa, todas viven aca aisladas. Reusan integramente
search/leaf_evaluation.py (material_normalized, value_eval,
evaluate_leaf_for_experiment) y el patron de search/negamax.py::_search -
no se reimplementa la logica base, solo se parametriza."""
import math

import chess

from search.leaf_evaluation import evaluate_leaf_for_experiment


def _minimax(board, depth, root_color, exp_config, model, alpha=-math.inf, beta=math.inf):
    if depth == 0 or board.is_game_over():
        return evaluate_leaf_for_experiment(board, root_color, exp_config, model)
    maximizing = board.turn == root_color
    value = -math.inf if maximizing else math.inf
    for move in board.legal_moves:
        board.push(move)
        score = _minimax(board, depth - 1, root_color, exp_config, model, alpha, beta)
        board.pop()
        if maximizing:
            value = max(value, score)
            alpha = max(alpha, value)
        else:
            value = min(value, score)
            beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def search_variant(board, evaluator, exp_config, variant: str, root_color=None,
                    lambda_policy: float = 0.10, epsilon_tie: float = 0.05, rank_bonus_lambda: float = 0.0):
    """variant:
      'policy_only'   -> top-1 de la Policy, sin busqueda.
      'depth1'        -> minimax profundidad 1 (root move + eval directa, sin modelar respuesta del rival).
      'depth2'        -> identico al operativo (baseline de comparacion).
      'depth3_sample' -> minimax profundidad 3 (caro, solo para muestra chica).
      'material_only' -> exp_config con leaf_mode forzado a material.
      'value_only'    -> exp_config con leaf_mode forzado a value.
      'log_prior'     -> depth2 + lambda_policy*log(policy_prob+eps) sumado al score de cada candidato raiz.
      'rank_bonus'    -> depth2 + rank_bonus_lambda/rank_policy sumado al score de cada candidato raiz.
      'tie_break'     -> depth2, pero si dos candidatos raiz quedan a <= epsilon_tie de distancia, desempata por Policy.
    """
    root_color = root_color if root_color is not None else board.turn

    if variant == "policy_only":
        top1 = evaluator.top_k_moves(board, k=1)
        return top1[0] if top1 else None

    policy_probs = None
    if variant in ("log_prior", "rank_bonus", "tie_break"):
        policy_probs, _ = evaluator.predict(board)
        ranked = sorted(policy_probs.items(), key=lambda kv: kv[1], reverse=True)
        policy_rank = {uci: i + 1 for i, (uci, _) in enumerate(ranked)}

    if variant == "material_only":
        exp_config = exp_config.__class__(**{**exp_config.__dict__, "leaf_mode": "material"})
        model = None
        depth = exp_config.depth
    elif variant == "value_only":
        exp_config = exp_config.__class__(**{**exp_config.__dict__, "leaf_mode": "value"})
        model = evaluator.model
        depth = exp_config.depth
    else:
        model = evaluator.model if exp_config.leaf_mode in ("value", "hybrid") else None
        depth = {"depth1": 1, "depth2": 2, "depth3_sample": 3}.get(variant, exp_config.depth)

    root_candidates = evaluator.top_k_moves(board, k=exp_config.top_k)
    if not root_candidates:
        return None

    scored = []
    for move in root_candidates:
        board.push(move)
        score = _minimax(board, depth - 1, root_color, exp_config, model)
        board.pop()

        if variant == "log_prior":
            p = policy_probs.get(move.uci(), 1e-6)
            score = score + lambda_policy * math.log(p + 1e-6)
        elif variant == "rank_bonus":
            r = policy_rank.get(move.uci(), len(root_candidates) + 1)
            score = score + rank_bonus_lambda / r

        scored.append((score, move))

    scored.sort(key=lambda sm: sm[0], reverse=True)

    if variant == "tie_break" and len(scored) > 1:
        best_score = scored[0][0]
        tied = [(s, m) for s, m in scored if abs(s - best_score) <= epsilon_tie]
        if len(tied) > 1:
            tied.sort(key=lambda sm: policy_rank.get(sm[1].uci(), 999))
            return tied[0][1]

    return scored[0][1] if scored else None
