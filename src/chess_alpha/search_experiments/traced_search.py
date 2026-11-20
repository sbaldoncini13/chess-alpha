"""Busqueda instrumentada (SOLO lectura/diagnostico, no operativa) que
replica EXACTAMENTE la logica de search/negamax.py::_search para
exp_config.depth=2 pero graba el arbol completo: cada jugada raiz (limitada
a top_k por la Policy, igual que el motor real), TODAS las respuestas
legales del rival en el nodo hijo (el motor real NO filtra al rival por
top_k - explora todas sus jugadas legales con evaluate_leaf_for_experiment
directo, ver docstring de negamax.py y el propio codigo: en el nivel
is_root=False se usa `list(board.legal_moves)`, no `top_k_moves`), y el
detalle material/value/hybrid de cada hoja.

No se importa desde ningun modulo operativo. Se puede borrar sin afectar
nada real."""
import chess

from search.leaf_evaluation import evaluate_leaf_for_experiment, material_normalized, value_eval


def trace_depth2(board: chess.Board, evaluator, exp_config, root_color=None):
    """Devuelve un dict con el trazado COMPLETO de una busqueda depth=2
    identica a search_best_move_experiment, pero con todo el detalle
    intermedio. Asume exp_config.depth == 2 (si es distinto, igual traza
    fielmente la profundidad configurada, pero el diseño de este trazador
    esta pensado para depth=2)."""
    root_color = root_color if root_color is not None else board.turn
    model = evaluator.model if exp_config.leaf_mode in ("value", "hybrid") else None

    policy_probs, _ = evaluator.predict(board)
    ranked_policy = sorted(policy_probs.items(), key=lambda kv: kv[1], reverse=True)
    policy_rank = {uci: i + 1 for i, (uci, _) in enumerate(ranked_policy)}

    root_candidates = evaluator.top_k_moves(board, k=exp_config.top_k)

    root_moves_trace = []
    for root_move in root_candidates:
        root_san = board.san(root_move)
        board.push(root_move)

        maximizing_at_child = board.turn == root_color  # normalmente False si el rival mueve en el hijo
        responses_trace = []
        best_response_score = None
        best_response_move = None

        if board.is_game_over() or exp_config.depth <= 1:
            leaf_score = evaluate_leaf_for_experiment(board, root_color, exp_config, model)
            best_response_score = leaf_score
        else:
            for reply in list(board.legal_moves):
                reply_san = board.san(reply)
                board.push(reply)
                material_norm = material_normalized(board, root_color, exp_config.material_scale_cp)
                _, value_norm = (None, None)
                if model is not None:
                    _, value_norm = value_eval(board, root_color, model)
                leaf_score = evaluate_leaf_for_experiment(board, root_color, exp_config, model)
                responses_trace.append({
                    "uci": reply.uci(), "san": reply_san, "material_normalized": material_norm,
                    "value_normalized": value_norm, "hybrid_score": leaf_score,
                })
                board.pop()

                if best_response_score is None:
                    best_response_score, best_response_move = leaf_score, reply
                elif maximizing_at_child and leaf_score > best_response_score:
                    best_response_score, best_response_move = leaf_score, reply
                elif not maximizing_at_child and leaf_score < best_response_score:
                    best_response_score, best_response_move = leaf_score, reply

        board.pop()
        root_moves_trace.append({
            "uci": root_move.uci(), "san": root_san,
            "policy_probability": policy_probs.get(root_move.uci()),
            "policy_rank": policy_rank.get(root_move.uci()),
            "n_replies_considered": len(responses_trace),
            "worst_case_reply_for_opponent_view": {
                "uci": best_response_move.uci() if best_response_move else None,
                "san": board.san(best_response_move) if best_response_move else None,
            } if best_response_move else None,
            "score_after_search": best_response_score,
            "replies_detail": responses_trace,
        })

    root_moves_trace.sort(key=lambda m: m["score_after_search"], reverse=True)
    for i, m in enumerate(root_moves_trace):
        m["search_rank"] = i + 1

    chosen = root_moves_trace[0] if root_moves_trace else None
    return {
        "root_fen": board.fen(), "root_color": "white" if root_color else "black",
        "root_candidates_considered": len(root_candidates),
        "chosen_move": {"uci": chosen["uci"], "san": chosen["san"]} if chosen else None,
        "policy_top1_uci": ranked_policy[0][0] if ranked_policy else None,
        "root_moves": root_moves_trace,
    }
