"""Busqueda minimax con poda alfa-beta, en perspectiva ROOT fija.

Aclaracion de nombre: se llama negamax.py para seguir la estructura de
archivos pedida, pero la implementacion NO es negamax clasico (el que
niega el score en cada nivel, `-negamax(child)`) - es minimax con
maximizacion/minimizacion alternada segun de quien es el turno respecto a
root_color. Esto es deliberado: leaf_evaluation.py devuelve todo en UNA
sola perspectiva fija (root), no en la perspectiva de "quien mueve en cada
nodo" que negamax clasico requiere para que el sign-flip tenga sentido.
Si mas adelante (bloque 12, PVS) hace falta la forma negamax "de libro",
se hace una conversion en ese momento - no antes.

Filtro top-k: en esta primera version SOLO se aplica en la raiz (moves de
profundidad 0). Los niveles mas profundos exploran todas las jugadas
legales. La variante "filtrar en todos los niveles" (mas rapida pero mas
ciega tacticamente, ya la probamos aparte en 31_search_ablation_diagnostic.py)
no esta incluida aca todavia.
"""

import math
import time

import chess

from search.leaf_evaluation import LeafEvalConfig, evaluate_leaf, evaluate_leaf_for_experiment

# Umbral para detectar un score de hoja "extremo" (mate forzado dentro del
# horizonte de busqueda) en la escala normalizada [-1,1] de
# evaluate_leaf_for_experiment: un mate real satura exactamente a +-1.0
# (material_normalized satura via tanh, value_eval devuelve +-1.0 exacto en
# jaque mate) - una posicion NO terminal con ventaja material/valor enorme
# se acerca a +-1.0 pero nunca lo alcanza exactamente en la practica (ver
# self-test de leaf_evaluation.py: tanh(7200/600)=0.9999999958..., no 1.0).
# Se usa un margen chico (no "==") solo por prudencia ante redondeo de
# punto flotante, no porque el limite real sea ambiguo.
POLICY_PRIOR_MATE_BOUNDARY = 1.0 - 1e-9


def search_best_move(board, evaluator, config: LeafEvalConfig, depth: int = 2, top_k: int = 8, root_color=None):
    """Devuelve la mejor jugada segun minimax + poda alfa-beta, evaluando
    hojas con `config` (material/value/hybrid). `evaluator` es un
    NeuralEvaluator/HybridNeuralEvaluator (se usa su .model para value, y
    su top_k_moves para filtrar la raiz - incluso en modo material, para
    que la comparacion entre modos sea sobre las MISMAS candidatas de raiz)."""
    if root_color is None:
        root_color = board.turn

    root_candidates = evaluator.top_k_moves(board, k=top_k)
    if not root_candidates:
        return None

    model = evaluator.model if config.mode in ("value", "hybrid") else None
    _, move = _search(board, depth, -math.inf, math.inf, root_color, config, model, root_candidates, is_root=True)
    return move


def _search(board, depth, alpha, beta, root_color, config, model, root_candidates, is_root,
            eval_fn=evaluate_leaf, stats=None,
            policy_prior_enabled=False, policy_prior_lambda=0.0, policy_prior_epsilon=1e-8,
            policy_probs=None, policy_prior_trace=None, always_trace_root=False):
    if depth == 0 or board.is_game_over():
        if stats is not None:
            t0 = time.perf_counter()
            score = eval_fn(board, root_color, config, model)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            stats["leaf_evaluations"] += 1
            if model is not None:
                stats["value_inference_time_ms"] += elapsed_ms
            else:
                stats["material_eval_time_ms"] += elapsed_ms
            return score, None
        return eval_fn(board, root_color, config, model), None

    moves = root_candidates if is_root else list(board.legal_moves)
    if not moves:
        if stats is not None:
            stats["leaf_evaluations"] += 1
        return eval_fn(board, root_color, config, model), None

    if stats is not None:
        stats["nodes_expanded"] += 1
        stats["branching_samples"].append(len(moves))

    maximizing = board.turn == root_color
    best_move = moves[0]
    value = -math.inf if maximizing else math.inf

    # El prior SOLO se aplica en el nivel raiz (is_root=True, que ocurre
    # unicamente en la primera llamada - las llamadas recursivas siempre
    # pasan is_root=False) - Fase E del pedido, "policy_prior_root_only".
    # Cuando policy_prior_enabled=False (default), esta rama nunca se activa
    # y el bucle de abajo ejecuta exactamente el mismo codigo que antes.
    apply_prior_here = is_root and policy_prior_enabled
    # always_trace_root (Fase 6, benchmark de aperturas): permite pedir el
    # trace de candidatos raiz TAMBIEN cuando el prior esta deshabilitado
    # (para poder rankear el baseline igual que al candidato y calcular MRR
    # comparable) SIN tocar el contrato ya auditado en
    # 107_policy_prior_equivalence_test.py: ese test nunca pasa
    # always_trace_root=True, asi que con el default (False) esta rama es
    # identica a como era antes - is_extreme/prior_contribution siguen sin
    # calcularse y el trace sigue quedando vacio con enabled=False.
    want_trace_here = is_root and policy_prior_trace is not None and (policy_prior_enabled or always_trace_root)

    for move in moves:
        board.push(move)
        need_terminal_check = apply_prior_here or want_trace_here
        immediately_terminal = need_terminal_check and board.is_game_over()
        score, _ = _search(board, depth - 1, alpha, beta, root_color, config, model, root_candidates,
                            is_root=False, eval_fn=eval_fn, stats=stats)
        board.pop()

        if need_terminal_check:
            # Proteccion de terminales (Fase 3 del pedido): un mate forzado
            # (descubierto en cualquier hoja dentro del horizonte de
            # busqueda, o la jugada raiz siendo terminal de inmediato) SATURA
            # a +-1.0 exacto via evaluate_leaf_for_experiment - el prior
            # nunca debe alterar eso, ni hacer que una jugada no-terminal
            # parezca mejor que un mate real, ni al reves.
            is_extreme = immediately_terminal or abs(score) >= POLICY_PRIOR_MATE_BOUNDARY
        else:
            is_extreme = False

        if apply_prior_here:
            if is_extreme:
                prior_contribution = 0.0
            else:
                p = policy_probs.get(move.uci(), 0.0)
                prior_contribution = policy_prior_lambda * math.log(p + policy_prior_epsilon)
            score_for_ranking = score + prior_contribution
        else:
            prior_contribution = 0.0
            score_for_ranking = score

        if want_trace_here:
            policy_prior_trace.append({
                "uci": move.uci(),
                "san": board.san(move),
                "policy_probability": (policy_probs.get(move.uci(), 0.0) if policy_probs is not None else None),
                "score_search": score,
                "prior_contribution": prior_contribution,
                "score_final": score_for_ranking,
                "is_extreme_protected": is_extreme,
            })

        if maximizing and score_for_ranking > value:
            value, best_move = score_for_ranking, move
            alpha = max(alpha, value)
        elif not maximizing and score_for_ranking < value:
            value, best_move = score_for_ranking, move
            beta = min(beta, value)

        if alpha >= beta:
            break

    return value, best_move


def search_best_move_experiment(board, evaluator, exp_config, root_color=None, stats=None,
                                 policy_prior_trace=None, always_trace_root=False):
    """Igual que search_best_move, pero toma un SearchExperimentConfig
    (src/configs/block3_experiments.py) - depth/top_k/leaf_mode/pesos ya
    vienen todos juntos en el config, y evalua las hojas con
    evaluate_leaf_for_experiment (normalizado a [-1,1], no en centipawns).
    Si se pasa `stats` (dict), se le suman nodes_expanded/leaf_evaluations/
    branching_samples durante la busqueda.

    Si `exp_config.policy_prior_enabled` (default False, preserva el
    comportamiento operativo exacto): suma lambda_policy*log(prob+eps) al
    score de cada candidato RAIZ antes de rankear (Fase E del pedido de
    integracion real del prior). `policy_prior_trace`, si se pasa una lista,
    se llena con el detalle por candidato (UCI/SAN/policy_probability/
    score_search/prior_contribution/score_final/proteccion de terminal) para
    auditoria - no tiene efecto en la decision, solo instrumentacion.

    `always_trace_root` (default False, Fase 6 - benchmark de aperturas):
    si es True Y se paso `policy_prior_trace`, llena el trace TAMBIEN cuando
    el prior esta deshabilitado (para poder rankear el baseline con el mismo
    mecanismo que el candidato y calcular MRR comparable) - agrega UN forward
    pass extra de Policy en ese caso especifico (no se usa para elegir
    root_candidates del baseline, que sigue siendo evaluator.top_k_moves(),
    solo para poblar policy_probability en el trace). Con el default False,
    o sin pasar policy_prior_trace, el comportamiento es IDENTICO a antes de
    este parametro (contrato validado por
    107_policy_prior_equivalence_test.py, que nunca lo pasa)."""
    if root_color is None:
        root_color = board.turn

    if stats is not None:
        stats.setdefault("nodes_expanded", 0)
        stats.setdefault("leaf_evaluations", 0)
        stats.setdefault("branching_samples", [])
        stats.setdefault("value_inference_time_ms", 0.0)
        stats.setdefault("material_eval_time_ms", 0.0)
        t_policy0 = time.perf_counter()

    need_policy_probs = exp_config.policy_prior_enabled or (policy_prior_trace is not None and always_trace_root)

    policy_probs = None
    if need_policy_probs:
        # Un solo forward pass de Policy: se reusa `policy_probs` tanto para
        # el top-k de raiz (si el prior esta habilitado) como para el prior
        # y/o el trace, en vez de llamar evaluator.top_k_moves() (que
        # internamente vuelve a llamar evaluator.predict() y duplicaria la
        # inferencia de Policy sin necesidad - la latencia de referencia
        # anterior, ~2.2x, era en buena parte por esto, ver Fase 9 del
        # pedido). Cuando el prior esta deshabilitado pero se pidio trace
        # (always_trace_root=True), este es un forward pass EXTRA solo para
        # instrumentacion - no cambia root_candidates del baseline.
        policy_probs, _ = evaluator.predict(board)

    if exp_config.policy_prior_enabled:
        ranked = sorted(board.legal_moves, key=lambda m: policy_probs.get(m.uci(), 0.0), reverse=True)
        root_candidates = ranked[:exp_config.top_k]
    else:
        root_candidates = evaluator.top_k_moves(board, k=exp_config.top_k)

    if stats is not None:
        stats["policy_inference_time_ms"] = (time.perf_counter() - t_policy0) * 1000

    if not root_candidates:
        return None

    model = evaluator.model if exp_config.leaf_mode in ("value", "hybrid") else None

    _, move = _search(
        board, exp_config.depth, -math.inf, math.inf, root_color, exp_config, model, root_candidates,
        is_root=True, eval_fn=evaluate_leaf_for_experiment, stats=stats,
        policy_prior_enabled=exp_config.policy_prior_enabled,
        policy_prior_lambda=exp_config.policy_prior_lambda,
        policy_prior_epsilon=exp_config.policy_prior_epsilon,
        policy_probs=policy_probs, policy_prior_trace=policy_prior_trace,
        always_trace_root=always_trace_root,
    )
    return move
