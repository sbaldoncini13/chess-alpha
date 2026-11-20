import math
import chess

from chess_alpha.board_features import material_score

CHECKMATE_SCORE = 10_000.0


def material_eval(board: chess.Board) -> float:
    """Evaluación clásica desde la perspectiva de blancas (positivo = mejor
    para blancas). Sirve de fallback cuando no hay modelo entrenado."""
    if board.is_checkmate():
        return -CHECKMATE_SCORE if board.turn == chess.WHITE else CHECKMATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    white_material = material_score(board, chess.WHITE)
    black_material = material_score(board, chess.BLACK)
    mobility = board.legal_moves.count() * (1 if board.turn == chess.WHITE else -1)
    return (white_material - black_material) + 0.01 * mobility


def alpha_beta(board, depth, alpha=-math.inf, beta=math.inf, eval_fn=material_eval, move_provider=None):
    """Minimax con poda alfa-beta. `move_provider(board)` opcionalmente
    restringe/ordena las jugadas a explorar en la raíz (p. ej. las
    candidatas top-k de la red neuronal); si no se pasa, explora todas las
    jugadas legales."""
    if depth == 0 or board.is_game_over():
        return eval_fn(board), None

    moves = move_provider(board) if move_provider else list(board.legal_moves)
    if not moves:
        return eval_fn(board), None

    maximizing = board.turn == chess.WHITE
    best_move = moves[0]
    value = -math.inf if maximizing else math.inf

    for move in moves:
        board.push(move)
        score, _ = alpha_beta(board, depth - 1, alpha, beta, eval_fn, move_provider)
        board.pop()

        if maximizing and score > value:
            value, best_move = score, move
            alpha = max(alpha, value)
        elif not maximizing and score < value:
            value, best_move = score, move
            beta = min(beta, value)

        if alpha >= beta:
            break

    return value, best_move


def best_move(board, depth=3, eval_fn=material_eval, move_provider=None):
    _, move = alpha_beta(board, depth, eval_fn=eval_fn, move_provider=move_provider)
    return move


def neural_value_eval(board: chess.Board, evaluator) -> float:
    """Evalua la posición usando el value head de la red (perspectiva
    blancas, en [0,1]) en vez de material puro. Los finales de partida se
    manejan explícitamente para no depender de que la red los haya visto
    bien representados en el dataset de entrenamiento."""
    if board.is_checkmate():
        return 0.0 if board.turn == chess.WHITE else 1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    return evaluator.predict_value(board)


def neural_guided_best_move(
    board: chess.Board,
    evaluator,
    depth: int = 2,
    top_k: int = 8,
    use_value_head: bool = False,
    filter_every_node: bool = False,
):
    """El modelo propone top-k jugadas candidatas en la raíz (policy head).
    Por defecto, alpha-beta evalua las hojas con material puro (rapido,
    ~0.2s/jugada). Con `use_value_head=True`, evalua cada hoja con el value
    head de la red en vez de material.

    `filter_every_node=False` (default): el filtro top-k solo se aplica en
    la raiz; los demas niveles exploran todas las jugadas legales (~35 en
    promedio) - mas seguro tacticamente, pero con use_value_head=True se
    vuelve 200-300x mas lento (miles de pasadas de red por jugada).

    `filter_every_node=True`: aplica el filtro top-k en TODOS los niveles
    del arbol (llama a la red en cada nodo interno para elegir sus propias
    candidatas), no solo la raiz. Reduce muchisimo la cantidad de hojas
    exploradas (de ~35^depth a top_k^depth), haciendo practico usar
    use_value_head=True. A cambio, la busqueda puede no ver una jugada
    tactica fuerte que la policy no haya puntuado alto en algun nivel
    intermedio."""
    eval_fn = (lambda b: neural_value_eval(b, evaluator)) if use_value_head else material_eval

    if filter_every_node:
        move_provider = lambda b: evaluator.top_k_moves(b, k=top_k)
    else:
        # board se muta in-place (push/pop) durante la recursion, asi que
        # comparamos la profundidad de la pila de jugadas contra la raiz
        # para distinguir "estoy en la raiz" de "estoy mas abajo" - en la
        # raiz uso las candidatas de la red, mas abajo, todas las legales.
        candidates = evaluator.top_k_moves(board, k=top_k)
        root_ply = len(board.move_stack)
        move_provider = lambda b: candidates if len(b.move_stack) == root_ply else list(b.legal_moves)

    return best_move(board, depth=depth, eval_fn=eval_fn, move_provider=move_provider)
