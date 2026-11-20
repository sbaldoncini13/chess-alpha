"""Negamax de referencia de search_v1 (Fase II) - deliberadamente simple,
SIN ninguna poda. Es el oraculo interno contra el que se valida
alpha_beta.py en profundidades pequenas (mismo score siempre, ver
tests/search_v1/test_negamax.py y test_alpha_beta_vs_negamax.py).

Orden de exploracion: UCI determinista simple (sorted por string) - a
diferencia de alpha_beta.py, el orden NO afecta el resultado (negamax no
poda, explora todas las jugadas siempre), solo puede afectar CUAL de
varias jugadas empatadas en score exacto se reporta como best_move (se
queda con la PRIMERA en orden UCI que iguale el mejor score encontrado
hasta el momento, por el uso de comparacion estricta `>`)."""

from __future__ import annotations

from search_v1.mate_scores import terminal_score
from search_v1.move_generator import generate_legal_moves
from search_v1.position import Position
from search_v1.search_context import SearchContext
from search_v1.static_evaluation import evaluate_static_relative
from search_v1.terminal import detect_terminal, is_fifty_move_rule


def negamax(position: Position, depth: int, context: SearchContext) -> tuple[int, list[str]]:
    """Devuelve (score, principal_variation) en la perspectiva del lado a
    mover en `position`. NO muta `position` mas alla de la duracion de la
    llamada (todo apply() tiene su undo() correspondiente, incluso si
    SearchAborted se propaga)."""
    context.visit_node()

    terminal = detect_terminal(position)
    if terminal.is_terminal:
        context.terminal_nodes += 1
        return terminal_score(terminal, context.ply_from_root), []

    if is_fifty_move_rule(position):
        context.terminal_nodes += 1
        return 0, []

    if context.repetition.is_threefold_repetition(position):
        context.terminal_nodes += 1
        return 0, []

    if depth <= 0:
        context.leaf_nodes += 1
        return evaluate_static_relative(position), []

    best_score = None
    best_pv: list[str] = []
    moves = sorted(generate_legal_moves(position), key=lambda lm: lm.uci)

    for lm in moves:
        position.apply(lm.move)
        context.repetition.push(position)
        context.ply_from_root += 1
        try:
            child_score, child_pv = negamax(position, depth - 1, context)
        finally:
            context.ply_from_root -= 1
            context.repetition.pop()
            position.undo()

        score = -child_score
        if best_score is None or score > best_score:
            best_score = score
            best_pv = [lm.uci] + child_pv

    return best_score, best_pv
