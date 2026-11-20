"""Mascara de legalidad: nunca se construye una matriz densa
posiciones x 20480 -- se calcula al vuelo por posicion, como pide el punto 14."""
import chess

from sequence_policy.action_codec import move_to_action_id


def legal_action_ids(board: chess.Board) -> set:
    return {move_to_action_id(m) for m in board.legal_moves}
