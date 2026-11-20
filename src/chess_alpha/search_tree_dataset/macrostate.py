"""Clasificador de macroestados (Bloque metodologico - Markov, transitions.parquet)
a partir de senales YA existentes en el proyecto (game_phase de
board_features.py + flags tacticos ya calculados durante el labeling) -
heuristico y documentado, NO un modelo entrenado. No se usa como autoridad
tactica, solo para agrupar transiciones en macroestados auditables."""

import chess

from chess_alpha.board_features import game_phase

MACROSTATES = (
    "apertura_tranquila", "medio_juego_estable", "tacticamente_inestable",
    "ataque_forzado", "defensa_obligatoria", "final", "mate", "tablas",
)


def classify_macrostate(board: chess.Board, is_tactically_unstable: bool = False,
                         is_forced_attack: bool = False, is_forced_defense: bool = False) -> str:
    if board.is_checkmate():
        return "mate"
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return "tablas"
    if is_forced_defense:
        return "defensa_obligatoria"
    if is_forced_attack:
        return "ataque_forzado"
    if is_tactically_unstable or board.is_check():
        return "tacticamente_inestable"
    phase = game_phase(board)
    if phase == "endgame":
        return "final"
    if phase == "opening":
        return "apertura_tranquila"
    return "medio_juego_estable"
