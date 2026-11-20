"""#12 -- piezas sobrecargadas. Definicion CONSERVADORA (config
overloaded_piece.min_critical_defenses, default 2):

  Una pieza propia P es "overloaded" si es la UNICA defensora (entre las
  propias) de al menos `min_critical_defenses` casillas que estan
  SIMULTANEAMENTE atacadas por el rival (casillas "disputadas" que P
  sostiene en solitario). Si P se mueve o se elimina, 2+ cosas caen a la
  vez. NO se marca overloaded a cualquier pieza con dos defensas
  cualquiera -- deben ser casillas realmente disputadas (atacadas por el
  rival), tal como pide el encargo explicitamente.
"""
import chess


def compute_overloaded(board: chess.Board, color: bool, attackers_count_white, attackers_count_black,
                        min_critical_defenses: int) -> dict:
    own_attackers = attackers_count_white if color == chess.WHITE else attackers_count_black
    enemy_attackers = attackers_count_black if color == chess.WHITE else attackers_count_white

    overloaded_count = 0
    for sq in board.pieces(chess.PAWN, color) | board.pieces(chess.KNIGHT, color) | \
              board.pieces(chess.BISHOP, color) | board.pieces(chess.ROOK, color) | board.pieces(chess.QUEEN, color):
        critical = 0
        for target in board.attacks(sq):
            if own_attackers[target] == 1 and enemy_attackers[target] > 0:
                critical += 1
        if critical >= min_critical_defenses:
            overloaded_count += 1

    return {"overloaded_piece_count": overloaded_count}


def compute_overloaded_both(board: chess.Board, attackers_count_white, attackers_count_black, config: dict) -> dict:
    threshold = config["overloaded_piece"]["min_critical_defenses"]
    w = compute_overloaded(board, chess.WHITE, attackers_count_white, attackers_count_black, threshold)
    b = compute_overloaded(board, chess.BLACK, attackers_count_white, attackers_count_black, threshold)
    return {"white_overloaded_piece_count": w["overloaded_piece_count"],
            "black_overloaded_piece_count": b["overloaded_piece_count"]}
