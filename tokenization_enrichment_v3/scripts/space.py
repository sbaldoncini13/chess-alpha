"""#16 -- espacio. Metrica reproducible basada en attackers_count ya
calculado (attack_maps.py), sin recomputar ataques.

  space(color) := cantidad de casillas en el MEDIO CAMPO RIVAL (ranks 4-7
                  indice 0-based para blancas atacando; ranks 0-3 para
                  negras) que estan controladas (attackers_count>0) por
                  `color`.
  central_space(color) := lo mismo pero restringido al centro extendido
                  (files 2-5, ranks 2-5).
"""
import chess

CENTER_FILES = {2, 3, 4, 5}
CENTER_RANKS = {2, 3, 4, 5}


def compute_space(attackers_count_white, attackers_count_black) -> dict:
    white_half_ranks = {4, 5, 6, 7}
    black_half_ranks = {0, 1, 2, 3}

    space_white = sum(1 for sq in range(64) if (sq // 8) in white_half_ranks and attackers_count_white[sq] > 0)
    space_black = sum(1 for sq in range(64) if (sq // 8) in black_half_ranks and attackers_count_black[sq] > 0)

    central_white = sum(
        1 for sq in range(64)
        if (sq // 8) in CENTER_RANKS and (sq % 8) in CENTER_FILES and attackers_count_white[sq] > 0
    )
    central_black = sum(
        1 for sq in range(64)
        if (sq // 8) in CENTER_RANKS and (sq % 8) in CENTER_FILES and attackers_count_black[sq] > 0
    )

    return {
        "space_white": space_white, "space_black": space_black, "space_diff": space_white - space_black,
        "central_space_white": central_white, "central_space_black": central_black,
        "central_space_diff": central_white - central_black,
    }
