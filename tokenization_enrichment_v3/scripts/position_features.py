"""Orquesta TODAS las familias que dependen de un chess.Board reconstruido
desde el FEN (attack_maps, hanging_pieces, mobility, king_safety,
pawn_structure, piece_activity, outposts, pins_skewers, overloaded, see,
threats, tactical_motifs, space, center_structure). Calcula
attackers_count UNA sola vez (attack_maps.compute_attack_density) y lo
inyecta a todos los modulos que lo necesitan -- evita recalcular
board.attackers() por familia.
"""
import chess

from attack_maps import compute_attack_density
from center_structure import compute_center_structure
from hanging_pieces import compute_hanging_pieces
from king_safety import compute_king_safety_both
from mobility import compute_mobility
from outposts import compute_outposts_both
from overloaded import compute_overloaded_both
from pawn_structure import compute_pawn_structure_both
from piece_activity import compute_piece_activity_both
from pins_skewers import compute_pins_skewers_both
from see import compute_see_summary
from space import compute_space
from tactical_motifs import compute_tactical_motifs_both
from threats import compute_threats


def compute_all_position_features(fen: str, config: dict, move_to_action_id=None) -> dict:
    board = chess.Board(fen)
    out = {}

    ad = compute_attack_density(board)
    aw, ab = ad["attackers_count_white"], ad["attackers_count_black"]
    out["squares_attacked_white_bitboard"] = ad["squares_attacked_white_bitboard"]
    out["squares_attacked_black_bitboard"] = ad["squares_attacked_black_bitboard"]
    out["attacked_square_count_white"] = ad["attacked_square_count_white"]
    out["attacked_square_count_black"] = ad["attacked_square_count_black"]
    out["contested_squares"] = ad["contested_squares"]

    hp = compute_hanging_pieces(board, aw, ab)
    for k, v in hp.items():
        out[k] = v

    mob = compute_mobility(board)
    for k, v in mob.items():
        out[k] = v

    ks = compute_king_safety_both(board, aw, ab, config["king_safety"])
    for k, v in ks.items():
        out[k] = v

    ps = compute_pawn_structure_both(board)
    for k, v in ps.items():
        out[k] = v

    pa = compute_piece_activity_both(board, aw, ab, config)
    for k, v in pa.items():
        out[k] = v

    op = compute_outposts_both(board)
    for k, v in op.items():
        out[k] = v

    pins = compute_pins_skewers_both(board)
    for k, v in pins.items():
        out[k] = v

    ov = compute_overloaded_both(board, aw, ab, config)
    for k, v in ov.items():
        out[k] = v

    see_summary = compute_see_summary(board)
    for k, v in see_summary.items():
        out[f"see_{k}"] = v

    th = compute_threats(board, aw, ab, config)
    for k, v in th.items():
        out[k] = v

    tm = compute_tactical_motifs_both(board, pins, ov)
    for k, v in tm.items():
        out[k] = v

    sp = compute_space(aw, ab)
    for k, v in sp.items():
        out[k] = v

    cs = compute_center_structure(board)
    for k, v in cs.items():
        out[k] = v

    return out
