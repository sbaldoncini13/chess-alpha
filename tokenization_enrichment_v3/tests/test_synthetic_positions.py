"""#35/#43 -- tests unitarios de las 28 familias sobre posiciones SINTETICAS
deterministas (isolated pawn, doubled pawn, passed pawn, pin, skewer,
fork, exposed king, outpost, mate, repetition). No usa el Golden Dataset
-- estas son pruebas de CORRECCION del algoritmo, independientes de los
datos reales."""
import os
import sys

import chess
import numpy as np
import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPTS_DIR)

from attack_maps import compute_attack_density  # noqa: E402
from hanging_pieces import compute_hanging_pieces  # noqa: E402
from king_safety import compute_king_safety_both  # noqa: E402
from mobility import compute_mobility  # noqa: E402
from outposts import compute_outposts_both  # noqa: E402
from overloaded import compute_overloaded_both  # noqa: E402
from pawn_structure import compute_pawn_structure_both  # noqa: E402
from pins_skewers import compute_pins_skewers_both  # noqa: E402
from see import static_exchange_eval  # noqa: E402
from tactical_motifs import compute_tactical_motifs_both  # noqa: E402
from threats import compute_threats  # noqa: E402

CONFIG = {
    "king_safety": {
        "king_ring_radius": 1, "queen_proximity_max_distance": 3, "rook_proximity_max_distance": 3,
        "exposure_score_weights": {"w_king_ring_attacked": 0.30, "w_pawn_shield_missing": 0.20,
                                    "w_open_files": 0.15, "w_semi_open_files": 0.10,
                                    "w_enemy_attackers_near": 0.15, "w_queen_near": 0.05, "w_rook_near": 0.05},
    },
    "overloaded_piece": {"min_critical_defenses": 2},
    "tactical_thresholds": {"immediate_material_threat_min_value": 3},
    "bad_bishop_min_same_color_pawns": 4,
}


def _ad(board):
    return compute_attack_density(board)


# ---------- isolated pawn ----------
def test_isolated_pawn():
    board = chess.Board("4k3/8/8/8/3P4/8/1P1P4/4K3 w - - 0 1")  # b2, d2, d4 -- d-file peones aislados (sin c/e)
    ps = compute_pawn_structure_both(board)
    assert ps["white_isolated_pawns"] >= 2  # d2 y d4 (mismo file, sin apoyo lateral en c/e)


# ---------- doubled pawn ----------
def test_doubled_pawn():
    board = chess.Board("4k3/8/8/8/3P4/8/3P4/4K3 w - - 0 1")
    ps = compute_pawn_structure_both(board)
    assert ps["white_doubled_pawn_files"] == 1


# ---------- passed pawn ----------
def test_passed_pawn():
    board = chess.Board("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1")  # sin peones negros
    ps = compute_pawn_structure_both(board)
    assert ps["white_passed_pawns"] == 1


def test_non_passed_pawn_blocked_by_enemy_file():
    board = chess.Board("4k3/3p4/8/3P4/8/8/8/4K3 w - - 0 1")  # peon negro por delante en el mismo file
    ps = compute_pawn_structure_both(board)
    assert ps["white_passed_pawns"] == 0


# ---------- pin (absoluto) ----------
def test_absolute_pin():
    # alfil negro a5 - caballo blanco d2 - rey blanco e1, todos en la
    # misma diagonal (a5-b4-c3-d2-e1): el caballo esta clavado
    board = chess.Board("4k3/8/8/b7/8/8/3N4/4K3 w - - 0 1")
    pins = compute_pins_skewers_both(board)
    assert pins["white_absolute_pins"] == 1


# ---------- skewer ----------
def test_skewer_geometry():
    # torre blanca e1 ataca por la columna e: rey negro e5 en frente, torre
    # negra e8 detras -- skewer clasico (mover el rey expone la torre)
    board = chess.Board("4r3/8/8/4k3/8/8/8/4R3 w - - 0 1")
    pins = compute_pins_skewers_both(board)
    assert pins["black_skewers_against"] >= 1


# ---------- fork ----------
def test_knight_fork():
    # caballo blanco en d6 ataca simultaneamente rey negro e8 y torre negra b7... usemos un caso claro:
    board = chess.Board("r3k3/8/3N4/8/8/8/8/4K3 w - - 0 1")  # Nd6 ataca e8 (rey) y b7(vacio)? verificar con torre en b7
    board = chess.Board("1r2k3/8/3N4/8/8/8/8/4K3 w - - 0 1")  # torre en b8... probemos Nc7 clasico
    board = chess.Board("r3k3/2N5/8/8/8/8/8/4K3 w - - 0 1")  # caballo c7 ataca a8(vacio) y e8(rey) -- fork rey+casilla
    tm = compute_tactical_motifs_both(board, compute_pins_skewers_both(board), compute_overloaded_both(board, _ad(board)["attackers_count_white"], _ad(board)["attackers_count_black"], CONFIG))
    assert tm["white_fork_count"] >= 1


# ---------- exposed king ----------
def test_exposed_king_high_score_vs_sheltered():
    exposed = chess.Board("4k3/8/8/8/4K3/8/8/8 w - - 0 1")  # rey en el centro, sin peones, sin refugio
    sheltered = chess.Board("4k3/8/8/8/8/8/PPPPPPPP/R3K2R w KQ - 0 1")  # rey blanco enrocable con escudo completo
    ad_e, ad_s = _ad(exposed), _ad(sheltered)
    ks_e = compute_king_safety_both(exposed, ad_e["attackers_count_white"], ad_e["attackers_count_black"], CONFIG["king_safety"])
    ks_s = compute_king_safety_both(sheltered, ad_s["attackers_count_white"], ad_s["attackers_count_black"], CONFIG["king_safety"])
    assert ks_e["white_king_exposure_score"] > ks_s["white_king_exposure_score"]


# ---------- outpost ----------
def test_knight_outpost():
    # caballo blanco en d6, defendido por peon e5, sin peon negro en c/e que
    # pueda desalojarlo (peones negros solo en f7,g7,h7)
    board = chess.Board("4k3/5ppp/3N4/4P3/8/8/8/4K3 w - - 0 1")
    op = compute_outposts_both(board)
    assert op["white_occupied_knight_outposts"] >= 1


# ---------- mate ----------
def test_checkmate_detected():
    mated = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        mated.push_san(san)
    assert mated.is_checkmate()

    pre_mate = chess.Board()
    for san in ["f3", "e5", "g4"]:
        pre_mate.push_san(san)
    ad = _ad(pre_mate)
    th = compute_threats(pre_mate, ad["attackers_count_white"], ad["attackers_count_black"], CONFIG)
    assert th["mate_in_1_available"] is True  # Qh4# esta disponible desde la posicion previa


# ---------- repetition (via sequences.compute_sequences, misma funcion que produce V3) ----------
def test_repetition_detection_in_sequence():
    import chess.polyglot
    import pandas as pd

    from sequences import compute_sequences

    board = chess.Board()
    sans = ["Nf3", "Nf6", "Ng1", "Ng8"] * 2  # vuelve a la posicion inicial 2 veces mas (plies 4 y 8)
    rows = [{"row_id": 0, "game_id": 0, "ply": 0, "position_id": chess.polyglot.zobrist_hash(board),
             "move_is_capture": False, "move_is_check": False}]
    for i, san in enumerate(sans, start=1):
        board.push_san(san)
        rows.append({"row_id": i, "game_id": 0, "ply": i, "position_id": chess.polyglot.zobrist_hash(board),
                     "move_is_capture": False, "move_is_check": False})
    positions = pd.DataFrame(rows)
    game_index = pd.DataFrame({"row_id_start": [0], "row_id_end": [8], "n_rows": [9]}, index=pd.Index([0], name="game_id"))

    res = compute_sequences(positions, game_index, max_length=8)
    assert positions.loc[0, "position_id"] == positions.loc[4, "position_id"] == positions.loc[8, "position_id"]
    assert res["repetition_in_sequence"][8] == True  # noqa: E712 -- fila 8 ve las filas 0 y 4 en su ventana
    assert res["repetition_in_sequence"][1] == False  # noqa: E712 -- sin historial repetido todavia


def test_sequences_no_future_leakage_and_no_cross_game():
    import chess.polyglot
    import pandas as pd

    from sequences import compute_sequences

    # dos partidas distintas: la partida 1 NUNCA debe aparecer en el
    # historial de la partida 2, ni siquiera en la primera fila de esta.
    rows = []
    rid = 0
    for game_id in (0, 1):
        board = chess.Board()
        for ply, san in enumerate(["e4", "e5", "Nf3"]):
            rows.append({"row_id": rid, "game_id": game_id, "ply": ply,
                         "position_id": chess.polyglot.zobrist_hash(board),
                         "move_is_capture": False, "move_is_check": False})
            board.push_san(san)
            rid += 1
    positions = pd.DataFrame(rows)
    game_index = pd.DataFrame(
        {"row_id_start": [0, 3], "row_id_end": [2, 5], "n_rows": [3, 3]}, index=pd.Index([0, 1], name="game_id")
    )
    res = compute_sequences(positions, game_index, max_length=8)
    # primera fila de la partida 1 (row_id=3) no debe tener historial disponible
    assert res["sequence_available_length"][3] == 0
    assert res["truncated_at_game_start"][3] == True  # noqa: E712


# ---------- SEE sanity (ver ademas benchmark manual ya validado) ----------
def test_see_winning_capture_free_piece():
    board = chess.Board("4k3/8/8/4q3/8/8/4R3/4K3 w - - 0 1")
    move = board.parse_san("Rxe5")
    assert static_exchange_eval(board, move) == 9


def test_see_equal_queen_trade():
    board = chess.Board("4k3/4q3/8/8/8/8/4Q3/4K3 w - - 0 1")
    move = board.parse_san("Qxe7")
    assert static_exchange_eval(board, move) == 0


def test_see_losing_capture():
    # caballo c3 captura peon d5, defendido por peon negro e6 -> pierde
    # caballo(3) por peon(1), cambio perdedor
    board = chess.Board("4k3/8/4p3/3p4/8/2N5/8/4K3 w - - 0 1")
    move = board.parse_san("Nxd5")
    result = static_exchange_eval(board, move)
    assert result < 0


# ---------- bitboard dtype consistency (regresion real: full run, shard_00001) ----------
def test_attack_bitboard_dtype_consistent_regardless_of_bit63():
    """Bug real encontrado en el full run: squares_attacked_*_bitboard se
    devolvia como int de Python. pandas infiere el dtype de columna por
    lote (uint64 si ALGUNA fila del lote tiene el bit 63 / casilla h8
    prendido, int64 si ninguna) -- en un lote chico (el ultimo, parcial,
    de un shard) es facil que ninguna fila tenga el bit 63 prendido,
    produciendo un lote int64 mientras el resto del shard es uint64.
    pa.concat_tables revienta con ArrowInvalid al consolidar (shard_00001,
    batch 51/51, 119 filas). Fix: dtype np.uint64 explicito en
    attack_maps.py, que debe ser consistente sin importar el valor."""
    no_h8_attacked = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")  # nada ataca h8
    h8_attacked = chess.Board("k7/8/8/8/8/8/8/6KR w - - 0 1")  # torre h1 ataca toda la columna h, incl. h8

    ad_no = compute_attack_density(no_h8_attacked)
    ad_yes = compute_attack_density(h8_attacked)

    assert type(ad_no["squares_attacked_white_bitboard"]) == type(ad_yes["squares_attacked_white_bitboard"])
    assert isinstance(ad_no["squares_attacked_white_bitboard"], np.uint64)
    assert isinstance(ad_yes["squares_attacked_white_bitboard"], np.uint64)
    # bit 63 (h8, indice mas alto) efectivamente prendido en un caso y no en el otro
    assert (int(ad_yes["squares_attacked_white_bitboard"]) & (1 << 63)) != 0
    assert (int(ad_no["squares_attacked_white_bitboard"]) & (1 << 63)) == 0
