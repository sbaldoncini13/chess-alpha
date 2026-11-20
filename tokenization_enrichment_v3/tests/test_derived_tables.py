"""Tests de las familias basadas en tablas derivadas (engine-derived,
consenso humano) sobre DataFrames SINTETICOS pequenos -- no requieren el
Golden Dataset completo, corren rapido y de forma controlada."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPTS_DIR)

from eval_trajectory import build_eval_white_pov, compute_eval_trajectory  # noqa: E402
from human_consensus import compute_human_consensus, compute_human_engine_agreement  # noqa: E402
from mate_labels import compute_mate_labels  # noqa: E402
from policy_entropy_margins import compute_decision_margins, compute_policy_entropy  # noqa: E402
from terminal_distance import compute_terminal_distance  # noqa: E402
from wdl import compute_wdl, cp_to_winprob  # noqa: E402

MARGIN_THRESHOLDS = {"competitive_max_gap": 20, "forced_min_gap": 150}


# ---------- eval_trajectory (#19) ----------
def test_eval_trajectory_perspective_normalization_and_no_future():
    positions = pd.DataFrame({
        "row_id": [0, 1, 2, 3], "game_id": [0, 0, 0, 0], "side_to_move": [True, False, True, False],
    })
    game_index = pd.DataFrame({"row_id_start": [0], "row_id_end": [3], "n_rows": [4]},
                               index=pd.Index([0], name="game_id"))
    engine_eval = pd.DataFrame({
        "row_id": [0, 1, 2, 3], "best_eval_unified_cp_equivalent": [50, -30, 80, -20],  # side-to-move POV
    })
    wpov = build_eval_white_pov(positions, engine_eval)
    # fila0 (blancas mueven): 50 tal cual. fila1 (negras mueven): -(-30)=30. fila2: 80. fila3: -(-20)=20.
    assert wpov.loc[0] == 50
    assert wpov.loc[1] == 30
    assert wpov.loc[2] == 80
    assert wpov.loc[3] == 20

    traj = compute_eval_trajectory(positions, game_index, wpov).set_index("row_id")
    assert traj.loc[0, "delta_eval_1"] != traj.loc[0, "delta_eval_1"]  # NaN: sin historial (ply0)
    assert traj.loc[1, "delta_eval_1"] == 30 - 50
    assert traj.loc[2, "delta_eval_1"] == 80 - 30
    assert traj.loc[2, "delta_eval_2"] == 80 - 50


def test_eval_trajectory_never_crosses_game_boundary():
    positions = pd.DataFrame({
        "row_id": [0, 1, 2, 3], "game_id": [0, 0, 1, 1], "side_to_move": [True, False, True, False],
    })
    game_index = pd.DataFrame({"row_id_start": [0, 2], "row_id_end": [1, 3], "n_rows": [2, 2]},
                               index=pd.Index([0, 1], name="game_id"))
    engine_eval = pd.DataFrame({"row_id": [0, 1, 2, 3], "best_eval_unified_cp_equivalent": [10, 20, 999, 999]})
    wpov = build_eval_white_pov(positions, engine_eval)
    traj = compute_eval_trajectory(positions, game_index, wpov).set_index("row_id")
    assert pd.isna(traj.loc[2, "delta_eval_1"])  # primer ply de la partida 1 -- no ve la partida 0


# ---------- policy entropy / margins (#20, #21) ----------
def test_policy_entropy_uniform_vs_concentrated():
    uniform = pd.DataFrame({"row_id": [0, 0, 0], "probability": [1 / 3, 1 / 3, 1 / 3]})
    concentrated = pd.DataFrame({"row_id": [1, 1, 1], "probability": [0.98, 0.01, 0.01]})
    both = pd.concat([uniform, concentrated])
    res = compute_policy_entropy(both).set_index("row_id")
    assert res.loc[0, "policy_entropy"] > res.loc[1, "policy_entropy"]
    assert res.loc[0, "normalized_policy_entropy"] == pytest.approx(1.0, abs=1e-6)


def test_decision_margin_categories():
    topk = pd.DataFrame({
        "row_id": [0, 0, 1, 1, 2],
        "rank": [1, 2, 1, 2, 1],
        "eval_type": ["cp", "cp", "cp", "cp", "cp"],
        "eval_unified_cp_equivalent": [10, 5, 300, 50, 40],  # row0: gap=5 (competitive); row1: gap=250 (forced); row2: unica (forced)
    })
    res = compute_decision_margins(topk, MARGIN_THRESHOLDS).set_index("row_id")
    assert res.loc[0, "decision_margin_category"] == "competitive"
    assert res.loc[1, "decision_margin_category"] == "forced"
    assert res.loc[2, "decision_margin_category"] == "forced"  # unica jugada legal


def test_decision_margin_mate_forces_category_regardless_of_cp():
    topk = pd.DataFrame({
        "row_id": [0, 0], "rank": [1, 2], "eval_type": ["mate", "cp"],
        "eval_unified_cp_equivalent": [99999, 5],  # gap numerico pequeno-ish pero top1 es mate
    })
    res = compute_decision_margins(topk, MARGIN_THRESHOLDS).set_index("row_id")
    assert res.loc[0, "decision_margin_category"] == "forced"
    assert bool(res.loc[0, "top1_is_mate"]) is True


# ---------- WDL (#22) ----------
def test_wdl_win_probability_matches_existing_repo_formula():
    assert cp_to_winprob(0) == pytest.approx(0.5)
    assert cp_to_winprob(400) == pytest.approx(10 / 11, abs=1e-6)
    out = compute_wdl(np.array([0.0, 400.0]), np.array(["cp", "cp"]))
    assert out["draw_probability"] is None
    assert out["loss_probability"] is None
    assert "NOT_IMPLEMENTED_REQUIRES_DEFINITION" in out["wdl_status"]


# ---------- mate labels (#24) ----------
def test_mate_labels_direct_from_engine_eval_no_new_search():
    engine_eval = pd.DataFrame({
        "row_id": [0, 1, 2], "best_eval_type": ["mate", "mate", "cp"], "best_eval_mate_in": [3, -2, None],
    })
    res = compute_mate_labels(engine_eval).set_index("row_id")
    assert res.loc[0, "mate_for_side"] == "side_to_move"
    assert res.loc[1, "mate_for_side"] == "opponent"
    assert res.loc[0, "mate_in_n"] == 3
    assert res.loc[1, "mate_in_n"] == 2
    assert res.loc[2, "mate_available"] == False  # noqa: E712


# ---------- terminal distance (#23, FUTURE_DEPENDENT) ----------
def test_terminal_distance_marked_future_dependent():
    positions = pd.DataFrame({"row_id": [0, 1, 2], "game_id": [0, 0, 0], "ply": [0, 1, 2]})
    game_index = pd.DataFrame({"row_id_start": [0], "row_id_end": [2], "n_rows": [3]},
                               index=pd.Index([0], name="game_id"))
    res = compute_terminal_distance(positions, game_index).set_index("row_id")
    assert res.loc[0, "plies_to_terminal"] == 2
    assert res.loc[2, "plies_to_terminal"] == 0
    assert res["leakage_for_input"].all()


# ---------- human consensus (#26, #27) ----------
def test_human_consensus_repeated_position_and_probabilities():
    positions = pd.DataFrame({
        "row_id": [0, 1, 2, 3], "position_id": [111, 111, 111, 222], "human_move_id": [5, 5, 7, 9],
    })
    consensus_df, group_stats = compute_human_consensus(positions, min_occurrence=2)
    row0 = consensus_df.set_index("row_id").loc[0]
    assert row0["occurrence_count"] == 3
    assert row0["unique_human_moves"] == 2
    assert row0["consensus_top_move"] == 5
    assert row0["consensus_strength"] == pytest.approx(2 / 3)
    row3 = consensus_df.set_index("row_id").loc[3]
    assert row3["occurrence_count"] == 1
    assert row3["has_multiple_observations"] == False  # noqa: E712


def test_human_consensus_large_position_id_survives_uint64_range():
    """Regresion directa del bug real: position_id (zobrist) puede superar
    2**53 y perder precision si se pasa por float64 en algun momento (p.ej.
    via DataFrame.iterrows()). Se prueba con un valor grande real."""
    big_pid = 5060803636482931868  # zobrist real observado en produccion
    positions = pd.DataFrame({
        "row_id": [0, 1], "position_id": np.array([big_pid, big_pid], dtype=np.uint64),
        "human_move_id": [1, 2],
    })
    consensus_df, _ = compute_human_consensus(positions, min_occurrence=2)
    assert len(consensus_df) == 2
    assert consensus_df["occurrence_count"].iloc[0] == 2


def test_human_engine_agreement_top1_match_and_mismatch():
    positions = pd.DataFrame({"row_id": [0, 1], "position_id": [1, 2], "human_move_id": [5, 6]})
    consensus_df, group_stats = compute_human_consensus(positions, min_occurrence=1)
    engine_eval = pd.DataFrame({
        "row_id": [0, 1], "best_move_id": [5, 999], "best_eval_unified_cp_equivalent": [10, 10],
    })
    engine_policy = pd.DataFrame({"row_id": [0, 0, 1, 1], "move_id": [5, 3, 999, 6], "probability": [0.6, 0.4, 0.7, 0.3]})
    agree = compute_human_engine_agreement(consensus_df, group_stats, engine_eval, engine_policy, 0.7, 150).set_index("row_id")
    assert bool(agree.loc[0, "human_engine_top1_agreement"]) is True
    assert bool(agree.loc[1, "human_engine_top1_agreement"]) is False
    assert bool(agree.loc[1, "human_disagreement"]) is True
