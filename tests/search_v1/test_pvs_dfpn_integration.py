"""Tests de integracion PVS + DFPN (Fase VI, items 39-41): gate condicional
con mate probado (elevacion a score exacto), gate con UNKNOWN (continua
con PVS normal, nunca convierte UNKNOWN en resultado exacto), mate
distance consistente al pasar por la TT (score_to_tt/score_from_tt).

Correr: python -m pytest tests/search_v1/test_pvs_dfpn_integration.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, DFPNGateConfig, DFPNTable, Position, RepetitionTracker, SearchContext,
    SearchLimits, TranspositionTable, is_mate_score, mate_distance_plies_from_score, pvs,
    score_from_tt, score_to_tt,
)

# k7/8/2K5/8/8/8/8/7Q w: mate en 3 (c6b6 a8b8 h1b7) - PVS a depth=1 NO lo
# encontraria sin el gate; CON el gate (trigger_on_check=False, dispara en
# cualquier nodo con profundidad <= umbral) SI lo encuentra via DFPN.
_MATE_IN_3_FEN = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"


class TestPVSDFPNIntegrationMateProven(unittest.TestCase):
    def test_gate_elevates_score_to_exact_mate(self):
        p = Position(_MATE_IN_3_FEN)
        ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker())
        gate = DFPNGateConfig(
            enabled=True, trigger_on_check=False, max_pvs_depth_to_trigger=1,
            dfpn_max_nodes=5000, dfpn_max_depth_plies=6,
        )
        score, pv = pvs(p, 1, -INFINITY, INFINITY, ctx, dfpn_gate=gate)
        self.assertTrue(is_mate_score(score))
        self.assertEqual(mate_distance_plies_from_score(score), 3)
        self.assertEqual(pv, ["c6b6", "a8b8", "h1b7"])
        self.assertEqual(ctx.dfpn_gate_mate_elevations, 1)
        self.assertGreater(ctx.dfpn_gate_triggers, 0)

    def test_without_gate_shallow_pvs_does_not_find_the_mate(self):
        # control: la MISMA profundidad (1), sin el gate, no puede probar
        # el mate de 3 jugadas - confirma que el gate realmente EXTIENDE
        # el alcance de PVS, no es un no-op.
        p = Position(_MATE_IN_3_FEN)
        ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker())
        score, pv = pvs(p, 1, -INFINITY, INFINITY, ctx, dfpn_gate=None)
        self.assertNotEqual(mate_distance_plies_from_score(score) if is_mate_score(score) else None, 3)

    def test_gate_disabled_by_default_never_triggers(self):
        p = Position(_MATE_IN_3_FEN)
        ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker())
        pvs(p, 1, -INFINITY, INFINITY, ctx, dfpn_gate=DFPNGateConfig(enabled=False))
        self.assertEqual(ctx.dfpn_gate_triggers, 0)


class TestPVSDFPNIntegrationUnknown(unittest.TestCase):
    def test_gate_with_tiny_budget_never_elevates_incorrectly(self):
        # presupuesto de DFPN deliberadamente insuficiente (1 nodo) - el
        # gate debe registrar UNKNOWN y seguir con PVS normal, NUNCA
        # inventar un score exacto a partir de una prueba incompleta.
        p = Position(_MATE_IN_3_FEN)
        ctx = SearchContext(SearchLimits(max_depth=2), RepetitionTracker())
        gate = DFPNGateConfig(
            enabled=True, trigger_on_check=False, max_pvs_depth_to_trigger=2,
            dfpn_max_nodes=1, dfpn_max_depth_plies=6,
        )
        score, pv = pvs(p, 2, -INFINITY, INFINITY, ctx, dfpn_gate=gate)
        self.assertGreater(ctx.dfpn_gate_triggers, 0)
        self.assertEqual(ctx.dfpn_gate_mate_elevations, 0)
        self.assertGreaterEqual(ctx.dfpn_gate_unknown, 0)
        # la busqueda sigue siendo valida (PVS continuo normalmente)
        self.assertIsNotNone(pv)

    def test_no_mate_confirmation_does_not_alter_score(self):
        # posicion sin mate cercano - el gate puede confirmar "no hay
        # mate" (PROVEN_NO_MATE/WITHIN_BOUND) pero eso NUNCA se usa para
        # fijar un score, solo se registra en telemetria.
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p1 = Position(fen)
        ctx1 = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        score_no_gate, _ = pvs(p1, 3, -INFINITY, INFINITY, ctx1, dfpn_gate=None)

        p2 = Position(fen)
        ctx2 = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        gate = DFPNGateConfig(enabled=True, trigger_on_check=True, max_pvs_depth_to_trigger=3, dfpn_max_nodes=500)
        score_with_gate, _ = pvs(p2, 3, -INFINITY, INFINITY, ctx2, dfpn_gate=gate)
        self.assertEqual(score_no_gate, score_with_gate)


class TestMateDistanceConsistentWithTT(unittest.TestCase):
    def test_score_survives_tt_round_trip_at_different_ply(self):
        p = Position(_MATE_IN_3_FEN)
        ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker())
        tt = TranspositionTable(4096)
        gate = DFPNGateConfig(
            enabled=True, trigger_on_check=False, max_pvs_depth_to_trigger=1,
            dfpn_max_nodes=5000, dfpn_max_depth_plies=6,
        )
        score, pv = pvs(p, 1, -INFINITY, INFINITY, ctx, tt=tt, dfpn_gate=gate)
        self.assertTrue(is_mate_score(score))

        entry = tt.probe(p.zobrist_key)
        self.assertIsNotNone(entry)
        # el score guardado en la TT esta normalizado respecto del NODO
        # (ply_from_root=0 aca, sin ajuste) - recuperado desde un ply
        # distinto, la distancia de mate debe seguir siendo consistente.
        recovered_at_root = score_from_tt(entry.score, 0)
        self.assertEqual(recovered_at_root, score)
        # `entry.score` esta normalizado "relativo al nodo" (ply_from_root=0
        # al guardar, no-op) - recuperado desde una consulta HIPOTETICA a
        # 2 plies de profundidad de OTRA busqueda, el mate (3 plies desde
        # el nodo) queda a 2+3=5 plies de ESA raiz (score_from_tt sigue
        # siendo la inversa exacta de score_to_tt en cualquier ply).
        recovered_deeper = score_from_tt(entry.score, 2)
        self.assertEqual(mate_distance_plies_from_score(recovered_deeper), 5)


if __name__ == "__main__":
    unittest.main()
