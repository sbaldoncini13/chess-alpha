"""Tests de round-trip FEN sobre una muestra reproducible de SEARCH TREE
DATASET V1.1 (read-only, sin Stockfish) + los 2 historicos de regresion de
Fase 7 (legalidad unicamente en esta fase - la correccion tactica de esas
2 posiciones depende de busqueda, todavia no implementada; se registran
como pendientes de Fase VI, no se finge que pueden aprobarse ahora).

Seed fija: 20260805 (misma que la validacion ligera de dataset de Fase I).

Correr: python -m pytest tests/search_v1/test_roundtrip.py -v
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from chess_alpha.config import SEARCH_TREE_V1_FINAL_DIR
from search_v1 import InvalidFENError, Move, Position, generate_legal_moves

SEED = 20260805
REGRESSION_PATH = Path(__file__).resolve().parents[2] / "tests" / "regression" / "forced_defense_critical.json"


def _load_sample(n=200):
    pos = pd.read_parquet(SEARCH_TREE_V1_FINAL_DIR / "positions.parquet", columns=["fen", "legal_move_count", "best_move_uci"])
    return pos.sample(n=min(n, len(pos)), random_state=SEED)


class TestFENRoundtripAgainstDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = _load_sample(200)

    def test_all_sample_fens_load_without_error(self):
        for fen in self.sample["fen"]:
            with self.subTest(fen=fen):
                try:
                    Position(fen)
                except InvalidFENError as exc:
                    self.fail(f"FEN del dataset (ya verificado en Fase 19) fallo al cargar: {exc}")

    def test_fen_roundtrip_exact(self):
        for fen in self.sample["fen"]:
            with self.subTest(fen=fen):
                p = Position(fen)
                self.assertEqual(p.fen, fen)

    def test_legal_move_count_matches_dataset(self):
        for fen, expected in zip(self.sample["fen"], self.sample["legal_move_count"]):
            with self.subTest(fen=fen):
                p = Position(fen)
                self.assertEqual(len(generate_legal_moves(p)), int(expected))

    def test_best_move_uci_is_legal(self):
        for fen, best_move in zip(self.sample["fen"], self.sample["best_move_uci"]):
            if best_move is None:
                continue
            with self.subTest(fen=fen, best_move=best_move):
                p = Position(fen)
                self.assertTrue(p.is_legal(Move.from_uci(best_move)))


class TestPhase7HistoricalRegressions(unittest.TestCase):
    """Los 2 historicos de Fase 7 (defensa obligatoria ante mate ignorada
    por el prior de Policy, RECHAZADO). En Fase I solo se verifica que la
    posicion carga y que la jugada 'solution' es legal - NO que el motor
    la elija (eso requiere busqueda, Fase VI en adelante)."""

    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))["cases"]

    def test_historicals_load_and_solution_is_legal(self):
        # NOTA: el position_id almacenado en forced_defense_critical.json
        # NO coincide con chess.polyglot.zobrist_hash recalculado aca
        # (verificado independientemente contra identity.py::position_id,
        # que si coincide con nuestro calculo - el desvio esta en el JSON
        # historico, generado por un esquema anterior/distinto, no en
        # search_v1). Documentado como hallazgo en el reporte de Fase I,
        # no bloquea el gate: no se exige que el id historico coincida,
        # solo que el FEN cargue y la solucion sea legal.
        self.assertEqual(len(self.cases), 2)
        for case in self.cases:
            with self.subTest(position_id=case["position_id"]):
                p = Position(case["fen"])
                solution = Move.from_uci(case["solution"])
                self.assertTrue(p.is_legal(solution), "la jugada solucion documentada debe ser legal")

    def test_tactical_correctness_pending_search(self):
        # Documentado explicitamente como PENDIENTE, no aprobado: verificar
        # que el motor efectivamente ELIGE la defensa correcta requiere
        # busqueda (Negamax/Alpha-Beta/PVS), no implementada en Fase I.
        self.skipTest("Correccion tactica de los historicos de Fase 7 requiere busqueda - pendiente de Fase VI")


if __name__ == "__main__":
    unittest.main()
