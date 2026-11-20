"""Search Tree V1 demo: PVS + transposition table + quiescence, exact mate
proof (DFPN/PNS) and forced-defense analysis on small synthetic positions.

Only requires `python-chess`. No datasets, checkpoints or Stockfish.

    python examples/search_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_v1 import (  # noqa: E402
    DFPNLimits,
    OrderingState,
    Position,
    QuiescenceLimits,
    SearchLimits,
    TranspositionTable,
    analyze_forced_defense,
    find_forced_mate,
    iterative_deepening_search,
    verify_mate_proof,
)

# Scholar's-mate pattern: the normal search must report a mate score.
MIDDLEGAME_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
# K+R vs K: forced mate in 3 plies (fixture from tests/search_v1/test_dfpn.py).
MATE_IN_THREE_PLIES_FEN = "6k1/8/6K1/8/8/7R/8/8 w - - 0 1"
# Exactly one black move avoids being mated within a 6-ply horizon
# (fixture from tests/search_v1/test_forced_defense.py). With no depth
# bound every move eventually loses (K+Q+P vs K is won), which is why the
# bound is part of the question.
UNIQUE_DEFENSE_FEN = "3k4/8/3P4/2K5/8/8/8/6Q1 b - - 0 1"


def run_search(fen: str, depth: int) -> None:
    position = Position(fen)
    result = iterative_deepening_search(
        position,
        SearchLimits(max_depth=depth),
        tt=TranspositionTable(),
        ordering_state=OrderingState(),
        quiescence_limits=QuiescenceLimits(),
        use_pvs=True,
    )
    print(f"FEN            : {fen}")
    print(f"best move      : {result.best_move}")
    print(f"score          : {result.score} ({result.score_type.name})")
    print(f"PV             : {' '.join(result.principal_variation)}")
    print(f"depth / nodes  : {result.completed_depth} / {result.nodes}")
    print(f"legal / intact : {result.legal} / {position.fen == fen}")


def run_mate_proof(fen: str) -> None:
    position = Position(fen)
    result = find_forced_mate(position, limits=DFPNLimits(max_nodes=50_000))
    problems = verify_mate_proof(Position(fen), result)
    print(f"FEN            : {fen}")
    print(f"proof status   : {result.proof_status.name}")
    print(f"mate in plies  : {result.mate_distance_plies}")
    print(f"PV             : {' '.join(result.principal_variation)}")
    print(f"independent verifier problems: {problems or 'none'}")


def run_forced_defense(fen: str) -> None:
    result = analyze_forced_defense(Position(fen), limits=DFPNLimits(max_nodes=200_000, max_depth_plies=6))
    print(f"FEN            : {fen}")
    print(f"legal moves    : {result.legal_moves_count}")
    print(f"preserving     : {result.preserving_moves}")
    print(f"losing         : {result.losing_moves}")
    print(f"unknown        : {result.unknown_moves}")
    print(f"unique defense : {result.unique_forced_defense}")


if __name__ == "__main__":
    print("== 1. PVS + TT + quiescence (depth 3) ==")
    run_search(MIDDLEGAME_FEN, depth=3)
    print("\n== 2. Exact forced-mate proof (DFPN/PNS) ==")
    run_mate_proof(MATE_IN_THREE_PLIES_FEN)
    print("\n== 3. Forced-defense analysis (6-ply proof horizon) ==")
    run_forced_defense(UNIQUE_DEFENSE_FEN)
