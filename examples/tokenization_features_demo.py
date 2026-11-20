"""Tokenization + position-feature demo on a short synthetic game.

1. Encodes every move as a 16-bit `action_id`
   ((promotion_code << 12) | (from_square << 6) | to_square) and checks the
   round trip against python-chess, including castling, en passant and an
   under-promotion with capture.
2. Computes the board-derived V3 feature families for the final position.

Requires `python-chess` and `numpy`. No datasets, checkpoints or Stockfish.

    python examples/tokenization_features_demo.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenization_v1" / "scripts"))
sys.path.insert(0, str(ROOT / "tokenization_enrichment_v3" / "scripts"))

import chess  # noqa: E402

from tokenizer import action_id_to_move, move_to_action_id  # noqa: E402
from position_features import compute_all_position_features  # noqa: E402

# Synthetic move sequence (not taken from any game database): white
# castles short, black castles long, a pawn captures en passant and a
# pawn captures with under-promotion to a knight.
SYNTHETIC_GAME_UCI = [
    "e2e4", "d7d5", "e4e5", "f7f5", "e5f6",   # en passant capture
    "c8e6", "g1f3", "b8c6", "f1e2", "d8d6",
    "e1g1", "e8c8",                            # both castlings
    "f6g7", "h7h6", "g7h8n",                   # capture + under-promotion
]


def main() -> None:
    board = chess.Board()
    action_ids = []
    for uci in SYNTHETIC_GAME_UCI:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"illegal move in synthetic game: {uci}"
        aid = move_to_action_id(move)
        assert action_id_to_move(aid) == move, f"round-trip failed for {uci}"
        tags = []
        if board.is_en_passant(move):
            tags.append("en passant")
        if board.is_castling(move):
            tags.append("castling")
        if move.promotion:
            tags.append(f"promotion={chess.piece_name(move.promotion)}")
        action_ids.append(aid)
        print(f"{uci:6} -> action_id {aid:5d}  {', '.join(tags)}")
        board.push(move)

    print(f"\n{len(action_ids)} moves encoded, all round trips OK (uint16 range: max {max(action_ids)})")
    print(f"final FEN: {board.fen()}")

    config = json.loads((ROOT / "tokenization_enrichment_v3" / "config" / "enrichment_v3_config.json").read_text(encoding="utf-8"))
    features = compute_all_position_features(board.fen(), config, move_to_action_id=move_to_action_id)
    scalars = {k: v for k, v in features.items() if isinstance(v, (int, float, bool))}
    print(f"\n{len(features)} feature columns computed ({len(scalars)} scalar). Sample:")
    for key in sorted(scalars)[:20]:
        print(f"  {key:45} {scalars[key]}")


if __name__ == "__main__":
    main()
