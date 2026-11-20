"""Identidad de partida y de posicion para SEARCH TREE DATASET V1.1.

game_fingerprint: formula EXACTA pedida por el usuario -
SHA256(variant + initial_fen + secuencia_uci_completa). Distinto del
`move_hash` ya existente en epoch4_exclusion.py (que no incluye variant ni
initial_fen) - ese se sigue usando como pre-filtro barato por 'site', este
es el fingerprint de cierre que demuestra novedad de partida completa.

position_id: Zobrist hash de python-chess (ya usado en 90/91 para
transposiciones de apertura) + lado al turno - combina board+turno+
enroques+al paso en un unico entero de 64 bits, tal como pide el pedido
("Polyglot/Zobrist hash; FEN canonico de 4 campos; side to move; castling;
en passant")."""

import hashlib

import chess
import chess.polyglot


def game_fingerprint(variant: str, initial_fen: str, moves_uci: list) -> str:
    payload = f"{variant}|{initial_fen}|{' '.join(moves_uci)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def position_id(board: chess.Board) -> str:
    return format(chess.polyglot.zobrist_hash(board), "016x")


def canonical_fen_4field(fen: str) -> str:
    """FEN canonico de los primeros 4 campos (board+turno+enroques+al paso),
    sin halfmove clock ni fullmove number."""
    return " ".join(fen.split(" ")[:4])
