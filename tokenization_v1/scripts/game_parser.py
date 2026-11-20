"""Parseo y validacion de una partida PGN individual, sin depender de PlyCount.

Disenado para correr dentro de un worker de ProcessPoolExecutor: recibe offsets
de bytes, abre el PGN el mismo (cada worker abre su propio file handle), parsea
headers + movetext a mano, valida legalidad jugada a jugada con python-chess, y
devuelve un resultado serializable (dict) -- nunca deja escapar una excepcion no
controlada, para que una partida fallida no tumbe el pool.
"""
import hashlib
import re
import traceback

import chess

from tokenizer import move_to_action_id

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
COMMENT_RE = re.compile(r"\{[^}]*\}")
NAG_RE = re.compile(r"\$\d+")
MOVE_NUMBER_RE = re.compile(r"\d+\.(\.\.)?")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
SUPPORTED_VARIANTS = {"standard", "normal", ""}

VALID = "valida"
INVALID_EMPTY = "partida_vacia"
INVALID_RESULT_MISSING = "resultado_faltante"
INVALID_ILLEGAL_MOVE = "movimiento_ilegal"
INVALID_PARSE_ERROR = "error_de_parseo"
INVALID_VARIANT = "variante_no_soportada"
INVALID_CORRUPT = "corrupta"
FAILED_EXCEPTION = "fallida_por_excepcion"
# "duplicada" se asigna en el proceso principal, no aqui (requiere estado global).


def _strip_variations(text: str) -> str:
    """Elimina variaciones entre parentesis, tolerando anidamiento simple."""
    out = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            if depth > 0:
                depth -= 1
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def parse_headers(header_block: str) -> dict:
    headers = {}
    for line in header_block.splitlines():
        m = HEADER_RE.match(line.strip())
        if m:
            headers[m.group(1)] = m.group(2)
    return headers


def split_header_and_movetext(game_text: str):
    lines = game_text.splitlines()
    header_lines = []
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            header_lines.append(line)
            body_start = i + 1
        elif stripped == "":
            continue
        else:
            break
    movetext = "\n".join(lines[body_start:])
    return "\n".join(header_lines), movetext


def tokenize_movetext(movetext: str):
    text = COMMENT_RE.sub(" ", movetext)
    text = _strip_variations(text)
    text = NAG_RE.sub(" ", text)
    text = MOVE_NUMBER_RE.sub(" ", text)
    raw_tokens = text.split()
    tokens = [t for t in raw_tokens if t not in RESULT_TOKENS]
    trailing_result = None
    for t in raw_tokens:
        if t in RESULT_TOKENS:
            trailing_result = t
    return tokens, trailing_result


def compute_identity_hash(variant: str, initial_fen: str, uci_sequence: list) -> str:
    payload = f"{variant}|{initial_fen}|{' '.join(uci_sequence)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_and_validate_game(pgn_path: str, byte_start: int, byte_end: int, draw_rank: int,
                             game_index: int) -> dict:
    """Punto de entrada usado por los workers. Nunca lanza excepcion."""
    result = {
        "draw_rank": draw_rank,
        "game_index": game_index,
        "byte_start": byte_start,
        "byte_end": byte_end,
        "status": FAILED_EXCEPTION,
        "reason": None,
        "error_ply": None,
        "error_move": None,
        "error_fen": None,
        "ply_count": 0,
        "action_ids": [],
        "identity_hash": None,
        "headers": {},
        "exception_text": None,
    }
    try:
        with open(pgn_path, "rb") as f:
            f.seek(byte_start)
            raw = f.read(byte_end - byte_start)
        game_text = raw.decode("utf-8-sig", errors="replace")

        header_block, movetext = split_header_and_movetext(game_text)
        headers = parse_headers(header_block)
        result["headers"] = headers

        if not headers:
            result["status"] = INVALID_CORRUPT
            result["reason"] = "sin_cabeceras"
            return result

        variant = headers.get("Variant", "").strip().lower()
        if variant not in SUPPORTED_VARIANTS:
            result["status"] = INVALID_VARIANT
            result["reason"] = f"variante_no_soportada:{variant}"
            return result
        variant_norm = "standard"

        setup = headers.get("SetUp", "0").strip()
        fen_header = headers.get("FEN", "").strip()
        if setup == "1" and fen_header:
            try:
                board = chess.Board(fen_header)
            except ValueError as exc:
                result["status"] = INVALID_CORRUPT
                result["reason"] = f"fen_invalido:{exc}"
                return result
        else:
            board = chess.Board()
        initial_fen_normalized = board.fen()

        tokens, trailing_result = tokenize_movetext(movetext)

        if not tokens:
            result["status"] = INVALID_EMPTY
            result["reason"] = "sin_movimientos"
            return result

        action_ids = []
        uci_sequence = []
        ply = 0
        for tok in tokens:
            fen_before = board.fen()
            try:
                move = board.parse_san(tok)
            except chess.IllegalMoveError as exc:
                result["status"] = INVALID_ILLEGAL_MOVE
                result["reason"] = str(exc)
                result["error_ply"] = ply
                result["error_move"] = tok
                result["error_fen"] = fen_before
                return result
            except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError) as exc:
                result["status"] = INVALID_PARSE_ERROR
                result["reason"] = str(exc)
                result["error_ply"] = ply
                result["error_move"] = tok
                result["error_fen"] = fen_before
                return result

            action_ids.append(move_to_action_id(move))
            uci_sequence.append(move.uci())
            board.push(move)
            ply += 1

        result_header = headers.get("Result", "").strip()
        if result_header not in {"1-0", "0-1", "1/2-1/2"}:
            result["status"] = INVALID_RESULT_MISSING
            result["reason"] = f"result_header={result_header!r}"
            result["ply_count"] = ply
            return result

        if trailing_result is not None and trailing_result != result_header:
            result["status"] = INVALID_CORRUPT
            result["reason"] = (
                f"result_header={result_header!r} distinto de resultado en movetext="
                f"{trailing_result!r}"
            )
            result["ply_count"] = ply
            return result

        identity_hash = compute_identity_hash(variant_norm, initial_fen_normalized, uci_sequence)

        result.update({
            "status": VALID,
            "reason": None,
            "ply_count": ply,
            "action_ids": action_ids,
            "identity_hash": identity_hash,
            "initial_fen": initial_fen_normalized,
            "variant": variant_norm,
            "result_code": {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[result_header],
        })
        return result

    except Exception:
        result["status"] = FAILED_EXCEPTION
        result["reason"] = "excepcion_no_controlada"
        result["exception_text"] = traceback.format_exc()
        return result
