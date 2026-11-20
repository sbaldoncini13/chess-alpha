"""Verificacion de pruebas de DFPN de search_v1 (Fase VI).

Dos responsabilidades separadas:
1. `verify_mate_proof()`: revalida un `DFPNResult` YA producido - la PV es
   legal jugada por jugada, termina en jaque mate real, la distancia
   coincide con la longitud de la PV, la posicion original queda intacta.
2. `exhaustive_mate_oracle()`: busqueda de mate EXHAUSTIVA, independiente
   de DFPN (sin tabla de transposiciones, sin poda de ningun tipo,
   recursion directa OR/AND, orden UCI determinista) - el "oraculo de
   referencia" pedido explicitamente para fixtures chicos, usado SOLO en
   tests/benchmarks para comparar contra DFPN, nunca en produccion (séria
   demasiado costoso sin memoizacion en posiciones no triviales)."""

from __future__ import annotations

from search_v1.move import Move
from search_v1.move_generator import generate_legal_moves
from search_v1.position import Position
from search_v1.terminal import TerminalStatus, detect_terminal, is_fifty_move_rule


def verify_mate_proof(original_position: Position, result) -> list[str]:
    """Devuelve la lista de problemas encontrados (vacia si la prueba es
    valida). `result` es un DFPNResult con proof_status==PROVEN_MATE."""
    from search_v1.proof_status import ProofStatus

    problems = []
    if result.proof_status != ProofStatus.PROVEN_MATE:
        return ["verify_mate_proof llamado con proof_status distinto de PROVEN_MATE"]

    if not result.principal_variation:
        problems.append("PROVEN_MATE sin PV")
        return problems

    if result.mate_distance_plies != len(result.principal_variation):
        problems.append(
            f"mate_distance_plies ({result.mate_distance_plies}) no coincide con la longitud de la PV ({len(result.principal_variation)})"
        )

    working = original_position.copy()
    for i, uci in enumerate(result.principal_variation):
        terminal = detect_terminal(working)
        if terminal.is_terminal:
            problems.append(f"PV contiene una jugada ({uci!r}) despues de una posicion terminal en el indice {i}")
            break
        try:
            move = Move.from_uci(uci)
        except ValueError:
            problems.append(f"PV[{i}]={uci!r} no es un UCI valido")
            break
        if not working.is_legal(move):
            problems.append(f"PV[{i}]={uci!r} no es legal en la posicion alcanzada")
            break
        working.apply(move)
    else:
        final_terminal = detect_terminal(working)
        if final_terminal.status != TerminalStatus.CHECKMATE:
            problems.append(f"la PV no termina en jaque mate (termino en {final_terminal.status})")

    return problems


def exhaustive_mate_oracle(position: Position, max_depth_plies: int, game_repetition=None):
    """Busqueda de mate exhaustiva de referencia. Devuelve
    (mate_distance_plies | None, pv: list[str]) - None si no se encontro
    mate DENTRO de `max_depth_plies` (no es una prueba de "nunca", solo de
    "no en este horizonte" - mismo matiz que PROVEN_NO_MATE_WITHIN_BOUND
    de DFPN, para poder comparar de forma justa)."""
    working = position.copy()
    path_keys: list[int] = []

    def classify_terminal_leaf(pos, is_or_node):
        terminal = detect_terminal(pos)
        if terminal.status == TerminalStatus.CHECKMATE:
            return "win" if not is_or_node else "loss"
        if terminal.status in (TerminalStatus.STALEMATE, TerminalStatus.INSUFFICIENT_MATERIAL):
            return "draw"
        if is_fifty_move_rule(pos):
            return "draw"
        if game_repetition is not None and game_repetition.is_threefold_repetition(pos):
            return "draw"
        # path_keys[:-1]: excluye la clave del propio `pos` (el llamador ya
        # la agrego a path_keys ANTES de invocar rec()) - solo interesan
        # los ANCESTROS reales, no la coincidencia trivial de un nodo
        # consigo mismo (mismo bug ya corregido en dfpn.py::_expand).
        if pos.zobrist_key in path_keys[:-1]:
            return "draw"
        return None

    def rec(pos, is_or_node, depth_remaining):
        leaf = classify_terminal_leaf(pos, is_or_node)
        if leaf == "win":
            return 0, []
        if leaf in ("loss", "draw"):
            return None, []
        if depth_remaining <= 0:
            return None, []

        moves = sorted(generate_legal_moves(pos), key=lambda lm: lm.uci)
        if is_or_node:
            best_distance, best_pv = None, []
            for lm in moves:
                pos.apply(lm.move)
                path_keys.append(pos.zobrist_key)
                child_distance, child_pv = rec(pos, False, depth_remaining - 1)
                path_keys.pop()
                pos.undo()
                if child_distance is not None:
                    candidate = child_distance + 1
                    if best_distance is None or candidate < best_distance:
                        best_distance, best_pv = candidate, [lm.uci] + child_pv
            return best_distance, best_pv
        else:
            worst_distance, worst_pv = 0, []
            for lm in moves:
                pos.apply(lm.move)
                path_keys.append(pos.zobrist_key)
                child_distance, child_pv = rec(pos, True, depth_remaining - 1)
                path_keys.pop()
                pos.undo()
                if child_distance is None:
                    return None, []  # el defensor escapa via esta jugada - no hay mate forzado
                candidate = child_distance + 1
                if candidate > worst_distance:
                    worst_distance, worst_pv = candidate, [lm.uci] + child_pv
            return worst_distance, worst_pv

    distance, pv = rec(working, True, max_depth_plies)
    return distance, pv
