"""Move ordering avanzado de search_v1 (Fase III).

ORDEN (documentado exactamente, adaptado del sugerido por el pedido -
se preserva ademas la deteccion barata de "mate inmediato" ya validada en
Fase II inmediatamente despues del hash move, por ser igual de barata y
valiosa; el pedido permite ajustar el orden si queda explicito):

    tier 0 - hash move (de la TT, SIEMPRE validado legal antes de usarse)
    tier 1 - mate inmediato (1 ply de verificacion barata, igual que Fase II)
    tier 2 - promociones
    tier 3 - capturas, ordenadas por MVV-LVA descendente
    tier 4 - killer moves (no capturantes, validados legales)
    tier 5 - countermove (no capturante, validado legal)
    tier 6 - resto, ordenado por history heuristic descendente
    (dentro de cada tier: desempate UCI determinista)

No se realiza NINGUNA busqueda adicional para ordenar mas alla del chequeo
de mate inmediato ya usado en Fase II (1 ply, acotado). No se usa Policy
ni Sequence Policy."""

from __future__ import annotations

from search_v1.position import Position
from search_v1.static_evaluation import PIECE_VALUES_CP
from search_v1.terminal import TerminalStatus, detect_terminal

MAX_KILLERS_PER_PLY = 2
MAX_HISTORY_VALUE = 1_000_000  # umbral de decay - ver OrderingState.update_history

_TIER_HASH = 0
_TIER_MATE = 1
_TIER_PROMOTION = 2
_TIER_CAPTURE = 3
_TIER_KILLER = 4
_TIER_COUNTERMOVE = 5
_TIER_HISTORY = 6


class OrderingState:
    """Estado de las heuristicas de ordenamiento para UNA corrida de
    iterative_deepening_search (killers/history/countermove NO persisten
    entre llamadas distintas a iterative_deepening_search - "reinicio
    controlado" pedido explicitamente - pero SI persisten entre las
    iteraciones de profundidad creciente DENTRO de una misma llamada,
    que es donde realmente aportan valor)."""

    def __init__(self, max_ply: int = 128):
        self.killers: list[list[str]] = [[] for _ in range(max_ply + 1)]
        self.history: dict[tuple[str, str], int] = {}
        self.countermove: dict[str, str] = {}

        # Telemetria (Fase III)
        self.hash_move_first_count = 0
        self.killer_cutoffs = 0
        self.history_updates = 0
        self.countermove_hits = 0
        self.ordering_time_ms = 0.0

    def _ply_killers(self, ply: int) -> list[str]:
        if ply >= len(self.killers):
            return []
        return self.killers[ply]

    def update_killer(self, ply: int, uci: str) -> None:
        if ply >= len(self.killers):
            return
        bucket = self.killers[ply]
        if uci in bucket:
            return  # sin duplicados
        bucket.insert(0, uci)
        del bucket[MAX_KILLERS_PER_PLY:]

    def update_history(self, from_sq: str, to_sq: str, depth: int) -> None:
        key = (from_sq, to_sq)
        self.history[key] = self.history.get(key, 0) + depth * depth
        self.history_updates += 1
        if self.history[key] > MAX_HISTORY_VALUE:
            # decay: reduce TODAS las entradas a la mitad para evitar
            # overflow y para que valores viejos pierdan peso relativo
            # frente a actividad reciente (evita que una racha temprana
            # domine para siempre el ordenamiento)
            for k in self.history:
                self.history[k] //= 2

    def update_countermove(self, prev_move_uci: str, response_uci: str) -> None:
        self.countermove[prev_move_uci] = response_uci

    def history_score(self, from_sq: str, to_sq: str) -> int:
        return self.history.get((from_sq, to_sq), 0)


def _mvv_lva_score(position: Position, lm) -> int:
    """Mayor = mejor (victima valiosa capturada por atacante barato).
    Al paso: la victima es un peon aunque el destino este vacio. Promocion
    con captura: victima = pieza en la casilla destino, atacante = peon
    (unica pieza que promociona)."""
    board = position.board
    if lm.is_en_passant:
        victim_value = PIECE_VALUES_CP[1]  # chess.PAWN == 1
    else:
        victim = board.piece_at(lm.move.raw.to_square)
        victim_value = PIECE_VALUES_CP[victim.piece_type] if victim else 0
    attacker = board.piece_at(lm.move.raw.from_square)
    attacker_value = PIECE_VALUES_CP[attacker.piece_type] if attacker else 0
    return victim_value * 100 - attacker_value


def _is_immediate_mate(position: Position, lm) -> bool:
    if not lm.gives_check:
        return False
    position.apply(lm.move)
    is_mate = detect_terminal(position).status == TerminalStatus.CHECKMATE
    position.undo()
    return is_mate


def order_moves(
    position: Position,
    moves: list,
    hash_move_uci: str | None,
    ply: int,
    ordering_state: OrderingState,
    prev_move_uci: str | None,
) -> list:
    legal_ucis = {lm.uci for lm in moves}
    by_uci = {lm.uci: lm for lm in moves}

    validated_hash_move = hash_move_uci if hash_move_uci in legal_ucis else None
    killer_candidates = [u for u in ordering_state._ply_killers(ply) if u in legal_ucis]
    countermove_uci = ordering_state.countermove.get(prev_move_uci) if prev_move_uci else None
    validated_countermove = countermove_uci if countermove_uci in legal_ucis else None

    keyed = []
    for lm in moves:
        if lm.uci == validated_hash_move:
            tier, secondary = _TIER_HASH, 0
        elif _is_immediate_mate(position, lm):
            tier, secondary = _TIER_MATE, 0
        elif lm.is_promotion:
            tier, secondary = _TIER_PROMOTION, 0
        elif lm.is_capture:
            tier, secondary = _TIER_CAPTURE, -_mvv_lva_score(position, lm)
        elif lm.uci in killer_candidates:
            tier, secondary = _TIER_KILLER, killer_candidates.index(lm.uci)
        elif lm.uci == validated_countermove:
            tier, secondary = _TIER_COUNTERMOVE, 0
        else:
            tier, secondary = _TIER_HISTORY, -ordering_state.history_score(lm.move.from_square, lm.move.to_square)
        keyed.append((tier, secondary, lm.uci, lm))

    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    ordered = [lm for *_, lm in keyed]

    # telemetria de COBERTURA de la heuristica (existio una sugerencia
    # legal disponible para esta posicion), no de si termino ganando el
    # cutoff (eso se mide en alpha_beta.py, que es quien sabe si realmente
    # se produjo un cutoff con esa jugada)
    if validated_hash_move:
        ordering_state.hash_move_first_count += 1
    if validated_countermove:
        ordering_state.countermove_hits += 1

    return ordered


# --- Ordenamiento TACTICO (Fase IV, quiescence) ---
#
# Orden (documentado exactamente, tal como pide el pedido de Fase IV):
#   tier 0 - hash move tactico (de la TT, SIEMPRE validado legal)
#   tier 1 - promociones
#   tier 2 - capturas (incluye captura al paso - ya es un tipo de captura,
#            LegalMove.is_capture/is_en_passant ya la distingue; no se
#            crea un tier separado redundante para ella, MVV-LVA ya la
#            valora correctamente via _mvv_lva_score) ordenadas MVV-LVA
#   tier 3 - jaques tacticos permitidos (solo si estan en la lista de
#            movimientos recibida - generate_tactical_moves ya decide si
#            se incluyen o no segun allow_tactical_checks)
#   tier 4 - resto (evasiones no tacticas cuando hay jaque, etc.) - orden
#            UCI determinista; si se provee ordering_state, killers/
#            history se usan SOLO como desempate secundario dentro de
#            este tier (nunca como senal de correccion tactica, tal como
#            exige el pedido explicitamente)
_TIER_TACTICAL_HASH = 0
_TIER_TACTICAL_PROMOTION = 1
_TIER_TACTICAL_CAPTURE = 2
_TIER_TACTICAL_CHECK = 3
_TIER_TACTICAL_REST = 4


def order_tactical_moves(
    position: Position,
    moves: list,
    hash_move_uci: str | None = None,
    ordering_state=None,
) -> list:
    legal_ucis = {lm.uci for lm in moves}
    validated_hash_move = hash_move_uci if hash_move_uci in legal_ucis else None

    keyed = []
    for lm in moves:
        if lm.uci == validated_hash_move:
            tier, secondary = _TIER_TACTICAL_HASH, 0
        elif lm.is_promotion:
            tier, secondary = _TIER_TACTICAL_PROMOTION, 0
        elif lm.is_capture:
            tier, secondary = _TIER_TACTICAL_CAPTURE, -_mvv_lva_score(position, lm)
        elif lm.gives_check:
            tier, secondary = _TIER_TACTICAL_CHECK, 0
        else:
            secondary = -ordering_state.history_score(lm.move.from_square, lm.move.to_square) if ordering_state else 0
            tier = _TIER_TACTICAL_REST
        keyed.append((tier, secondary, lm.uci, lm))

    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [lm for *_, lm in keyed]
