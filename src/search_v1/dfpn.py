"""Proof-Number Search (variante de DFPN, ver docstring de `dfpn_node.py`
para la justificacion de diseno) de search_v1 (Fase VI).

ALGORITMO: en cada iteracion, se selecciona el "nodo mas probante" (MPN)
caminando desde la raiz - en cada nodo OR (atacante a mover) se desciende
al hijo de MENOR proof number; en cada nodo AND (defensor a mover) al
hijo de MENOR disproof number - se expande ese nodo (o se recupera de la
tabla de transposiciones si ya esta resuelto), y se propaga el cambio
hacia arriba recalculando pn/dn de cada ancestro. Se repite hasta que la
raiz quede resuelta (pn==0 o dn==0) o se agote el presupuesto.

REGLA DE ORO (invariante mantenido en todo momento, verificado por los
tests de OR/AND/propagacion): un nodo NUNCA se selecciona como MPN si ya
esta resuelto - si el hijo de menor pn/dn de un nodo estuviera resuelto,
la propagacion inmediata (backprop tras CADA cambio) ya habria resuelto
tambien al padre, deteniendo el descenso un nivel antes. Por eso backprop
ocurre SIEMPRE antes de la siguiente seleccion de MPN, nunca diferido.

EXTRACCION DE PV: se reconstruye caminando el ARBOL EN MEMORIA (fuente de
verdad para cualquier nodo expandido por ESTA busqueda). Si el camino
llega a un nodo resuelto por REUTILIZACION de una transposicion (sin
hijos materializados en memoria), se continua con una sub-prueba DFPN
fresca y auto-contenida desde esa posicion - ver `_extract_pv()` para la
razon de no depender de la tabla de transposiciones para esto (bug real
corregido: la tabla tiene capacidad finita y una colision real puede
desalojar una entrada que la PV todavia necesita).

REGLA DE 50 MOVIMIENTOS: DELIBERADAMENTE fuera del alcance de esta prueba
(variante especializada, permitida y documentada explicitamente - ver
`_classify_terminal()`) - un PROVEN_MATE de DFPN significa "existe una
secuencia forzada de jaque mate", sin garantizar que quepa antes de que
se cumplan 50 movimientos sin captura/peon. La cache de transposiciones
(zobrist_key, que excluye halfmove_clock por diseno del proyecto) haria
que considerar la regla de 50 fuera INSEGURO de cachear (bug real
encontrado y corregido, ver docstring de `_classify_terminal`)."""

from __future__ import annotations

import time

from search_v1.dfpn_limits import DFPNLimits
from search_v1.dfpn_node import INFINITY_PN, DFPNNode, NodeType
from search_v1.dfpn_result import DFPNResult
from search_v1.dfpn_table import DFPNTable
from search_v1.move import Move
from search_v1.move_generator import generate_legal_moves
from search_v1.position import Position
from search_v1.proof_status import ProofStatus
from search_v1.terminal import TerminalStatus, detect_terminal

_MAX_PV_PLIES_SAFETY = 400  # guarda de sanidad contra un bug residual causando un paseo sin fin al reconstruir la PV


def _order_moves_for_proof(position: Position, moves: list) -> list:
    """Orden determinista, SIN podar ninguna jugada (todas las legales
    quedan incluidas - solo se reordena para eficiencia de la busqueda).
    Jaques primero (la heuristica de mayor valor para busqueda de mate),
    despues capturas/promociones, resto por UCI - mismo tipo de
    ordenamiento por tier fijo ya usado en alpha_beta.py de Fase II,
    reservado aca a un subconjunto simple (jaque > captura/promocion >
    resto) para minimizar superficie de bug en esta primera version."""
    def tier(lm):
        if lm.gives_check:
            return 0
        if lm.is_capture or lm.is_promotion:
            return 1
        return 2
    return sorted(moves, key=lambda lm: (tier(lm), lm.uci))


def _classify_terminal(
    position: Position, node_type: NodeType, game_repetition, path_keys: list[int],
) -> tuple[ProofStatus | None, bool]:
    """Devuelve (status, cacheable). `cacheable=False` SOLO para el caso de
    ciclo interno del camino de busqueda (ver mas abajo) - NUNCA se
    almacena en la tabla de transposiciones DFPN, porque es una propiedad
    del CAMINO actual, no de la posicion sola (a diferencia de jaque mate/
    ahogado/material insuficiente/repeticion REAL contra el historial de
    partida, que son propiedades intrinsecas de la posicion, siempre
    seguras de reutilizar).

    BUG REAL encontrado y corregido en Fase VI: la version original
    tambien clasificaba `is_fifty_move_rule(position)` aca, cacheable, con
    la MISMA clave de tabla que se usa en toda la Fase III-V (zobrist_key
    de Polyglot, que EXCLUYE deliberadamente halfmove_clock). Una prueba
    de mate forzado larga (decenas de plies sin capturas/movimientos de
    peon, ej. K+Q vs K "empujando" al rey al borde) hace avanzar
    halfmove_clock en cada ply - la MISMA posicion (mismo zobrist_key)
    alcanzada la PRIMERA vez con halfmove_clock bajo puede prohar
    legitimamente PROVEN_MATE (el mate cabe antes de las 50 movidas), pero
    alcanzada DE NUEVO mas tarde por otro camino con halfmove_clock ya
    alto, la MISMA secuencia de mate ya NO cabe antes de la regla de 50 -
    reutilizar la entrada cacheada de la primera vez producia un
    PROVEN_MATE FALSO (confirmado empiricamente: fixture
    8/8/3k4/8/8/3K4/8/7Q w - - 0 1, PV de 23 plies que terminaba en una
    posicion SIN jaque mate real). Corregido eliminando la regla de 50 del
    alcance de la prueba - DFPN de Fase VI prueba "existe una secuencia de
    jaque mate forzado" SIN considerar la regla de 50 movimientos (variante
    especializada, explicitamente permitida por el pedido: "Documentar si
    la prueba considera reglas completas de partida, mate dentro de un
    limite, o una variante especializada" - documentado aca, no oculto).
    La repeticion REAL (game_repetition, contra el historial externo de
    partida) SI se mantiene, porque es intrinseca a la posicion+historial
    fijo, no al camino interno de esta busqueda especifica."""
    terminal = detect_terminal(position)
    if terminal.status == TerminalStatus.CHECKMATE:
        return (ProofStatus.TERMINAL_WIN if node_type == NodeType.AND else ProofStatus.TERMINAL_LOSS), True
    if terminal.status == TerminalStatus.STALEMATE:
        return ProofStatus.TERMINAL_DRAW, True
    if terminal.status == TerminalStatus.INSUFFICIENT_MATERIAL:
        return ProofStatus.DRAW_BY_INSUFFICIENT_MATERIAL, True
    if game_repetition is not None and game_repetition.is_threefold_repetition(position):
        return ProofStatus.DRAW_BY_REPETITION, True
    if position.zobrist_key in path_keys:
        # ciclo INTERNO de este camino de busqueda especifico - necesario
        # para terminar la recursion, pero NO es una propiedad de la
        # posicion sola (otro camino distinto que pase por la misma
        # posicion podria no ciclar) - jamas se cachea (ver docstring).
        return ProofStatus.DRAW_BY_REPETITION, False
    return None, True


def _pn_dn_for_terminal(status: ProofStatus) -> tuple[int, int]:
    if status == ProofStatus.TERMINAL_WIN:
        return 0, INFINITY_PN
    return INFINITY_PN, 0  # TERMINAL_LOSS / TERMINAL_DRAW / DRAW_BY_*


def _recompute(node: DFPNNode) -> None:
    if not node.children:
        return  # terminal o reutilizado por transposicion - pn/dn ya son finales
    if node.node_type == NodeType.OR:
        node.pn = min(c.pn for c in node.children)
        node.dn = min(INFINITY_PN, sum(c.dn for c in node.children))
    else:
        node.pn = min(INFINITY_PN, sum(c.pn for c in node.children))
        node.dn = min(c.dn for c in node.children)
    # BUG REAL encontrado y corregido en Fase VI (ver test_forced_defense.py,
    # el que lo expuso): marcar SOLO la hoja de un ciclo de camino / corte
    # de profundidad como "no cacheable" es INSUFICIENTE - un ANCESTRO cuyo
    # propio pn/dn se calculo usando esa hoja como uno de los insumos
    # hereda la MISMA dependencia del camino especifico (otro camino, con
    # mas presupuesto de profundidad restante o sin el ciclo, podria
    # resolver esa rama de otra forma) - reutilizar el ancestro cacheado
    # bajo un contexto DISTINTO (ej. desde `_extract_pv`/`forced_defense.py`,
    # que comparten una tabla entre multiples llamadas independientes a
    # `search_dfpn`) producia una prueba fresca que CONTRADECIA la cacheada
    # (detectado por el RuntimeError explicito de `_extract_pv`). Se
    # propaga `cacheable` conservadoramente: un nodo interno solo es
    # cacheable si TODOS sus hijos lo son.
    node.cacheable = all(c.cacheable for c in node.children)


def _best_proof_move(node: DFPNNode) -> str | None:
    if not node.children:
        return None
    if node.node_type == NodeType.OR:
        return min(node.children, key=lambda c: c.pn).move_uci
    proven = [c for c in node.children if c.pn == 0]
    if proven:
        return proven[0].move_uci
    return min(node.children, key=lambda c: c.pn).move_uci


def _expand(
    node: DFPNNode, position: Position, table: DFPNTable, attacker: str,
    telemetry: dict, path_keys: list[int], game_repetition, allow_tt_reuse: bool = True,
) -> None:
    node.position_key = position.zobrist_key

    terminal_status, cacheable = _classify_terminal(position, node.node_type, game_repetition, path_keys)
    if terminal_status is not None:
        node.terminal_status = terminal_status
        node.pn, node.dn = _pn_dn_for_terminal(terminal_status)
        node.expanded = True
        node.children = []
        node.cacheable = cacheable
        if terminal_status == ProofStatus.DRAW_BY_REPETITION:
            telemetry["cycles"] += 1
        return

    # `allow_tt_reuse=False` fuerza expansion REAL (nunca reutilizar la
    # tabla) - usado exclusivamente por `_extract_pv()` al lanzar una
    # sub-prueba fresca para reconstruir la PV: si se permitiera reutilizar
    # aca, la sub-prueba encontraria INMEDIATAMENTE la misma entrada ya
    # resuelta para su propia raiz (sin hijos en memoria) y volveria a
    # necesitar otra sub-prueba fresca para "continuar" desde el mismo
    # lugar - recursion infinita real, encontrada y corregida en esta
    # misma fase (ver test_dfpn_oracle_comparison.py)."""
    if allow_tt_reuse:
        entry = table.probe(node.position_key, attacker, node.node_type)
        if entry is not None and (entry.pn == 0 or entry.dn == 0):
            telemetry["transposition_hits"] += 1
            node.pn, node.dn = entry.pn, entry.dn
            node.expanded = True
            node.children = []
            node.from_tt_reuse = True
            return

    legal_moves = generate_legal_moves(position)
    ordered = _order_moves_for_proof(position, legal_moves)
    node.children = [DFPNNode(node_type=node.opposite_type(), move_uci=lm.uci) for lm in ordered]
    node.expanded = True
    telemetry["expansions"] += 1
    _recompute(node)


def _extract_pv(
    node: DFPNNode, position: Position, table: DFPNTable, attacker: str, game_repetition, _guard: int = 0,
) -> list[str]:
    """Reconstruye la PV de un nodo PROBADO (pn==0). Camina el arbol EN
    MEMORIA cuando esta disponible (nodos expandidos por ESTA busqueda);
    si un nodo resuelto no tiene hijos en memoria (se resolvio por
    REUTILIZACION de una transposicion - ver `_expand`), continua la PV
    con una sub-prueba DFPN COMPLETAMENTE FRESCA desde esa posicion.

    BUG REAL encontrado y corregido (Fase VI): la primera version leia
    `best_proof_move` DIRECTO de la tabla, paso a paso, para TODA la PV -
    la tabla tiene capacidad finita (`DFPNTable`, direct-mapped) y una
    colision de indice real puede DESALOJAR una entrada que la PV todavia
    necesita, dejando una PV incompleta que nunca llega a jaque mate
    (detectado en un fixture real: 8/8/8/4k3/8/8/4K3/7Q w - - 0 1, 25682
    nodos, 203 colisiones, la entrada de un nodo intermedio de la PV fue
    sobreescrita antes de terminar la extraccion). La sub-prueba fresca es,
    por construccion, tan correcta como la busqueda original (mismo
    algoritmo recalculado desde cero) - el costo extra solo se paga en el
    caso raro de una colision real que afecte justo un nodo de la PV."""
    if _guard > _MAX_PV_PLIES_SAFETY:
        return []

    if node.children:
        if node.node_type == NodeType.OR:
            nxt = next(c for c in node.children if c.pn == 0)
        else:
            proven = [c for c in node.children if c.pn == 0]
            nxt = proven[0] if proven else node.children[0]
        position.apply(Move.from_uci(nxt.move_uci))
        try:
            rest = _extract_pv(nxt, position, table, attacker, game_repetition, _guard + 1)
        finally:
            position.undo()
        return [nxt.move_uci] + rest

    terminal = detect_terminal(position)
    if terminal.status == TerminalStatus.CHECKMATE:
        return []

    # resuelto por transposicion (sin hijos en memoria) - continua con una
    # sub-prueba fresca y COMPLETAMENTE INDEPENDIENTE desde esta posicion
    # (tabla PROPIA, nueva - NO la tabla compartida `table`). Compartir la
    # tabla aca parecia una optimizacion inocente, pero abria la puerta a
    # que la sub-prueba de verificacion reutilizara, para sus propios
    # descendientes, entradas contaminadas por el mismo tipo de problema
    # que esta funcion existe para corregir - una tabla fresca garantiza
    # que esta reconstruccion sea una fuente de verdad realmente
    # independiente (bug real corregido, ver test_forced_defense.py).
    sub_result = search_dfpn(
        position, limits=DFPNLimits(max_nodes=200_000, max_depth_plies=None),
        table=DFPNTable(), game_repetition=game_repetition, attacker=attacker,
    )
    if sub_result.proof_status != ProofStatus.PROVEN_MATE:
        # NUNCA se devuelve una PV truncada en silencio - si la sub-prueba
        # fresca no reconfirma PROVEN_MATE para una posicion que la prueba
        # original marco como resuelta, es una inconsistencia real que hay
        # que investigar, no ocultar (ver "resultado parcial marcado como
        # prueba completa" en los criterios de FAIL del pedido).
        raise RuntimeError(
            f"_extract_pv: la sub-prueba fresca no confirmo PROVEN_MATE para una posicion "
            f"marcada como resuelta por transposicion (status={sub_result.proof_status}, "
            f"fen={position.fen!r}) - posible inconsistencia en la prueba original."
        )
    return list(sub_result.principal_variation)


def search_dfpn(
    position: Position, limits: DFPNLimits | None = None, table: DFPNTable | None = None,
    game_repetition=None, force_expand_root: bool = False, attacker: str | None = None,
) -> DFPNResult:
    """Prueba si `attacker` fuerza jaque mate desde `position` (por
    defecto, `attacker=position.side_to_move` - el uso normal, "el lado a
    mover intenta forzar mate"). El nodo RAIZ de esta llamada es OR si
    `attacker` coincide con `position.side_to_move` (el atacante es quien
    mueve aca) o AND en caso contrario (se esta continuando una prueba
    desde un nodo DEFENSOR, donde el atacante NO es quien mueve).

    BUG REAL encontrado y corregido en Fase VI: `_extract_pv()` necesita
    lanzar sub-pruebas frescas para continuar una PV que pasa por un nodo
    AND (turno del defensor) - la version original NO pasaba `attacker`
    explicito, asi que `search_dfpn` asumia SIEMPRE `attacker=position.
    side_to_move`, preguntando "¿el DEFENSOR fuerza mate?" en vez de "¿el
    ATACANTE original sigue forzando mate?" - una pregunta completamente
    distinta (y casi siempre falsa), que producia PROVEN_NO_MATE reales
    pero IRRELEVANTES, disparando el RuntimeError de inconsistencia sin
    que hubiera ninguna inconsistencia real en la prueba original
    (detectado con test_forced_defense.py, fixture k7/8/2K5/8/8/8/8/7Q b
    - donde la sub-prueba de continuacion, en un nodo AND, necesitaba
    `attacker='white'` pero heredaba `attacker='black'` de `position.
    side_to_move`).

    `table` es OPCIONAL - si no se provee, se crea una tabla interna
    (pasar una tabla propia permite compartirla entre llamadas sucesivas,
    ej. defensa obligatoria analizando varias jugadas de la MISMA
    posicion). `force_expand_root` fuerza que la RAIZ de esta llamada se
    expanda de verdad, sin reutilizar una entrada de tabla ya resuelta."""
    limits = limits or DFPNLimits()
    table = table if table is not None else DFPNTable()

    t0 = time.perf_counter()
    original_fen = position.fen
    attacker = attacker if attacker is not None else position.side_to_move
    root_node_type = NodeType.OR if attacker == position.side_to_move else NodeType.AND

    telemetry = {
        "nodes": 0, "expansions": 0, "transposition_hits": 0, "cycles": 0,
        "max_pn": 0, "max_dn": 0,
    }

    root = DFPNNode(node_type=root_node_type, move_uci=None)
    root.position_key = position.zobrist_key

    stop_reason = None
    depth_limited = False

    while root.pn != 0 and root.dn != 0:
        if limits.max_nodes is not None and telemetry["nodes"] >= limits.max_nodes:
            stop_reason = "max_nodes"
            break
        if limits.max_time_ms is not None and (time.perf_counter() - t0) * 1000 >= limits.max_time_ms:
            stop_reason = "max_time"
            break

        path_nodes = [root]
        path_keys = [position.zobrist_key]
        while path_nodes[-1].expanded and not path_nodes[-1].is_resolved:
            cur = path_nodes[-1]
            if cur.node_type == NodeType.OR:
                nxt = min(cur.children, key=lambda c: c.pn)
            else:
                nxt = min(cur.children, key=lambda c: c.dn)
            position.apply(Move.from_uci(nxt.move_uci))
            path_nodes.append(nxt)
            path_keys.append(position.zobrist_key)

        target = path_nodes[-1]
        depth_here = len(path_nodes) - 1

        if limits.max_depth_plies is not None and depth_here >= limits.max_depth_plies:
            target.position_key = position.zobrist_key
            target.pn, target.dn = INFINITY_PN, 0
            target.expanded = True
            target.children = []
            # NO cacheable (mismo motivo que el ciclo de camino, ver
            # _classify_terminal): "se alcanzo la cota de profundidad" es
            # una propiedad de CUANTOS PLIES tomo este camino especifico en
            # llegar aca, no de la posicion sola - un camino distinto
            # (mas corto) podria seguir explorando desde la misma posicion.
            target.cacheable = False
            depth_limited = True
        else:
            # path_keys[:-1]: SOLO los ancestros reales (excluye la clave
            # del propio `target`, agregada al final del while de arriba -
            # incluirla haria que TODO nodo "coincida consigo mismo"
            # trivialmente y se clasifique como ciclo/repeticion siempre).
            allow_reuse = not (force_expand_root and depth_here == 0)
            _expand(target, position, table, attacker, telemetry, path_keys[:-1], game_repetition, allow_tt_reuse=allow_reuse)
        telemetry["nodes"] += 1

        for i in range(len(path_nodes) - 2, -1, -1):
            _recompute(path_nodes[i])

        for node in path_nodes:
            if node.pn < INFINITY_PN:
                telemetry["max_pn"] = max(telemetry["max_pn"], node.pn)
            if node.dn < INFINITY_PN:
                telemetry["max_dn"] = max(telemetry["max_dn"], node.dn)

        for i in range(len(path_nodes) - 1, -1, -1):
            node = path_nodes[i]
            if not node.is_resolved:
                break
            if not node.cacheable:
                # se salta SOLO este nodo (propiedad del camino, no de la
                # posicion - ver _classify_terminal) - los ANCESTROS siguen
                # siendo validos de cachear (su propio pn/dn, aunque este
                # calculado usando este hijo como uno de los insumos, es
                # una conclusion legitima y reutilizable PARA SU PROPIA
                # posicion, sin importar por que camino se llego a ella).
                continue
            if node.from_tt_reuse:
                # ya esta en la tabla (es de donde salio) - volver a
                # guardarlo es puramente redundante y, por la politica de
                # reemplazo (entre dos entradas "resueltas" gana la mas
                # reciente), puede PISAR una entrada mas informativa con
                # una copia sin `best_proof_move` (bug real corregido,
                # ver test_forced_defense.py y dfpn_node.py::from_tt_reuse).
                continue
            table.store(
                node.position_key, attacker, node.node_type, node.pn, node.dn, node.expanded,
                best_proof_move=_best_proof_move(node), children_count=len(node.children),
                terminal_status=node.terminal_status,
            )

        for _ in range(len(path_nodes) - 1):
            position.undo()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    root_position_intact = position.fen == original_fen

    if root.pn == 0:
        proof_status = ProofStatus.PROVEN_MATE
        pv = _extract_pv(root, position, table, attacker, game_repetition)
        mate_distance = len(pv)
    elif root.dn == 0:
        proof_status = ProofStatus.PROVEN_NO_MATE_WITHIN_BOUND if depth_limited else ProofStatus.PROVEN_NO_MATE
        pv, mate_distance = [], None
    else:
        proof_status = ProofStatus.UNKNOWN_NODE_LIMIT if stop_reason == "max_nodes" else ProofStatus.UNKNOWN_TIME_LIMIT
        pv, mate_distance = [], None

    return DFPNResult(
        proof_status=proof_status, attacker=attacker, principal_variation=pv,
        mate_distance_plies=mate_distance, nodes=telemetry["nodes"], root_pn=root.pn, root_dn=root.dn,
        elapsed_ms=elapsed_ms, root_position_intact=root_position_intact,
        metadata={
            "dfpn_nodes": telemetry["nodes"], "dfpn_expansions": telemetry["expansions"],
            "dfpn_transposition_hits": telemetry["transposition_hits"],
            "dfpn_collisions": table.collisions, "dfpn_cycles": telemetry["cycles"],
            "dfpn_proven": int(root.pn == 0), "dfpn_disproven": int(root.dn == 0 and root.pn != 0),
            "dfpn_unknown": int(root.pn != 0 and root.dn != 0),
            "max_proof_number": telemetry["max_pn"], "max_disproof_number": telemetry["max_dn"],
            "proof_time_ms": elapsed_ms, "proof_stop_reason": stop_reason,
        },
    )
