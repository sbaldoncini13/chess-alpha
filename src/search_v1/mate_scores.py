"""Convencion centralizada de mate scores de search_v1 (Fase II).

Se usa la convencion "MATE_SCORE - ply_from_root", donde ply_from_root es
la profundidad ABSOLUTA desde la posicion raiz de la busqueda (no relativa
al nodo actual). Es la forma mas simple y menos propensa a errores de
implementar mate scores en un negamax/alpha-beta plano: al detectar jaque
mate en un nodo, el score en la perspectiva de ESE nodo (el lado a mover
ahi, que acaba de recibir jaque mate) es SIEMPRE
`-(MATE_SCORE - ply_from_root)` - muy negativo. Como ply_from_root ya
crece con cada nivel de recursion, la propagacion normal de negamax
(`score = -child_score`) ya ajusta correctamente la distancia sin ninguna
logica especial adicional en cada nivel - no hace falta sumar/restar 1
manualmente al subir por el arbol, a diferencia de la convencion clasica
"distancia relativa al nodo" (mas propensa a bugs de off-by-one).

Consecuencias automaticas de esta convencion (verificadas en los tests):
- Mates mas cortos (ply_from_root chico) dan |score| mas grande -> negamax
  los prefiere automaticamente sobre mates mas largos, sin logica extra.
- Un mate inevitable en contra se retrasa automaticamente lo mas posible:
  entre dos derrotas forzadas, la que tarda mas ply_from_root en
  concretarse tiene un score MENOS negativo, y negamax maximiza su propio
  score incluso entre opciones perdedoras.

MATE_SCORE se fija muy por encima de cualquier evaluacion estatica
material realista (balance material acotado por PIECE_VALUES_CP de
static_evaluation.py, incluso con ejercitos completos de piezas
promovidas) - garantiza que un score de mate nunca se confunda con un
score de centipawns ni haga overflow en las comparaciones alpha/beta."""

from __future__ import annotations

from search_v1.terminal import TerminalResult, TerminalStatus

MATE_SCORE = 1_000_000
# Cualquier score con |valor| > MATE_THRESHOLD se considera un mate score,
# no una evaluacion de centipawns. Margen de 1000 plies - muy por encima
# de cualquier profundidad practica de esta fase.
MATE_THRESHOLD = MATE_SCORE - 1000
INFINITY = MATE_SCORE + 1000  # cota inicial de alpha/beta, estrictamente mayor que cualquier mate score


def is_mate_score(score: int) -> bool:
    return abs(score) > MATE_THRESHOLD


def mate_distance_plies_from_score(score: int) -> int | None:
    """Distancia de mate en plies, con signo (positivo=a favor del lado a
    mover en la RAIZ, negativo=en contra) - misma convencion de signo que
    `positions.parquet.mate_distance_plies` en SEARCH TREE DATASET V1.1.
    None si `score` no es un mate score."""
    if not is_mate_score(score):
        return None
    if score > 0:
        return MATE_SCORE - score
    return -(MATE_SCORE - (-score))


def terminal_score(terminal: TerminalResult, ply_from_root: int) -> int:
    """Score, en la perspectiva del lado a mover en el nodo terminal
    (checkmate siempre es muy malo para quien tiene que mover ahi -
    stalemate/material insuficiente/repeticion/50 movimientos son tablas,
    score 0 independiente de la perspectiva)."""
    if terminal.status == TerminalStatus.CHECKMATE:
        return -(MATE_SCORE - ply_from_root)
    return 0


# --- Normalizacion de mate scores para la Transposition Table (Fase III) ---
#
# Los scores normales (MATE_SCORE - ply_from_root) son relativos a la RAIZ
# de una busqueda en particular. Si se guardan tal cual en la TT, un hit
# posterior desde OTRA busqueda (u otro camino dentro de la misma
# busqueda, alcanzando la misma posicion a una profundidad distinta desde
# la raiz) heredaria una distancia de mate INCORRECTA - el bug clasico de
# TT + mate scores.
#
# Solucion estandar (la misma que usan la mayoria de los motores): al
# GUARDAR, convertir el score de "distancia desde la raiz de ESTA
# busqueda" a "distancia desde el NODO donde se guarda" (independiente de
# quien pregunte despues). Al RECUPERAR, convertir de vuelta sumando el
# ply_from_root de quien esta preguntando AHORA.
#
# Derivacion (ver tests/search_v1/test_transposition_table.py para casos
# numericos verificados):
#   favor:   score = MATE_SCORE - N        (N = ply totales desde la raiz)
#            distancia_desde_nodo = N - ply_from_root
#            stored = MATE_SCORE - (N - ply_from_root) = score + ply_from_root
#   contra:  score = -(MATE_SCORE - N)
#            distancia_desde_nodo = N - ply_from_root
#            stored = -(MATE_SCORE - (N - ply_from_root)) = score - ply_from_root
#
# score_from_tt es exactamente la inversa de score_to_tt.


def score_to_tt(score: int, ply_from_root: int) -> int:
    """Convierte un score relativo a la raiz de la busqueda actual a un
    score relativo al NODO donde se va a guardar en la TT (independiente
    de la raiz). No-op para scores que no son de mate (centipawns)."""
    if score > MATE_THRESHOLD:
        return score + ply_from_root
    if score < -MATE_THRESHOLD:
        return score - ply_from_root
    return score


def score_from_tt(stored_score: int, ply_from_root: int) -> int:
    """Inversa de score_to_tt: convierte un score guardado (relativo al
    nodo) a un score relativo a la raiz de la busqueda ACTUAL (que puede
    ser una busqueda distinta a la que lo guardo)."""
    if stored_score > MATE_THRESHOLD:
        return stored_score - ply_from_root
    if stored_score < -MATE_THRESHOLD:
        return stored_score + ply_from_root
    return stored_score
