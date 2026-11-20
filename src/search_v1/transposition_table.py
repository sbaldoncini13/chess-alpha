"""Transposition Table de search_v1 (Fase III).

DISENO: tabla direct-mapped (un unico bucket por indice = key % capacity,
sin chaining) - la implementacion mas simple y auditable posible, la mas
facil de razonar sobre colisiones. El INDICE (key % capacity) decide en
que bucket mirar, pero la CLAVE COMPLETA (key de 64 bits, Position.
zobrist_key) siempre se compara contra la clave guardada antes de aceptar
un hit - dos posiciones distintas pueden compartir indice (colision de
indice) sin nunca producir un falso hit, porque la clave completa
guardada difiere y se detecta explicitamente (contador `collisions`).

POLITICA DE REEMPLAZO (documentada, ver tests/search_v1/
test_transposition_table.py::TestReplacementStrategy):
    1. bucket vacio -> siempre reemplaza
    2. entrada de generacion ANTERIOR (busqueda vieja) -> siempre reemplaza
    3. entre misma generacion: profundidad nueva > profundidad guardada -> reemplaza
    4. profundidad igual: EXACT reemplaza a un bound (LOWER/UPPER), un
       bound NUNCA reemplaza a un EXACT ya guardado a la misma profundidad
    5. cualquier otro caso (profundidad nueva menor, o ambos EXACT a
       igual profundidad) -> NO reemplaza (se conserva la entrada
       existente, mas informativa o igual de informativa)

BOUNDS: EXACT (el score es el valor minimax real), LOWER_BOUND (el score
real es >= al guardado - hubo fail-high/cutoff, beta), UPPER_BOUND (el
score real es <= al guardado - fail-low, no se supero alpha). Nunca se
usa una entrada de profundidad MENOR a la solicitada como si fuera valida
para la profundidad actual (ver `probe`).

CODIFICACION DE PROFUNDIDAD (busqueda normal vs quiescence, Fase IV):
`depth` de busqueda normal (alpha_beta.py) es siempre >= 1. `depth` de
quiescence (quiescence.py) se codifica via `quiescence_depth_key()` -
ver esa funcion para la derivacion completa. Para comparar "cual es mas
profunda/confiable" entre CUALQUIER par de entradas (normal-normal,
quiescence-quiescence o mixto) se usa `_depth_priority()`, NO el valor
crudo de `depth`.

BUG REAL #1 encontrado y corregido en Fase IV (ver PHASE_IV_IMPLEMENTATION_
REPORT.md, seccion de hallazgos): la primera version codificaba la
profundidad de quiescence como `-(qdepth_remaining + 1)` directamente.
Durante las extensiones de jaque (`check_extensions_remaining`, ver
quiescence.py), `qdepth_remaining` sigue decrementando en cada llamada
recursiva y puede volverse MUY negativo (mas alla de -1) - con la formula
original, `qdepth_remaining=-2` producia `depth=1`, **colisionando
exactamente con la codificacion de una entrada de busqueda NORMAL a
profundidad 1**. Una entrada de quiescence calculada durante una
extension de jaque (busqueda restringida solo a evasiones, mucho menos
completa que una busqueda normal de ancho completo) quedaba entonces
indistinguible de - y se reutilizaba como si fuera - un resultado de
busqueda normal completo, corrompiendo el score en posiciones reales
del dataset (detectado en la validacion de Fase IV, position_id
d5c599830621cd23: -530 esperado vs -630 obtenido). `quiescence_depth_key()`
usa un offset fijo, muy por encima de cualquier profundidad de extension
de jaque realista, que garantiza que el valor codificado NUNCA deja de
ser negativo sin importar cuanto se extienda la busqueda de jaques.

BUG REAL #2 encontrado y corregido en Fase IV, DISTINTO del anterior y
descubierto DESPUES de corregirlo (la validacion contra la misma posicion
d5c599830621cd23 seguia fallando, -530 vs -630, con el bug #1 ya
resuelto): la regla "una entrada calculada con mas qdepth_remaining
siempre es valida para una consulta con menos" es correcta para el
PRESUPUESTO DE PROFUNDIDAD en si, pero quiescence.py tambien usa
`qdepth_remaining` (via `qdepth_used = max_qdepth - qdepth_remaining`)
para decidir si se permiten JAQUES TACTICOS no capturantes
(`allow_tactical_checks and qdepth_used < tactical_checks_max_qdepth`) -
una politica de busqueda que se vuelve MAS RESTRICTIVA a medida que
qdepth_remaining DISMINUYE, en la direccion OPUESTA a "mas profundo es
siempre mejor". Una entrada guardada con mas remaining (jaques
permitidos) puede reflejar un mate encontrado SOLO via un jaque tactico
que una consulta posterior con menos remaining (jaques ya prohibidos)
nunca habria explorado - reutilizarla cruza silenciosamente esa frontera
de politica y corrompe el score (mismo caso real: la rama 1.Rd1 del FEN
base, resuelta por interceptar TT una entrada de un nodo con jaques
permitidos). Corregido exigiendo en `probe_for_quiescence()` que
`_quiescence_checks_regime()` coincida entre la entrada guardada
(recuperado su qdepth_remaining original via
`_quiescence_remaining_from_depth()`, inversa de `quiescence_depth_key()`)
y la consulta actual, ademas de la condicion de profundidad ya existente."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Bound(Enum):
    EXACT = auto()
    LOWER_BOUND = auto()
    UPPER_BOUND = auto()


# Offset FIJO, deliberadamente muy por encima de cualquier
# max_check_extensions realista (que tipicamente es de un digito) - ver
# quiescence_depth_key() para por que hace falta ser generoso aca: sin
# este margen, una extension de jaque profunda puede hacer que
# qdepth_remaining se vuelva mas negativo que -_QUIESCENCE_DEPTH_OFFSET,
# lo cual haria que el valor codificado cruce a territorio no-negativo y
# colisione con la escala de profundidad de busqueda normal (>=1) - el
# bug real ya documentado arriba.
_QUIESCENCE_DEPTH_OFFSET = 10_000

# Offset arbitrario, mayor que cualquier max_qdepth practico - garantiza
# que CUALQUIER entrada de busqueda normal (depth>=1) tenga prioridad
# estrictamente mayor que CUALQUIER entrada de quiescence (depth<=-1),
# sin alterar el ORDEN RELATIVO entre entradas normales entre si (el
# offset es constante, la comparacion sigue siendo monotona en depth).
_NORMAL_DEPTH_OFFSET = 1_000_000


def _depth_priority(depth: int) -> int:
    """Prioridad de reemplazo comparable entre profundidades de busqueda
    normal (positivas) y de quiescence (negativas, codificadas). Mayor
    prioridad = entrada mas profunda/confiable = debe preferirse al
    reemplazar."""
    if depth >= 1:
        return depth + _NORMAL_DEPTH_OFFSET
    return -depth - 1  # qdepth_remaining equivalente al momento de guardar


def quiescence_depth_key(qdepth_remaining: int) -> int:
    """Codifica `qdepth_remaining` (puede ser negativo durante
    extensiones de jaque, ver quiescence.py) en un valor de `depth` para
    la TT que NUNCA deja de ser negativo, sin importar cuan larga sea la
    extension - evita la colision con la escala de profundidad normal
    (bug real documentado en el docstring del modulo). Preserva el orden
    relativo: mayor `qdepth_remaining` -> valor codificado mas negativo
    -> mayor prioridad via `_depth_priority()`. Usar SIEMPRE esta funcion
    (nunca calcular el encoding a mano) tanto al guardar (`store`) como
    al pedir profundidad minima (`probe_for_quiescence`)."""
    return -(qdepth_remaining + _QUIESCENCE_DEPTH_OFFSET)


def _quiescence_checks_regime(qdepth_remaining: int, qlimits) -> bool:
    """BUG REAL #2 de Fase IV (ver PHASE_IV_IMPLEMENTATION_REPORT.md):
    replica EXACTAMENTE la formula de quiescence.py que decide si se
    permiten jaques tacticos no capturantes (`allow_checks`):
    `qdepth_used = qlimits.max_qdepth - qdepth_remaining`,
    `allow_checks = qlimits.allow_tactical_checks and qdepth_used <
    qlimits.tactical_checks_max_qdepth`.

    Por que hace falta esto en la TT: `allow_checks` NO es monotono en
    `qdepth_remaining` de la misma forma que la "profundidad restante" -
    mas remaining (menos qdepth_used) SIGNIFICA una politica de busqueda
    MAS PERMISIVA (jaques incluidos), no simplemente "mas profundo dentro
    del mismo tipo de busqueda". La regla general de TT "una entrada
    calculada con mas presupuesto restante siempre sirve para una consulta
    con menos presupuesto" (correcta para busqueda normal, y correcta
    DENTRO de un mismo regimen de quiescence) se rompe exactamente al
    cruzar este limite: una entrada calculada con jaques permitidos
    (remaining grande, ej. 5) puede reportar un mate encontrado SOLO
    gracias a un jaque tactico que la politica de una consulta posterior
    con remaining chico (ej. 1, jaques ya NO permitidos) nunca habria
    considerado - reutilizarla corrompe el score (detectado en la
    validacion de Fase IV con datos reales: posicion tras 1.Rd1 en el FEN
    base '6k1/7p/pQ4p1/1p1qp3/1P2b3/P7/2r5/1R2K3 w - - 2 39', score
    esperado -530, obtenido -630 con TT). Ver `probe_for_quiescence()`."""
    qdepth_used = qlimits.max_qdepth - qdepth_remaining
    return qlimits.allow_tactical_checks and qdepth_used < qlimits.tactical_checks_max_qdepth


def _quiescence_remaining_from_depth(depth: int) -> int:
    """Inversa de `quiescence_depth_key()` - recupera el `qdepth_remaining`
    original con el que se guardo una entrada de quiescence, a partir de
    `entry.depth`. Solo valida para depth<0 (entradas de quiescence)."""
    return -depth - _QUIESCENCE_DEPTH_OFFSET


@dataclass
class TTEntry:
    key: int              # clave COMPLETA de 64 bits (Position.zobrist_key), no solo el indice
    depth: int
    score: int             # YA normalizado para TT (score_to_tt aplicado), no relativo a ninguna raiz
    bound: Bound
    best_move: str | None  # UCI, o None si no hay (posicion terminal/sin jugadas)
    generation: int


# Tamano estimado por entrada: key(8) + depth(~1, guardado como int pero
# acotado) + score(8) + bound(~1) + best_move (string corta, ~5 bytes
# tipicos UCI + overhead de objeto Python) + generation(~1) - estimacion
# GENEROSA en bytes reales de CPython (overhead de objeto), documentada
# como aproximacion, no medicion exacta de memoria real del proceso.
ESTIMATED_BYTES_PER_ENTRY = 120


class TranspositionTable:
    def __init__(self, capacity_entries: int = 1024):
        if capacity_entries < 1:
            raise ValueError("capacity_entries debe ser >= 1")
        self.capacity = capacity_entries
        self._table: list[TTEntry | None] = [None] * capacity_entries
        self.generation = 0

        # Telemetria (Fase III, seccion TELEMETRIA)
        self.probes = 0
        self.hits = 0
        self.exact_hits = 0
        self.lower_hits = 0
        self.upper_hits = 0
        self.stores = 0
        self.replacements = 0
        self.collisions = 0

    def _index(self, key: int) -> int:
        return key % self.capacity

    def new_generation(self) -> None:
        self.generation += 1

    def clear(self) -> None:
        self._table = [None] * self.capacity
        self.generation = 0
        self.probes = self.hits = self.exact_hits = self.lower_hits = self.upper_hits = 0
        self.stores = self.replacements = self.collisions = 0

    def occupancy(self) -> int:
        return sum(1 for e in self._table if e is not None)

    def estimated_bytes(self) -> int:
        return self.capacity * ESTIMATED_BYTES_PER_ENTRY

    def probe(self, key: int, min_depth: int = 0) -> TTEntry | None:
        """Devuelve la entrada SOLO si la clave completa coincide (nunca
        un falso hit por colision de indice) Y su profundidad es >= a
        `min_depth` (nunca se reutiliza una entrada mas superficial que
        lo que la busqueda actual necesita). Registra probes/hits/
        hits-por-bound y colisiones detectadas."""
        self.probes += 1
        idx = self._index(key)
        entry = self._table[idx]
        if entry is None:
            return None
        if entry.key != key:
            self.collisions += 1
            return None
        if entry.depth < min_depth:
            return None
        self.hits += 1
        if entry.bound == Bound.EXACT:
            self.exact_hits += 1
        elif entry.bound == Bound.LOWER_BOUND:
            self.lower_hits += 1
        else:
            self.upper_hits += 1
        return entry

    def probe_for_quiescence(self, key: int, qdepth_remaining: int, qlimits) -> TTEntry | None:
        """Probe especializado de Fase IV para quiescence.py - NO usa
        `probe()` porque las profundidades de busqueda normal (siempre
        >=1) y las de quiescence (codificadas via `quiescence_depth_key()`)
        viven en escalas distintas y no son directamente comparables con
        `entry.depth < min_depth`.

        Regla de aceptacion (documentada, ver tests/search_v1/
        test_quiescence.py::TestQuiescenceTT):
          - una entrada de busqueda NORMAL (entry.depth >= 1) siempre es
            aceptable para quiescence - un resultado de busqueda completa
            (full-width) es siempre al menos tan bueno como cualquier
            resultado de quiescence, nunca se rechaza por ser "insuficiente".
          - una entrada de QUIESCENCE (entry.depth < 0) es aceptable solo
            si (a) fue calculada con qdepth_remaining igual o mayor
            (entry.depth <= quiescence_depth_key(qdepth_remaining)) Y
            (b) fue calculada bajo el MISMO regimen de `allow_checks` que
            la consulta actual (ver BUG REAL #2 abajo).
          - nunca se acepta una entrada de quiescence como si fuera
            equivalente a una busqueda normal completa (la asimetria es
            intencional, no un bug).

        IMPORTANTE (BUG REAL #1 de Fase IV, ver docstring del modulo y
        posicion d5c599830621cd23): `required_q_depth` DEBE calcularse via
        `quiescence_depth_key()`, nunca con la formula cruda
        `-(qdepth_remaining + 1)` - esa formula cruzaba a valores >=0
        durante extensiones de jaque (qdepth_remaining < -1) y colisionaba
        con la escala de profundidad normal.

        IMPORTANTE (BUG REAL #2 de Fase IV): el criterio "mas remaining
        siempre sirve para menos remaining" es INCORRECTO cuando cruza el
        limite de `tactical_checks_max_qdepth` - una entrada calculada con
        jaques tacticos permitidos (regimen mas permisivo) puede reportar
        un score que depende de haber explorado un jaque que una consulta
        posterior, en el regimen SIN jaques, jamas habria considerado. Se
        exige `_quiescence_checks_regime()` identico entre la entrada y la
        consulta ademas de la condicion de profundidad. Ver
        `_quiescence_checks_regime()` para el caso real que expuso esto
        (rama b1d1 del FEN base, -530 esperado vs -630 obtenido)."""
        self.probes += 1
        idx = self._index(key)
        entry = self._table[idx]
        if entry is None:
            return None
        if entry.key != key:
            self.collisions += 1
            return None

        required_q_depth = quiescence_depth_key(qdepth_remaining)
        sufficient = entry.depth >= 1 or entry.depth <= required_q_depth
        if not sufficient:
            return None

        if entry.depth < 0:
            entry_qdepth_remaining = _quiescence_remaining_from_depth(entry.depth)
            if _quiescence_checks_regime(entry_qdepth_remaining, qlimits) != _quiescence_checks_regime(qdepth_remaining, qlimits):
                return None

        self.hits += 1
        if entry.bound == Bound.EXACT:
            self.exact_hits += 1
        elif entry.bound == Bound.LOWER_BOUND:
            self.lower_hits += 1
        else:
            self.upper_hits += 1
        return entry

    def store(self, key: int, depth: int, score: int, bound: Bound, best_move: str | None) -> None:
        idx = self._index(key)
        existing = self._table[idx]
        new_entry = TTEntry(key=key, depth=depth, score=score, bound=bound,
                             best_move=best_move, generation=self.generation)

        if existing is None:
            self._table[idx] = new_entry
            self.stores += 1
            return

        if existing.key == key:
            # misma posicion exacta (re-visitada) - politica de reemplazo
            # normal (puntos 3 y 4 de la politica documentada arriba)
            if self._should_replace_same_generation(existing, new_entry):
                self._table[idx] = new_entry
                self.replacements += 1
            self.stores += 1
            return

        # bucket ocupado por OTRA posicion (colision de indice real)
        if existing.generation < self.generation:
            self._table[idx] = new_entry
            self.replacements += 1
        elif self._should_replace_same_generation(existing, new_entry):
            self._table[idx] = new_entry
            self.replacements += 1
        self.stores += 1

    @staticmethod
    def _should_replace_same_generation(existing: TTEntry, new: TTEntry) -> bool:
        # BUG REAL detectado y corregido en Fase IV (ver PHASE_IV_
        # IMPLEMENTATION_REPORT.md): comparar `new.depth > existing.depth`
        # directamente es correcto para busqueda normal (depth siempre
        # >=1, mas alto = mas profundo = mejor) pero INCORRECTO para
        # entradas de quiescence (depth codificado como negativo via
        # `quiescence_depth_key()` - MAS profundo/confiable es MAS
        # NEGATIVO, es decir NUMERICAMENTE MENOR). Con la comparacion
        # directa, una entrada de quiescence mas profunda NUNCA
        # reemplazaba a una mas superficial ya guardada en el mismo
        # bucket, dejando resultados obsoletos y sub-optimos servidos por
        # transposicion - causaba divergencias de score reales entre la
        # configuracion optimizada (con TT) y la de referencia (sin TT).
        # Se usa _depth_priority() para comparar en una escala unificada.
        new_p, existing_p = _depth_priority(new.depth), _depth_priority(existing.depth)
        if new_p > existing_p:
            return True
        if new_p < existing_p:
            return False
        # profundidad equivalente: EXACT desplaza a un bound, nunca al reves;
        # entre dos EXACT o dos bounds a igual profundidad, se conserva
        # la entrada existente (desempate deterministico: "primero en
        # llegar, se queda" salvo que la nueva sea estrictamente mejor)
        if new.bound == Bound.EXACT and existing.bound != Bound.EXACT:
            return True
        return False
