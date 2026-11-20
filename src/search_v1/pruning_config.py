"""Configuracion de podas/reducciones experimentales de PVS (Fase V).

Cada tecnica es un toggle INDEPENDIENTE, desactivado por defecto - el
pedido de Fase V exige explicitamente que se habiliten "una por una
mediante configuracion independiente" y que ninguna se apruebe como
configuracion final combinada sin haber demostrado individualmente 0
divergencias tacticas criticas / 0 mates perdidos / 0 defensas
obligatorias omitidas / 0 movimientos ilegales / mejora medible.

`PruningConfig()` (todos los defaults) reproduce PVS SIN ninguna poda -
identico en comportamiento a `null_move_enabled=lmr_enabled=
futility_enabled=razoring_enabled=False`, la configuracion de referencia
para el Gate V-A/V-B."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PruningConfig:
    # --- Null move pruning ---
    null_move_enabled: bool = False
    null_move_min_depth: int = 3     # profundidad minima del nodo para intentarlo
    null_move_reduction: int = 2     # R: profundidad reducida de la sub-busqueda nula

    # --- Late Move Reductions ---
    lmr_enabled: bool = False
    lmr_min_depth: int = 3           # profundidad minima del nodo para reducir
    lmr_min_move_index: int = 4      # a partir de que indice (0-based) de jugada se reduce
    lmr_reduction: int = 1           # R: cuanto se reduce la profundidad de la jugada tardia

    # --- Futility pruning (diseñada, ver docstring de pvs.py sobre por
    # que queda desactivada por defecto y no se fuerza su aprobacion) ---
    futility_enabled: bool = False
    futility_max_depth: int = 2      # solo profundidades bajas (hojas cercanas)
    futility_margin_cp: int = 200    # margen sobre evaluacion estatica

    # --- Razoring (idem futility) ---
    razoring_enabled: bool = False
    razoring_max_depth: int = 1
    razoring_margin_cp: int = 300

    def __post_init__(self):
        if self.null_move_min_depth < 1:
            raise ValueError("null_move_min_depth debe ser >= 1")
        if self.null_move_reduction < 1:
            raise ValueError("null_move_reduction debe ser >= 1")
        if self.lmr_min_depth < 1:
            raise ValueError("lmr_min_depth debe ser >= 1")
        if self.lmr_min_move_index < 1:
            raise ValueError("lmr_min_move_index debe ser >= 1 (la primera jugada nunca se reduce)")
        if self.lmr_reduction < 1:
            raise ValueError("lmr_reduction debe ser >= 1")
        if self.futility_max_depth < 0:
            raise ValueError("futility_max_depth no puede ser negativo")
        if self.futility_margin_cp < 0:
            raise ValueError("futility_margin_cp no puede ser negativo")
        if self.razoring_max_depth < 0:
            raise ValueError("razoring_max_depth no puede ser negativo")
        if self.razoring_margin_cp < 0:
            raise ValueError("razoring_margin_cp no puede ser negativo")

    @property
    def any_enabled(self) -> bool:
        return self.null_move_enabled or self.lmr_enabled or self.futility_enabled or self.razoring_enabled
