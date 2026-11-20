"""Selector delgado de modo de ordenamiento de search_v1 (Fase VIII) - 4
modos de ablacion A/B/C/D. NO es una rama de ejecucion nueva ni una
abstraccion competidora con `epoch4_provider.py`/`sequence_policy_
provider.py`/`candidate_fusion.py`: `iterative_deepening_search`/`pvs()`
nunca "conocen" el modo como concepto, solo ven 3 configs independientes
(`epoch4_config`, `sequence_policy_config`, `fusion_config`) - el modo D
es exactamente "los dos proveedores habilitados + fusion habilitada", no
un cuarto camino de codigo paralelo.

Este modulo es solo una convencion de nombres + un chequeo de consistencia
para que el codigo de test/benchmark pueda etiquetar una corrida por modo
y verificar que lo que wireo realmente coincide con esa etiqueta."""

from __future__ import annotations

PRIOR_MODE_NONE = "none"              # A - legacy, ordenamiento clasico sin cambios
PRIOR_MODE_EPOCH4 = "epoch4"          # B - solo Epoch4
PRIOR_MODE_SEQUENCE = "sequence"      # C - solo Sequence Policy (root-only)
PRIOR_MODE_EPOCH4_SEQUENCE = "epoch4_sequence"  # D - fusion de ambos

PRIOR_MODES = (PRIOR_MODE_NONE, PRIOR_MODE_EPOCH4, PRIOR_MODE_SEQUENCE, PRIOR_MODE_EPOCH4_SEQUENCE)


def validate_prior_mode_configuration(mode: str, epoch4_config, sequence_config, fusion_config) -> None:
    """Chequeo de consistencia PURO (no ejecuta nada, no tiene efecto en
    la busqueda) - lanza `ValueError` si el `mode` declarado no coincide
    con los flags `.enabled` reales de los 3 configs. Uso previsto:
    codigo de test/benchmark que quiere etiquetar una corrida por modo y
    detectar a tiempo un wiring inconsistente (ej. declarar modo="epoch4"
    pero haber dejado sequence_config.enabled=True por error)."""
    if mode not in PRIOR_MODES:
        raise ValueError(f"mode invalido: {mode!r} - esperado uno de {PRIOR_MODES}")

    epoch4_enabled = epoch4_config is not None and epoch4_config.enabled
    sequence_enabled = sequence_config is not None and sequence_config.enabled
    fusion_enabled = fusion_config is not None and fusion_config.enabled

    expected = {
        PRIOR_MODE_NONE: (False, False),
        PRIOR_MODE_EPOCH4: (True, False),
        PRIOR_MODE_SEQUENCE: (False, True),
        PRIOR_MODE_EPOCH4_SEQUENCE: (True, True),
    }[mode]
    actual = (epoch4_enabled, sequence_enabled)
    if actual != expected:
        raise ValueError(
            f"mode={mode!r} espera (epoch4_enabled, sequence_enabled)={expected!r}, "
            f"pero la configuracion real es {actual!r}"
        )
    if mode == PRIOR_MODE_EPOCH4_SEQUENCE and not fusion_enabled:
        raise ValueError(f"mode={mode!r} requiere fusion_config habilitado")
    if mode != PRIOR_MODE_EPOCH4_SEQUENCE and fusion_enabled:
        raise ValueError(f"mode={mode!r} no deberia tener fusion_config habilitado")
