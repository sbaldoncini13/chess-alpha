"""Definiciones LOCALES, exclusivas del Dataset Epoch 4 - NUNCA modifican ni
reemplazan chess_alpha.board_features.game_phase() (funcion compartida,
usada por Epoch1/2/3 y v1.1, se deja intacta para no afectar nada existente).

Nace de una auditoria real (reports/epoch4_phase_definition_audit.{md,json},
2026-07-26): game_phase() daba solo ~7.4% de posiciones "middlegame" sobre
una muestra representativa de partidas humanas Elo 1800-2400 (blitz/rapid/
classical) contra un objetivo de ~55% - el umbral de material para salir de
"opening" (<=45 puntos combinados) es demasiado alto para partidas cortas de
este rango de Elo, que a menudo terminan sin haber cambiado tantas piezas.

phase_epoch4() (definicion "D" de la auditoria, elegida por el usuario tras
comparar 8 alternativas) da 26.7% apertura / 51.0% medio juego / 22.3% final
sobre la misma muestra - la mas cercana al objetivo 20/55/25 sin forzarlo
artificialmente."""

PHASE_EPOCH4_VERSION = "epoch4_phase_v1_D_ply20_mat20_35"

OPENING_MAX_PLY = 20
ENDGAME_MATERIAL_MAX = 20
ENDGAME_NO_QUEENS_MATERIAL_MAX = 35


def phase_epoch4(ply: int, material_total: int, queens_present: bool) -> str:
    """ply: 1-indexado (jugada 1 = ply 1). material_total: suma de material
    de PIEZAS de ambos colores (ver chess_alpha.board_features.material_score,
    excluye reyes). queens_present: True si queda al menos una dama en el
    tablero (de cualquier color).

    La condicion de apertura se evalua PRIMERO explicitamente, para que una
    posicion temprana con cambios de piezas rapidos (ej. gambito aceptado)
    no se clasifique de inmediato como final."""
    if ply <= OPENING_MAX_PLY:
        phase = "opening"
    elif material_total <= ENDGAME_MATERIAL_MAX or (not queens_present and material_total <= ENDGAME_NO_QUEENS_MATERIAL_MAX):
        phase = "endgame"
    else:
        phase = "middlegame"
    return phase
