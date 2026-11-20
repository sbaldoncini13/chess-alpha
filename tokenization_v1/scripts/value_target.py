"""Value target (PASO 14): resultado de partida desde la perspectiva del
jugador AL TURNO en cada posicion. Convencion: +1 gana quien mueve, 0 tablas,
-1 pierde quien mueve. Documentada y testeada de forma exhaustiva (ver
tests/test_value_target.py) porque una inversion de signo aca es el error
mas caro y silencioso posible en un dataset de entrenamiento.

`result_code_white_pov` es el mismo valor que v1 ya guarda en
metadata.parquet: {"1-0": 1, "0-1": -1, "1/2-1/2": 0} -- SIEMPRE desde la
perspectiva de blancas, sin importar quien mueve en la posicion.
"""


def value_target_stm(result_code_white_pov: int, side_to_move_is_white: bool) -> int:
    """result_code_white_pov in {-1, 0, 1}. Tablas (0) es invariante ante la
    perspectiva; para +1/-1 se invierte el signo si quien mueve es negras."""
    if result_code_white_pov == 0:
        return 0
    return result_code_white_pov if side_to_move_is_white else -result_code_white_pov
