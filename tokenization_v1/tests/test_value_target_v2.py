import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from value_target import value_target_stm  # noqa: E402


@pytest.mark.parametrize("result_white_pov,stm_white,expected", [
    (1, True, 1),    # blancas ganaron, blancas al turno en esta posicion -> +1 (el que mueve gana)
    (1, False, -1),  # blancas ganaron, negras al turno -> -1 (el que mueve pierde)
    (-1, True, -1),  # negras ganaron, blancas al turno -> -1
    (-1, False, 1),  # negras ganaron, negras al turno -> +1
    (0, True, 0),    # tablas: invariante ante side-to-move
    (0, False, 0),
])
def test_value_target_perspective_matrix(result_white_pov, stm_white, expected):
    assert value_target_stm(result_white_pov, stm_white) == expected


def test_value_target_draw_is_symmetric():
    assert value_target_stm(0, True) == value_target_stm(0, False) == 0


def test_value_target_never_confuses_winner_with_mover():
    """Verificacion cruzada explicita: el ganador de la partida (perspectiva
    blanca fija) y el valor desde side-to-move son cosas distintas y no deben
    coincidir siempre en signo -- solo coinciden cuando el que mueve es
    blancas."""
    # blancas ganaron (+1 white pov): coincide con stm cuando mueve blancas,
    # se invierte cuando mueve negras.
    assert value_target_stm(1, True) == 1
    assert value_target_stm(1, False) == -1
    assert value_target_stm(1, True) != value_target_stm(1, False)
