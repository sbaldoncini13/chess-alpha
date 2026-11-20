"""Evita que Windows suspenda la maquina durante corridas largas (benchmarks
de horas). No-op en cualquier otro SO. Solo afecta suspension/pantalla, no
mantiene la sesion desbloqueada ni interfiere con el uso normal de la PC."""

import contextlib
import platform

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake():
    if platform.system() != "Windows":
        yield
        return

    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
