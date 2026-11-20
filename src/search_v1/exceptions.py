"""Excepciones propias de search_v1 - envuelven los errores de python-chess
(y algun caso propio) en tipos con nombre estable para que el resto del
paquete/tests no dependa de que tipo de excepcion lanza python-chess
internamente (que puede variar entre versiones). Cada excepcion incluye el
FEN/UCI relevante en el mensaje para que el error sea reproducible sin
tener que re-ejecutar con logging adicional (item 15 del alcance de Fase I:
"errores reproducibles")."""


class SearchV1Error(Exception):
    """Base de todas las excepciones de search_v1."""


class InvalidFENError(SearchV1Error):
    """FEN sintacticamente invalido o que representa un tablero imposible
    (python-chess valida ambas cosas via Board(fen) + board.status())."""

    def __init__(self, fen: str, reason: str):
        self.fen = fen
        self.reason = reason
        super().__init__(f"FEN invalido: {fen!r} - {reason}")


class IllegalMoveError(SearchV1Error):
    """Se intento aplicar un movimiento que no esta en la lista de
    movimientos legales de la posicion actual (distincion pseudo-legal vs
    legal ya resuelta por python-chess: board.legal_moves excluye
    movimientos que dejan al propio rey en jaque)."""

    def __init__(self, uci: str, fen: str):
        self.uci = uci
        self.fen = fen
        super().__init__(f"Movimiento ilegal {uci!r} en la posicion {fen!r}")


class UndoStackEmptyError(SearchV1Error):
    """Se intento deshacer un movimiento sin haber aplicado ninguno (o mas
    deshacer que aplicar) - protege contra un bug de conteo en quien use
    Position.apply/undo, no depende de que python-chess lance IndexError
    internamente (se envuelve explicitamente, ver Position.undo)."""

    def __init__(self):
        super().__init__("No hay movimientos para deshacer (undo stack vacio)")
