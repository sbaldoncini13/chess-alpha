"""Legal move mask (PASO 12): representacion compacta (lista de action_id por
posicion, formato ragged igual al patron moves_uint16.bin/game_offsets.npy
de tokv1.0) + expansion a mascara densa 0/1 bajo demanda.

Se evita persistir 20480 floats/bools por posicion (2560 bytes empaquetado)
cuando en promedio solo ~35 de esos 20480 movimientos son legales: la lista
comprimida (~70 bytes/posicion) contiene exactamente la misma informacion.
"""
import numpy as np

from tokenizer import ACTION_ID_MAX, move_to_action_id

VOCAB_SIZE = ACTION_ID_MAX + 1  # 20480


def legal_action_ids(board) -> list:
    """Lista de action_id (tokv1.0) de todos los movimientos legales desde
    `board` en su estado actual. Vacia en posiciones terminales."""
    return [move_to_action_id(m) for m in board.legal_moves]


def expand_legal_mask(action_ids, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Reconstruye el vector denso legal_move_mask[move_id] = 1/0."""
    mask = np.zeros(vocab_size, dtype=np.bool_)
    if len(action_ids):
        mask[np.asarray(action_ids, dtype=np.int64)] = True
    return mask
