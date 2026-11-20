"""Constantes compartidas de sequence_policy_v1.

El espacio de action_id es EXACTAMENTE el mismo que tokenization_v1
(seccion 4 del pedido): no se reinterpreta ni se re-escala.
"""

# Espacio bruto de movimientos, identico a tokenization_v1/scripts/tokenizer.py
ACTION_ID_MIN = 0
ACTION_ID_MAX = 5 * 64 * 64 - 1  # 20479
RAW_VOCAB_SIZE = ACTION_ID_MAX - ACTION_ID_MIN + 1  # 20480

# Special tokens reservados FUERA del rango bruto 0-20479 (seccion 3 de la
# Etapa C: PAD/BOS/EOS unicamente, VOCAB_SIZE=20483). No existe token de FEN
# personalizado: el universo V1 excluye por completo esas 38 partidas
# (custom_fen_reserve.json), asi que no hace falta representarlas.
PAD_ID = RAW_VOCAB_SIZE + 0        # 20480
BOS_STD_ID = RAW_VOCAB_SIZE + 1    # 20481 (unico BOS: toda partida V1 arranca en posicion estandar)
EOS_ID = RAW_VOCAB_SIZE + 2        # 20482

FULL_VOCAB_SIZE = RAW_VOCAB_SIZE + 3  # 20483

PLY_BANDS = [
    ("1-10", 1, 11),
    ("11-20", 11, 21),
    ("21-30", 21, 31),
    ("31-50", 31, 51),
    ("51-80", 51, 81),
    ("81-120", 81, 121),
    ("121+", 121, None),
]  # (nombre, inicio_incl, fin_excl_o_None) -- 1-indexado por ply, usado en
   # SPLIT_AUDIT y en el baseline B1 (misma banda en todo el proyecto).

SPECIAL_TOKEN_NAMES = {
    PAD_ID: "PAD",
    BOS_STD_ID: "BOS_STD",
    EOS_ID: "EOS",
}
