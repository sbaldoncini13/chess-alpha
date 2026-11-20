"""Adapter de inferencia READ-ONLY de Sequence Policy V1 FULL (Fase VIII)
para consumo desde `search_v1`. Nunca escribe a `sequence_policy_v1/` ni al
checkpoint - solo importa (lectura) el paquete `sequence_policy` ya
existente en `sequence_policy_v1/src/sequence_policy/`, exactamente el
mismo patron que usa todo este repo para `chess_alpha` via `sys.path`.

NO se copia ningun archivo del paquete `sequence_policy` a una ubicacion
nueva - `sequence_policy_v1/src/sequence_policy/{model,full_checkpoint}.py`
son BYTE-IDENTICOS a las copias del bundle de entrega
`Chess_Alpha_Sequence_Policy_FULL_2286_PC/.../code/sequence_policy/`
(verificado con `diff` antes de escribir este modulo) - es la unica fuente
de verdad, importarla directamente evita cualquier riesgo de divergencia
entre una copia y el original.

`search_v1` NUNCA importa este paquete directamente (mismo aislamiento ya
aplicado a `chess_alpha`/Epoch4) - se construye e inyecta desde el codigo
de test/benchmark que lo use."""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import torch

_SEQUENCE_POLICY_SRC = Path(__file__).resolve().parents[2] / "sequence_policy_v1" / "src"
if str(_SEQUENCE_POLICY_SRC) not in sys.path:
    sys.path.insert(0, str(_SEQUENCE_POLICY_SRC))

from sequence_policy.action_codec import action_id_to_move, move_to_action_id  # noqa: E402
from sequence_policy.constants import BOS_STD_ID  # noqa: E402
from sequence_policy.full_checkpoint import load_full_checkpoint  # noqa: E402
from sequence_policy.legal_mask import legal_action_ids  # noqa: E402
from sequence_policy.model import CausalMoveTransformer  # noqa: E402

CONTEXT_LENGTH_FALLBACK_NOTE = (
    "context_length se lee SIEMPRE del propio checkpoint (ckpt['config']['model']), "
    "nunca se hardcodea - ver docstring de SequencePolicyEvaluator."
)


class SequencePolicyHistoryMismatchError(ValueError):
    """`history_ucis` no reproduce `board` (jugada ilegal en la secuencia,
    o el replay desde la posicion inicial termina en una posicion
    distinta de `board`). Error de INPUT controlado, nunca silencioso -
    quien llama a `.predict()` (el wrapper de reorder de pvs.py) debe
    tratar esto como "Sequence Policy no disponible para este llamado" y
    aplicar la cadena de fallback (Epoch4-only / clasico), NUNCA usar un
    prior calculado sobre una historia que no corresponde al tablero
    real."""


def replay_history_ucis(history_ucis: list[str]) -> chess.Board:
    """Reproduce `history_ucis` desde `chess.Board()` (posicion inicial
    estandar - V1 no soporta FEN custom) y devuelve el tablero resultante.
    Lanza `SequencePolicyHistoryMismatchError` ante cualquier UCI mal
    formado o ilegal en su turno correspondiente - nunca deja pasar una
    secuencia parcialmente invalida en silencio."""
    board = chess.Board()
    for uci in history_ucis:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise SequencePolicyHistoryMismatchError(f"UCI invalido en history_ucis: {uci!r}") from exc
        if move not in board.legal_moves:
            raise SequencePolicyHistoryMismatchError(
                f"jugada ilegal en history_ucis: {uci!r} en la posicion {board.fen()!r}"
            )
        board.push(move)
    return board


def _relevant_fen_fields(board: chess.Board) -> tuple:
    """Piezas/turno/enroque/al-paso, tomados del propio `.fen()` del
    tablero (no de atributos crudos como `.ep_square`) - DELIBERADO, ver
    hallazgo real documentado abajo. Excluye halfmove_clock/fullmove_number
    a proposito (mismo criterio que Position.zobrist_key/repetition.py ya
    aplican para "estado relevante de la posicion" en search_v1)."""
    return tuple(board.fen().split(" ")[:4])


def history_matches_board(board: chess.Board, history_ucis: list[str]) -> bool:
    """True si reproducir `history_ucis` desde la posicion inicial termina
    EXACTAMENTE en `board` - comparado via `.fen()` (piezas/turno/enroque/
    al paso), nunca via `.ep_square` crudo. Nunca lanza - jugadas ilegales
    en `history_ucis` cuentan como "no coincide", no como excepcion, para
    que el llamador pueda decidir con un solo chequeo booleano si asi lo
    prefiere.

    BUG REAL encontrado y corregido escribiendo los tests de este modulo:
    comparar `board.ep_square` directamente (el atributo crudo) es
    INCORRECTO - python-chess fija `ep_square` incondicionalmente tras
    CUALQUIER doble avance de peon, pero `.fen()` solo lo muestra si una
    captura al paso es LEGAL en ese momento (no hay peon enemigo
    adyacente). Un tablero reconstruido desde ese FEN (`chess.Board(fen)`,
    exactamente lo que hace `Position(fen)` en search_v1) pierde ese
    `ep_square` "fantasma": `chess.Board(); b.push_uci('e2e4'); b.ep_square`
    es `20` (E3), pero `chess.Board(b.fen()).ep_square` es `None`, aunque
    ambos tableros representan la MISMA posicion real y tienen el MISMO
    `.fen()`. Comparar el atributo crudo producia un
    `SequencePolicyHistoryMismatchError` FALSO POSITIVO ante cualquier
    posicion alcanzada por un doble avance de peon sin captura al paso
    disponible - exactamente el caso mas comun del ajedrez (1.e4, 1.d4,
    etc.). Comparar via `.fen()` evita el problema porque ambos lados
    pasan por la misma normalizacion "legal-only"."""
    try:
        replayed = replay_history_ucis(history_ucis)
    except SequencePolicyHistoryMismatchError:
        return False
    return _relevant_fen_fields(replayed) == _relevant_fen_fields(board)


def build_input_action_ids(history_ucis: list[str], context_length: int) -> list[int]:
    """Funcion PURA (sin modelo, testeable sin cargar el checkpoint de
    321MB) que construye la secuencia de action_id a alimentar al
    Transformer: `[BOS_STD_ID] + action_ids(history_ucis)`, SIN EOS (EOS
    es exclusivo de partidas terminadas, nunca se agrega al predecir la
    proxima jugada de una posicion viva). Si el resultado excede
    `context_length`, trunca por la derecha (conserva los ultimos
    `context_length` tokens) - misma ventana de continuacion usada en
    entrenamiento (stride=context_length-1)."""
    action_ids = [BOS_STD_ID] + [move_to_action_id(chess.Move.from_uci(u)) for u in history_ucis]
    if len(action_ids) > context_length:
        action_ids = action_ids[-context_length:]
    return action_ids


class SequencePolicyEvaluator:
    """Carga el checkpoint FULL de Sequence Policy V1 (Fase VIII) SOLO para
    inferencia (eval mode, sin grad, sin optimizer/scheduler/scaler - no
    hacen falta para inferencia). Camino de carga CANONICO, identico al
    usado por `scripts/run_full_dev_eval.py` (el script que genero los
    reportes dev/test ya verificados de este checkpoint) - no se
    reinventa: se lee `config["model"]` del propio checkpoint y se
    instancia `CausalMoveTransformer` con esos hiperparametros exactos
    (nunca hardcodeados a mano), con dropout forzado a 0 para eval."""

    _CONSTRUCTOR_KEYS = (
        "vocab_size", "d_model", "num_layers", "num_heads", "head_dim",
        "ffn_hidden", "context_length", "dropout", "attention_dropout",
        "tied_embeddings", "rope_base",
    )

    def __init__(self, checkpoint_path):
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_cfg = dict(raw["config"]["model"])
        del raw

        # `model_cfg` puede traer claves extra ajenas al constructor (p.ej.
        # `name` del yaml de config, ej. "BASE") - se filtra explicitamente
        # a las claves reales de CausalMoveTransformer, nunca **model_cfg
        # a ciegas, para no depender de que el checkpoint tenga EXACTAMENTE
        # las mismas claves que el constructor y nada mas.
        eval_cfg = {k: model_cfg[k] for k in self._CONSTRUCTOR_KEYS if k in model_cfg}
        eval_cfg["dropout"] = 0.0
        eval_cfg["attention_dropout"] = 0.0
        self.model = CausalMoveTransformer(**eval_cfg)
        load_full_checkpoint(checkpoint_path, self.model, expected_model_config=model_cfg)
        self.model.eval()

        self.context_length = model_cfg["context_length"]

    @torch.no_grad()
    def predict(self, board: chess.Board, history_ucis: list[str]) -> dict:
        """Devuelve {uci: prob}, legal-masked y renormalizado sobre las
        jugadas legales de `board` UNICAMENTE (nunca sobre el vocabulario
        crudo). `history_ucis` es la secuencia REAL de jugadas UCI desde
        el inicio de la partida (posicion estandar - V1 no soporta FEN
        custom) que llevo hasta `board`; una lista vacia significa "board
        es la posicion inicial".

        ASIMETRIA DELIBERADA respecto de `HybridNeuralEvaluator.predict()`
        (Epoch4, `src/chess_alpha/inference.py`): esta funcion devuelve
        SOLO el dict de probabilidades (Sequence Policy no tiene value
        head), no una tupla `(probs, value)` - no asumir duck-typing
        identico entre ambos evaluadores.

        VALIDACION OBLIGATORIA: `history_ucis` debe reproducir `board`
        exactamente (ver `history_matches_board`) - si no coincide, se
        lanza `SequencePolicyHistoryMismatchError` en vez de calcular un
        prior sobre una historia que no corresponde al tablero real. El
        llamador (el wrapper de reorder de pvs.py) captura esta excepcion
        como cualquier otra falla de Sequence Policy y aplica la cadena
        de fallback (Epoch4-only / clasico) - nunca debe propagar como un
        crash de la busqueda."""
        if not history_matches_board(board, history_ucis):
            raise SequencePolicyHistoryMismatchError(
                f"history_ucis={history_ucis!r} no reproduce el board recibido (fen={board.fen()!r})"
            )

        action_ids = build_input_action_ids(history_ucis, self.context_length)
        input_ids = torch.tensor([action_ids], dtype=torch.long)
        logits, _ = self.model(input_ids)
        next_logits = logits[0, -1, :]  # el ultimo token predice la proxima jugada

        legal_ids = sorted(legal_action_ids(board))
        if not legal_ids:
            return {}
        legal_ids_tensor = torch.tensor(legal_ids, dtype=torch.long)
        probs = torch.softmax(next_logits[legal_ids_tensor], dim=0)
        return {action_id_to_move(aid).uci(): p.item() for aid, p in zip(legal_ids, probs)}
