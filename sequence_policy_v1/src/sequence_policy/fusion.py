"""Fusion experimental Epoch4 (policy espacial) + baseline secuencial, offline
y read-only. NO modifica Epoch4, NO se integra al motor, NO reemplaza el
score de PVS -- solo un analisis de ranking (seccion 30/31 del pedido).

IMPORTANTE: chess_alpha (el paquete de Epoch4) se importa de forma PEREZOSA,
recien dentro de Epoch4ReadOnly.__init__ -- no a nivel de modulo. Este archivo
se importa como parte normal del paquete sequence_policy incluso en entornos
donde chess_alpha NO esta presente (por ejemplo el bundle de Colab, que
deliberadamente NO incluye Epoch4): si el import fuera a nivel de modulo,
`import sequence_policy.fusion` fallaria siempre en Colab aunque nadie use
Epoch4ReadOnly. La funcionalidad de Epoch4 solo se necesita -- y solo puede
usarse -- dentro del repo completo de chess-alpha, nunca en el bundle."""
import os
import sys

import torch

from sequence_policy.action_codec import move_to_action_id

EPOCH4_CHECKPOINT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models", "selected",
    "policy_value_hybrid_selected_epoch4.pt"
))
EPOCH4_VOCAB = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models", "policy_vocab_lichess_20k.json"
))
EPOCH4_EXPECTED_SHA256 = "917e036269dd4636d07b825afc9310c0eb1c1b92ad74253133b04767e96a28e0"


class Epoch4ReadOnly:
    """Carga Epoch4 SOLO para inferencia (eval mode, no_grad, no se guarda
    nada, no se hace backward nunca). Vocabulario propio de 1868 UCI strings
    -- distinto del espacio de 20480 action_id de sequence_policy_v1; el
    puente entre ambos es el UCI (ver hallazgo de la Etapa A).

    Requiere el repo completo de chess-alpha (chess_alpha.encoding,
    chess_alpha.model_hybrid) -- NO disponible ni utilizable dentro del
    bundle de Colab, que excluye Epoch4 a proposito."""

    def __init__(self):
        import json

        chess_alpha_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
        if chess_alpha_src not in sys.path:
            sys.path.insert(0, chess_alpha_src)
        try:
            from chess_alpha.encoding import board_to_tensor
            from chess_alpha.model_hybrid import PolicyValueNetHybrid
        except ImportError as exc:
            raise ImportError(
                "Epoch4ReadOnly requiere el repo completo de chess-alpha (chess_alpha.encoding, "
                "chess_alpha.model_hybrid), que no esta disponible en este entorno (por ejemplo, "
                "el bundle de Colab lo excluye a proposito). No usar Epoch4ReadOnly fuera del "
                "repo principal."
            ) from exc
        self._board_to_tensor = board_to_tensor

        with open(EPOCH4_VOCAB, encoding="utf-8") as f:
            vocab = json.load(f)
        self.uci_to_idx = {u: i for i, u in enumerate(vocab)}
        checkpoint = torch.load(EPOCH4_CHECKPOINT, map_location="cpu", weights_only=False)
        self.model = PolicyValueNetHybrid(
            policy_vocab_size=checkpoint["policy_vocab_size"],
            channels=checkpoint["channels"],
            num_conv_blocks=checkpoint["num_conv_blocks"],
            num_transformer_blocks=checkpoint["num_transformer_blocks"],
            num_heads=checkpoint["num_heads"],
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict_action_id_probs(self, board) -> dict:
        """Devuelve {action_id: prob} SOLO sobre movimientos legales que
        ademas esten en el vocabulario de 1868 de Epoch4 (softmax renormalizado
        sobre ese subconjunto, igual que NeuralEvaluator.predict en
        chess_alpha/inference.py -- mismo patron, adaptado a la arquitectura
        Hybrid y expresado en action_id en vez de UCI)."""
        tensor = torch.from_numpy(self._board_to_tensor(board)).unsqueeze(0)
        policy_logits, _ = self.model(tensor)

        legal_moves = list(board.legal_moves)
        legal_in_vocab = [m for m in legal_moves if m.uci() in self.uci_to_idx]
        if not legal_in_vocab:
            return {move_to_action_id(m): 1.0 / len(legal_moves) for m in legal_moves}

        idxs = torch.tensor([self.uci_to_idx[m.uci()] for m in legal_in_vocab])
        probs = torch.softmax(policy_logits[0, idxs], dim=0)
        return {move_to_action_id(m): p.item() for m, p in zip(legal_in_vocab, probs)}


def reciprocal_rank_fusion(rank_board: int, rank_sequence: int, k: int) -> float:
    """RRF(move) = 1/(K+rank_board) + 1/(K+rank_sequence). rank=None si el
    movimiento no aparece en el Top-K de esa fuente -- se trata como
    'no aporta' (0), NUNCA se excluye el movimiento del resultado final."""
    score = 0.0
    if rank_board is not None:
        score += 1.0 / (k + rank_board)
    if rank_sequence is not None:
        score += 1.0 / (k + rank_sequence)
    return score
