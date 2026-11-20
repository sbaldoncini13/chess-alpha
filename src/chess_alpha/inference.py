import chess
import torch

from chess_alpha.config import SUPERVISED_MODEL_PATH, POLICY_VOCAB_PATH
from chess_alpha.dataset import load_policy_vocab
from chess_alpha.encoding import board_to_tensor
from chess_alpha.model import PolicyValueNet
from chess_alpha.model_hybrid import PolicyValueNetHybrid
from chess_alpha.model_hybrid_relative2d import ARCHITECTURE_NAME as RELATIVE2D_ARCHITECTURE_NAME
from chess_alpha.model_hybrid_relative2d import PolicyValueNetHybridRelative2D
from chess_alpha.model_hybrid_separate_towers import PolicyValueNetHybridSeparateTowers
from chess_alpha.separate_towers_conversion import ARCHITECTURE_NAME as SEPARATE_TOWERS_ARCHITECTURE_NAME

HYBRID_ARCHITECTURE_NAME = "hybrid_cnn_transformer"


class NeuralEvaluator:
    """Carga el modelo policy/value entrenado (etapa 6) y expone predicciones
    ya filtradas/renormalizadas sobre las jugadas legales de una posición.
    La red no conoce las reglas del ajedrez: solo vio jugadas de partidas
    humanas, así que puede asignarle probabilidad a un UCI que no sea legal
    en la posición actual. Por eso `predict` restringe al vocabulario visto
    intersectado con jugadas legales, no a la salida cruda de la red."""

    def __init__(self, model_path=SUPERVISED_MODEL_PATH, vocab_path=POLICY_VOCAB_PATH):
        self.vocab = load_policy_vocab(vocab_path)
        self.move_to_idx = {move: i for i, move in enumerate(self.vocab)}
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model = PolicyValueNet(
            policy_vocab_size=checkpoint["policy_vocab_size"],
            channels=checkpoint.get("channels", 64),
            num_blocks=checkpoint.get("num_blocks", 4),
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, board: chess.Board):
        """Devuelve (probs: dict[uci -> prob], value_white: float en [0,1])."""
        tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
        policy_logits, value = self.model(tensor)
        value_for_mover = value.item()

        legal_moves = list(board.legal_moves)
        legal_in_vocab = [m for m in legal_moves if m.uci() in self.move_to_idx]

        if legal_in_vocab:
            idxs = torch.tensor([self.move_to_idx[m.uci()] for m in legal_in_vocab])
            softmax = torch.softmax(policy_logits[0, idxs], dim=0)
            probs = {m.uci(): p.item() for m, p in zip(legal_in_vocab, softmax)}
        else:
            probs = {m.uci(): 1.0 / len(legal_moves) for m in legal_moves}

        value_white = value_for_mover if board.turn == chess.WHITE else 1.0 - value_for_mover
        return probs, value_white

    def top_k_moves(self, board: chess.Board, k: int = 8):
        probs, _ = self.predict(board)
        ranked = sorted(board.legal_moves, key=lambda m: probs.get(m.uci(), 0.0), reverse=True)
        return ranked[:k]

    @torch.no_grad()
    def predict_value(self, board: chess.Board) -> float:
        """Version liviana de `predict` que se salta el filtrado/softmax de
        la policy (no hace falta para evaluar una posicion) - pensada para
        usarse como funcion de evaluacion de hojas dentro de la busqueda."""
        tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
        _, value = self.model(tensor)
        value_for_mover = value.item()
        return value_for_mover if board.turn == chess.WHITE else 1.0 - value_for_mover


class HybridNeuralEvaluator(NeuralEvaluator):
    """Igual que NeuralEvaluator, pero carga la familia de modelos hibridos
    (CNN residual + transformer) en vez de PolicyValueNet. Reusa predict/
    top_k_moves/predict_value tal cual (interfaz comun, ver
    search/negamax.py y search/leaf_evaluation.py: ambos solo usan
    `evaluator.model`/`evaluator.predict`/`.top_k_moves`, nunca el nombre de
    la clase concreta) - lo unico que cambia es la construccion del modelo,
    elegida por `checkpoint["architecture"]` (metadata), NUNCA por el nombre
    del archivo:

    - "hybrid_cnn_transformer"            -> PolicyValueNetHybrid (original)
    - "hybrid_cnn_transformer_relative2d" -> PolicyValueNetHybridRelative2D

    La carga NUNCA usa strict=False en silencio: se audita explicitamente
    que no falten ni sobren claves antes de aceptar el checkpoint."""

    def __init__(self, model_path, vocab_path=POLICY_VOCAB_PATH):
        self.vocab = load_policy_vocab(vocab_path)
        self.move_to_idx = {move: i for i, move in enumerate(self.vocab)}
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.architecture = checkpoint.get("architecture")

        common_kwargs = dict(
            policy_vocab_size=checkpoint["policy_vocab_size"],
            channels=checkpoint["channels"],
            num_conv_blocks=checkpoint["num_conv_blocks"],
            num_transformer_blocks=checkpoint["num_transformer_blocks"],
            num_heads=checkpoint["num_heads"],
        )

        if self.architecture == HYBRID_ARCHITECTURE_NAME:
            self.model = PolicyValueNetHybrid(**common_kwargs)
        elif self.architecture == RELATIVE2D_ARCHITECTURE_NAME:
            relative_2d_bias = checkpoint.get("relative_2d_bias") or {}
            per_layer_bias = relative_2d_bias.get("per_layer", True)
            self.model = PolicyValueNetHybridRelative2D(per_layer_bias=per_layer_bias, **common_kwargs)
        elif self.architecture == SEPARATE_TOWERS_ARCHITECTURE_NAME:
            tower_cfg = checkpoint.get("separate_towers") or {}
            self.model = PolicyValueNetHybridSeparateTowers(
                tower_hidden_dim=tower_cfg.get("hidden_dim", 128),
                tower_layer_scale_init=tower_cfg.get("layer_scale_init", 1.0),
                tower_pre_norm=tower_cfg.get("pre_norm", True),
                tower_residual=tower_cfg.get("residual", True),
                tower_final_projection_zero_init=tower_cfg.get("final_projection_zero_init", True),
                **common_kwargs,
            )
        else:
            raise ValueError(
                f"Arquitectura de checkpoint no reconocida: {self.architecture!r} (path={model_path!r}). "
                f"Soportadas: {HYBRID_ARCHITECTURE_NAME!r}, {RELATIVE2D_ARCHITECTURE_NAME!r}, "
                f"{SEPARATE_TOWERS_ARCHITECTURE_NAME!r}."
            )

        missing, unexpected = self.model.load_state_dict(checkpoint["model_state"], strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint incompleto/incompatible con arquitectura {self.architecture!r} "
                f"(path={model_path!r}): missing_keys={missing}, unexpected_keys={unexpected}. "
                "No se carga con strict=False en silencio."
            )
        self.model.eval()
