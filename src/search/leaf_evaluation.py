"""Evaluacion de posiciones hoja, en perspectiva RAIZ (root-relative), no
perspectiva blancas. Esto es deliberadamente distinto de
chess_alpha.search.material_eval / NeuralEvaluator.predict_value (que estan
en perspectiva blancas fija) - ver la discusion de riesgo de perspectiva en
la propuesta. Todo score positivo = bueno para quien inicio la busqueda
(root_color), sin importar de que color sea.

Convencion de escalas:
- material: centipawns (peon=100)
- value crudo del modelo: [0,1], probabilidad de que gane quien tiene el
  turno en esa posicion hoja (mover_color, NO root_color)
- value centrado: [-1,+1] en perspectiva ROOT (+1 = ganada para root)
- value_cp: value centrado * VALUE_SCALE_CP, para poder sumarlo con material
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import chess
import torch

logger = logging.getLogger("leaf_evaluation")

MATE_SCORE_CP = 100_000.0

PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


@dataclass
class LeafEvalConfig:
    mode: str = "material"  # "material" | "value" | "hybrid"
    material_weight: float = 0.7
    value_weight: float = 0.3
    value_scale_cp: float = 600.0
    # bonos opcionales de material, todos apagados por defecto para que la
    # primera comparacion sea material puro segun lo pedido
    use_mobility_bonus: bool = False
    use_check_bonus: bool = False
    use_bishop_pair_bonus: bool = False
    use_passed_pawn_bonus: bool = False
    use_king_safety_bonus: bool = False
    debug: bool = False


def _terminal_score_cp(board: chess.Board, root_color: bool) -> Optional[float]:
    """Devuelve el score en cp (perspectiva root) si la posicion es
    terminal, o None si no lo es (hay que seguir evaluando normalmente)."""
    if board.is_checkmate():
        # a quien le toca mover esta en jaque mate -> perdio
        loser = board.turn
        return -MATE_SCORE_CP if loser == root_color else MATE_SCORE_CP
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    return None


def _raw_material_cp(board: chess.Board) -> tuple[float, float]:
    white_cp = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in PIECE_VALUES_CP.items())
    black_cp = sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in PIECE_VALUES_CP.items())
    return white_cp, black_cp


def _mobility_bonus_cp(board: chess.Board) -> float:
    """+cp por movilidad de blancas, -cp por movilidad de negras (pequeno,
    perspectiva blancas; se convierte a perspectiva root en material_eval_cp)."""
    turn_backup = board.turn
    board.turn = chess.WHITE
    white_mobility = board.legal_moves.count()
    board.turn = chess.BLACK
    black_mobility = board.legal_moves.count()
    board.turn = turn_backup
    return 2.0 * (white_mobility - black_mobility)


def _bishop_pair_bonus_cp(board: chess.Board) -> float:
    white_pair = len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2
    black_pair = len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2
    return 30.0 * (int(white_pair) - int(black_pair))


def _check_bonus_cp(board: chess.Board) -> float:
    if not board.is_check():
        return 0.0
    # quien esta en jaque es board.turn; el bono es para el OTRO lado
    return -15.0 if board.turn == chess.WHITE else 15.0


def material_eval_cp(board: chess.Board, root_color: bool, config: LeafEvalConfig = None) -> float:
    """Material puro (+ bonos opcionales) en centipawns, perspectiva root."""
    terminal = _terminal_score_cp(board, root_color)
    if terminal is not None:
        return terminal

    config = config or LeafEvalConfig()
    white_cp, black_cp = _raw_material_cp(board)
    white_relative = white_cp - black_cp

    if config.use_mobility_bonus:
        white_relative += _mobility_bonus_cp(board)
    if config.use_bishop_pair_bonus:
        white_relative += _bishop_pair_bonus_cp(board)
    if config.use_check_bonus:
        white_relative += _check_bonus_cp(board)
    # use_passed_pawn_bonus / use_king_safety_bonus: reservados, no
    # implementados todavia (se agregan cuando haga falta, no antes)

    root_relative = white_relative if root_color == chess.WHITE else -white_relative

    if config.debug:
        logger.debug(
            "material_eval_cp: turn=%s root_color=%s white_cp=%.0f black_cp=%.0f -> root_relative=%.1f",
            "white" if board.turn == chess.WHITE else "black",
            "white" if root_color == chess.WHITE else "black",
            white_cp, black_cp, root_relative,
        )
    return root_relative


@torch.no_grad()
def value_eval(board: chess.Board, root_color: bool, model, config: LeafEvalConfig = None):
    """Evalua con el value head. Devuelve (value_raw, value_centered):
    - value_raw: [0,1], probabilidad de que gane MOVER_COLOR (board.turn) -
      None en posiciones terminales, donde no tiene sentido.
    - value_centered: [-1,+1], perspectiva ROOT (+1 = ganada para root).
    """
    config = config or LeafEvalConfig()

    if board.is_checkmate():
        loser = board.turn
        centered = -1.0 if loser == root_color else 1.0
        if config.debug:
            logger.debug("value_eval: jaque mate, loser=%s root_color=%s -> centered=%.1f",
                          "white" if loser == chess.WHITE else "black",
                          "white" if root_color == chess.WHITE else "black", centered)
        return None, centered
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5, 0.0

    from chess_alpha.encoding import board_to_tensor
    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
    _, value = model(tensor)
    value_raw = value.item()  # P(gana board.turn), board.turn = mover_color

    mover_color = board.turn
    if mover_color == root_color:
        root_value = value_raw
    else:
        root_value = 1.0 - value_raw
    centered = 2.0 * root_value - 1.0

    if config.debug:
        logger.debug(
            "value_eval: mover=%s root_color=%s value_raw(P mover gana)=%.4f root_value=%.4f centered=%.4f",
            "white" if mover_color == chess.WHITE else "black",
            "white" if root_color == chess.WHITE else "black",
            value_raw, root_value, centered,
        )
    return value_raw, centered


def hybrid_eval_cp(board: chess.Board, root_color: bool, model, config: LeafEvalConfig) -> float:
    """material_weight * material_cp + value_weight * (value_centered * value_scale_cp),
    perspectiva root. Si la posicion es terminal, devuelve directo el score
    de mate/tablas (material y value ya coinciden ahi, no hace falta combinar)."""
    terminal = _terminal_score_cp(board, root_color)
    if terminal is not None:
        return terminal

    material_cp = material_eval_cp(board, root_color, config)
    _, value_centered = value_eval(board, root_color, model, config)
    value_cp = value_centered * config.value_scale_cp

    score = config.material_weight * material_cp + config.value_weight * value_cp
    if config.debug:
        logger.debug(
            "hybrid_eval_cp: material_cp=%.1f value_cp=%.1f (w=%.2f/%.2f, scale=%.0f) -> %.1f",
            material_cp, value_cp, config.material_weight, config.value_weight, config.value_scale_cp, score,
        )
    return score


def material_normalized(board: chess.Board, root_color: bool, material_scale_cp: float = 600.0) -> float:
    """Material puro pasado por tanh(cp/scale), perspectiva root, en
    [-1, 1] - misma escala que value_centered para poder combinarlos sin
    mezclar centipawns crudos con [-1,1] directamente (bloque 3)."""
    material_cp = material_eval_cp(board, root_color)
    # el mate ya viene capeado a +-MATE_SCORE_CP desde material_eval_cp;
    # tanh lo satura correctamente a +-1 sin necesidad de un caso especial
    return math.tanh(material_cp / material_scale_cp)


def evaluate_leaf_for_experiment(board: chess.Board, root_color: bool, exp_config, model=None) -> float:
    """Evalua una hoja segun un SearchExperimentConfig (src/configs/
    block3_experiments.py). Todo en [-1, 1], perspectiva root:
    - material: tanh(material_cp / material_scale_cp)
    - value: value_centered puro
    - hybrid: material_weight*material_norm + value_weight*value_norm
      (ambos terminos ya normalizados a [-1,1] antes de combinarlos)."""
    if exp_config.leaf_mode == "material":
        return material_normalized(board, root_color, exp_config.material_scale_cp)

    if exp_config.leaf_mode == "value":
        if model is None:
            raise ValueError("leaf_mode='value' requiere pasar `model`")
        _, centered = value_eval(board, root_color, model)
        return centered

    if exp_config.leaf_mode == "hybrid":
        if model is None:
            raise ValueError("leaf_mode='hybrid' requiere pasar `model`")
        material_norm = material_normalized(board, root_color, exp_config.material_scale_cp)
        _, value_norm = value_eval(board, root_color, model)
        return exp_config.material_weight * material_norm + exp_config.value_weight * value_norm

    raise ValueError(f"leaf_mode desconocido: {exp_config.leaf_mode!r}")


def evaluate_leaf(board: chess.Board, root_color: bool, config: LeafEvalConfig, model=None) -> float:
    """Dispatcher unico segun config.mode. `model` es obligatorio para
    mode="value"/"hybrid", ignorado para mode="material"."""
    if config.mode == "material":
        return material_eval_cp(board, root_color, config)
    if config.mode == "value":
        if model is None:
            raise ValueError("mode='value' requiere pasar `model`")
        _, centered = value_eval(board, root_color, model, config)
        return centered * config.value_scale_cp
    if config.mode == "hybrid":
        if model is None:
            raise ValueError("mode='hybrid' requiere pasar `model`")
        return hybrid_eval_cp(board, root_color, model, config)
    raise ValueError(f"modo desconocido: {config.mode!r}")


# ---------------------------------------------------------------------------
# Tests unitarios de perspectiva (correr este archivo directo para validar)
# ---------------------------------------------------------------------------

def _load_test_model():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chess_alpha.config import LICHESS_MODEL_PATH
    from chess_alpha.model import PolicyValueNet

    ckpt = torch.load(LICHESS_MODEL_PATH, map_location="cpu", weights_only=True)
    model = PolicyValueNet(
        policy_vocab_size=ckpt["policy_vocab_size"],
        channels=ckpt.get("channels", 64),
        num_blocks=ckpt.get("num_blocks", 4),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_self_tests():
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    cfg = LeafEvalConfig(debug=True)
    model = _load_test_model()

    passed = 0
    failed = 0

    def check(label, cond):
        nonlocal passed, failed
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {label}")
        if cond:
            passed += 1
        else:
            failed += 1

    # 1) posicion claramente ganada para blancas (mucho material de mas),
    #    blancas al turno -> root=blancas debe dar material fuertemente positivo
    won_white_fen = "4k3/8/8/8/8/8/QQQQQQQQ/4K3 w - - 0 1"
    board = chess.Board(won_white_fen)
    m = material_eval_cp(board, chess.WHITE, cfg)
    check(f"ganada blancas, blancas al turno, root=blancas -> material>0 (dio {m:.0f})", m > 5000)

    # 2) misma posicion, pero preguntando desde la perspectiva de negras
    #    (root=negras) -> debe ser fuertemente NEGATIVO (mismo tablero!)
    m2 = material_eval_cp(board, chess.BLACK, cfg)
    check(f"misma posicion, root=negras -> material<0 (dio {m2:.0f})", m2 < -5000)
    check("simetria material: root=blancas y root=negras son opuestos exactos",
          abs(m + m2) < 1e-6)

    # 3) la MISMA posicion de piezas pero con NEGRAS al turno (fen distinto,
    #    solo cambia el "turno"), preguntada con root=negras -> tiene que
    #    seguir dando fuertemente positivo (negras tienen la ventaja Y le
    #    toca mover a negras Y estamos preguntando desde su perspectiva)
    won_white_black_to_move_fen = "4k3/8/8/8/8/8/QQQQQQQQ/4K3 b - - 0 1"
    board_b = chess.Board(won_white_black_to_move_fen)
    m3 = material_eval_cp(board_b, chess.WHITE, cfg)
    check(f"ganada blancas, negras al turno, root=blancas -> sigue dando >0 (dio {m3:.0f})", m3 > 5000)

    # 4) mate en uno: fools mate ya jugado, negras acaban de dar mate a
    #    blancas (blancas en jaque mate)
    fools_mate = chess.Board()
    for move_san in ["f3", "e5", "g4", "Qh4#"]:
        fools_mate.push_san(move_san)
    check("posicion es jaque mate", fools_mate.is_checkmate())
    check("mate: turn es blancas (perdedor)", fools_mate.turn == chess.WHITE)

    score_root_white = material_eval_cp(fools_mate, chess.WHITE, cfg)
    score_root_black = material_eval_cp(fools_mate, chess.BLACK, cfg)
    check(f"mate, root=blancas (perdio) -> score muy negativo (dio {score_root_white:.0f})",
          score_root_white <= -MATE_SCORE_CP)
    check(f"mate, root=negras (gano) -> score muy positivo (dio {score_root_black:.0f})",
          score_root_black >= MATE_SCORE_CP)

    value_raw, centered_white = value_eval(fools_mate, chess.WHITE, model, cfg)
    value_raw2, centered_black = value_eval(fools_mate, chess.BLACK, model, cfg)
    check(f"value en mate, root=blancas (perdio) -> centered=-1 (dio {centered_white})", centered_white == -1.0)
    check(f"value en mate, root=negras (gano) -> centered=+1 (dio {centered_black})", centered_black == 1.0)

    # 5) posicion claramente perdida para root (root=blancas, negras con
    #    mucho mas material)
    lost_white_fen = "4k3/qqqqqqqq/8/8/8/8/8/4K3 w - - 0 1"
    board_lost = chess.Board(lost_white_fen)
    m_lost = material_eval_cp(board_lost, chess.WHITE, cfg)
    check(f"perdida para root=blancas -> material<0 (dio {m_lost:.0f})", m_lost < -5000)

    # 6) posicion equilibrada (inicial) -> material cercano a 0 para
    #    cualquier root, y value cercano a 0.5 sin importar la perspectiva
    #    exacta (no exigimos == 0 porque el value head puede tener un sesgo
    #    de apertura aprendido, solo que no sea extremo)
    start = chess.Board()
    m_start_white = material_eval_cp(start, chess.WHITE, cfg)
    m_start_black = material_eval_cp(start, chess.BLACK, cfg)
    check(f"posicion inicial, material=0 para cualquier root (blancas={m_start_white:.0f}, negras={m_start_black:.0f})",
          m_start_white == 0.0 and m_start_black == 0.0)

    _, centered_start_white = value_eval(start, chess.WHITE, model, cfg)
    check(f"posicion inicial, value centrado no extremo (dio {centered_start_white:+.3f})",
          -0.9 < centered_start_white < 0.9)

    # 7) hybrid: sanity check de que con material_weight=1, value_weight=0
    #    el resultado coincide con material puro
    cfg_pure_material = LeafEvalConfig(mode="hybrid", material_weight=1.0, value_weight=0.0)
    h = hybrid_eval_cp(start, chess.WHITE, model, cfg_pure_material)
    check(f"hybrid con weight material=1/value=0 == material puro (hybrid={h}, material=0)", h == 0.0)

    print(f"\n{passed} pasaron, {failed} fallaron")
    return failed == 0


if __name__ == "__main__":
    ok = run_self_tests()
    raise SystemExit(0 if ok else 1)
