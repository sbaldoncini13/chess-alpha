import chess
import numpy as np

from chess_alpha.board_features import material_score, game_phase

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
GAME_PHASE_TO_NUM = {"opening": 0.0, "middlegame": 0.5, "endgame": 1.0}

# 6 planos piezas blancas + 6 negras + 1 turno + 4 enroques + 2 casillas
# atacadas + 4 contexto (material, movilidad, jaque, fase) = 23 planos 8x8.
NUM_PLANES = 23


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convierte un tablero en un tensor (NUM_PLANES, 8, 8). No ve "texto":
    cada plano es una matriz binaria o de contexto sobre las 64 casillas."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for color_idx, color in enumerate((chess.WHITE, chess.BLACK)):
        for piece_idx, piece_type in enumerate(PIECE_ORDER):
            plane = color_idx * 6 + piece_idx
            for square in board.pieces(piece_type, color):
                row, col = divmod(square, 8)
                planes[plane, row, col] = 1.0

    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    for square in chess.SQUARES:
        row, col = divmod(square, 8)
        if board.is_attacked_by(chess.WHITE, square):
            planes[17, row, col] = 1.0
        if board.is_attacked_by(chess.BLACK, square):
            planes[18, row, col] = 1.0

    white_material = material_score(board, chess.WHITE)
    black_material = material_score(board, chess.BLACK)
    material_balance = (white_material - black_material) / 39.0  # 39 = material max de un bando
    mobility = min(board.legal_moves.count(), 60) / 60.0
    phase_num = GAME_PHASE_TO_NUM[game_phase(board)]

    planes[19, :, :] = material_balance
    planes[20, :, :] = mobility
    planes[21, :, :] = 1.0 if board.is_check() else 0.0
    planes[22, :, :] = phase_num

    return planes
