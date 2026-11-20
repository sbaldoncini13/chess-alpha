import random
import chess

from chess_alpha.search import neural_guided_best_move


def play_self_play_game(evaluator, max_moves=80, temperature_moves=10, depth=2, top_k=8, seed=None):
    """El modelo juega contra sí mismo: movidas de apertura muestreadas de la
    policy (para diversificar partidas, como en self-play tipo AlphaZero) y
    luego jugadas elegidas por búsqueda guiada por la red. Devuelve las
    posiciones jugadas (mismo esquema que positions_sample.parquet) y el
    resultado final, para poder concatenarlas con datos humanos y reentrenar."""
    rng = random.Random(seed)
    board = chess.Board()
    move_records = []

    for ply in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        if ply < temperature_moves:
            probs, _ = evaluator.predict(board)
            legal_moves = list(board.legal_moves)
            weights = [probs.get(m.uci(), 0.0) for m in legal_moves]
            move = rng.choices(legal_moves, weights=weights, k=1)[0] if sum(weights) > 0 else rng.choice(legal_moves)
        else:
            move = neural_guided_best_move(board, evaluator, depth=depth, top_k=top_k)

        move_records.append({
            "move_index": ply,
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "played_move_uci": move.uci(),
        })
        board.push(move)

    result = board.result(claim_draw=True)
    return move_records, result


def run_self_play(evaluator, num_games=10, **game_kwargs):
    """Genera `num_games` partidas de self-play. Devuelve una lista de filas
    con el mismo esquema que positions_sample.parquet (game_id agregado)."""
    rows = []
    for game_id in range(num_games):
        move_records, result = play_self_play_game(evaluator, seed=game_id, **game_kwargs)
        for record in move_records:
            rows.append({"game_id": f"selfplay_{game_id}", "result": result, **record})
    return rows
