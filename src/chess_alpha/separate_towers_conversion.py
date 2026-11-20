"""Logica reusable de conversion Epoch4 (PolicyValueNetHybrid) ->
PolicyValueNetHybridSeparateTowers. Usada tanto por el script de conversion
standalone como por train_separate_towers_microablation.py (transferencia
on-the-fly cuando el checkpoint de entrada todavia es la arquitectura
original). Nunca escribe en el checkpoint fuente, nunca usa strict=False en
silencio."""
import torch

from chess_alpha.model_hybrid import PolicyValueNetHybrid
from chess_alpha.model_hybrid_separate_towers import PolicyValueNetHybridSeparateTowers, count_new_parameters

ARCHITECTURE_NAME = "hybrid_cnn_transformer_separate_towers"

# Posiciones fijas para el chequeo de equivalencia inicial (variedad de
# fase/complejidad).
FIXED_TEST_POSITIONS = (
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqk2r/ppp2ppp/2n1pn2/3p4/1b1P4/2NBPN2/PPP2PPP/R1BQK2R w KQkq - 0 6",
    "rnbqkb1r/pp3ppp/4pn2/2pp4/3P4/2N1PN2/PPP2PPP/R1BQKB1R w KQkq - 0 5",
    "8/8/4k3/8/4K3/8/8/4R3 w - - 0 1",  # final simple torre+rey
    "r4rk1/pp3ppp/2n1bn2/q1pp4/3P4/2N1PNP1/PP2PPBP/R1BQ1RK1 w - - 0 10",  # medio juego cerrado
)


def transfer_checkpoint(ckpt: dict, **tower_kwargs) -> tuple:
    """Construye PolicyValueNetHybridSeparateTowers y transfiere TODO el
    tronco/heads desde el checkpoint original via load_state_dict. Audita
    explicitamente que:
    - missing_keys sea EXACTAMENTE {policy_tower.*, value_tower.*} (las
      claves nuevas, nunca mas ni menos);
    - unexpected_keys este vacio (ninguna clave del checkpoint se descarta);
    - ninguna clave TRANSFERIDA cambio de valor (comparacion tensor a
      tensor contra el checkpoint original, no solo "se cargo sin error").
    Nunca usa strict=False en silencio: si falta o sobra algo distinto de
    lo esperado, levanta una excepcion."""
    new_model = PolicyValueNetHybridSeparateTowers(
        policy_vocab_size=ckpt["policy_vocab_size"], channels=ckpt["channels"],
        num_conv_blocks=ckpt["num_conv_blocks"], num_transformer_blocks=ckpt["num_transformer_blocks"],
        num_heads=ckpt["num_heads"], **tower_kwargs,
    )

    old_state = ckpt["model_state"]
    missing, unexpected = new_model.load_state_dict(old_state, strict=False)

    expected_new_keys = {n for n, _ in new_model.named_parameters() if n.split(".")[0] in ("policy_tower", "value_tower")}
    expected_new_keys |= {n for n, _ in new_model.named_buffers() if n.split(".")[0] in ("policy_tower", "value_tower")}

    if unexpected:
        raise RuntimeError(f"Claves inesperadas al transferir (no deberian existir): {unexpected}")
    if set(missing) != expected_new_keys:
        raise RuntimeError(
            f"missing_keys no coincide EXACTAMENTE con las claves nuevas esperadas.\n"
            f"missing_keys={sorted(missing)}\nesperadas={sorted(expected_new_keys)}\n"
            f"diferencia={sorted(set(missing) ^ expected_new_keys)}"
        )

    new_state = new_model.state_dict()
    unchanged_check = []
    for key in old_state:
        if key in expected_new_keys:
            continue
        equal = torch.equal(old_state[key], new_state[key])
        unchanged_check.append({"key": key, "unchanged": equal})
        if not equal:
            raise RuntimeError(f"La capa transferida {key!r} NO quedo identica al original - abortando.")

    transfer_report = {
        "n_transferred_keys": len(old_state),
        "n_new_keys": len(expected_new_keys),
        "new_keys": sorted(expected_new_keys),
        "all_transferred_keys_unchanged": all(c["unchanged"] for c in unchanged_check),
        "n_keys_verified_unchanged": len(unchanged_check),
    }
    return new_model, transfer_report


def run_equivalence_check(orig_model, new_model, extra_positions=None) -> dict:
    """Compara policy logits/value/top-k entre el modelo original y
    SeparateTowers (torres en cero) sobre FIXED_TEST_POSITIONS (+ extra
    opcionales). Con fc2 de cada torre inicializado en cero, la salida
    deberia ser IDENTICA bit a bit (no solo 'muy parecida')."""
    orig_model.eval()
    new_model.eval()
    positions = list(FIXED_TEST_POSITIONS) + list(extra_positions or [])
    rows = []
    max_policy_diff = 0.0
    max_value_diff = 0.0
    all_top16_match = True
    all_exact_zero_diff = True

    import chess
    from chess_alpha.encoding import board_to_tensor

    with torch.no_grad():
        for fen in positions:
            board = chess.Board(fen)
            tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
            p1, v1 = orig_model(tensor)
            p2, v2 = new_model(tensor)

            policy_diff = (p1 - p2).abs().max().item()
            value_diff = (v1 - v2).abs().max().item()
            k = min(16, p1.shape[-1])
            top_orig = p1[0].topk(k).indices.tolist()
            top_new = p2[0].topk(k).indices.tolist()
            top_match = top_orig == top_new

            max_policy_diff = max(max_policy_diff, policy_diff)
            max_value_diff = max(max_value_diff, value_diff)
            all_top16_match = all_top16_match and top_match
            all_exact_zero_diff = all_exact_zero_diff and (policy_diff == 0.0) and (value_diff == 0.0)

            rows.append({
                "fen": fen, "policy_logit_max_abs_diff": policy_diff, "value_max_abs_diff": value_diff,
                "top16_indices_match": top_match, "top16_orig": top_orig, "top16_new": top_new,
            })

    return {
        "n_positions": len(positions),
        "max_policy_logit_abs_diff": max_policy_diff,
        "max_value_abs_diff": max_value_diff,
        "all_top16_indices_match": all_top16_match,
        "all_exact_zero_diff": all_exact_zero_diff,
        "expected": "diferencia EXACTAMENTE 0.0 (no solo pequeña) - las torres son f(x)=x+0 en la inicializacion",
        "per_position": rows,
    }
