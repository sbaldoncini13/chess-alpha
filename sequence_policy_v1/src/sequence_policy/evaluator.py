"""Evaluacion ampliada y reproducible de un checkpoint sobre dev. Nunca toca
test. Read-only respecto del checkpoint (nunca se llama a .backward() ni
optimizer.step()).

Todas las metricas se acumulan PONDERADAS por cantidad de targets validos,
nunca promediando batches con igual peso (bug explicito a evitar, pedido por
el usuario)."""
import math
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F

from sequence_policy.action_codec import action_id_to_move, move_to_action_id
from sequence_policy.constants import RAW_VOCAB_SIZE
from sequence_policy.dataset_reader import SequenceWindowDataset
from sequence_policy.legal_mask import legal_action_ids

IGNORE_INDEX = SequenceWindowDataset.IGNORE_INDEX


def ply_band_of(ply: int) -> str:
    if ply <= 10:
        return "1-10"
    if ply <= 30:
        return "11-30"
    if ply <= 80:
        return "31-80"
    if ply <= 150:
        return "81-150"
    return "151+"


def context_len_band(n_prev_tokens: int) -> str:
    if n_prev_tokens < 16:
        return "corto(<16)"
    if n_prev_tokens < 64:
        return "medio(16-64)"
    return "largo(>=64)"


class RunningSegmentStats:
    """Acumulador ponderado por targets, por segmento."""

    def __init__(self):
        self.n = 0
        self.sum_nll = 0.0
        self.n_top1 = 0
        self.n_top3 = 0
        self.n_top5 = 0
        self.n_top10 = 0
        self.n_top20 = 0
        self.sum_reciprocal_rank = 0.0
        self.ranks = []  # para mean/median rank (se acota el tamano si hace falta)
        self.sum_entropy = 0.0
        self.sum_margin = 0.0
        # ECE: bins de confianza del top-1
        self.ece_bins = np.zeros(10)
        self.ece_bin_correct = np.zeros(10)
        self.ece_bin_conf_sum = np.zeros(10)

    def update(self, nll, ranks, top1_correct, entropy, margin, confidences):
        n = len(nll)
        if n == 0:
            return
        self.n += n
        self.sum_nll += float(nll.sum())
        self.n_top1 += int((ranks == 1).sum())
        self.n_top3 += int((ranks <= 3).sum())
        self.n_top5 += int((ranks <= 5).sum())
        self.n_top10 += int((ranks <= 10).sum())
        self.n_top20 += int((ranks <= 20).sum())
        self.sum_reciprocal_rank += float((1.0 / ranks).sum())
        if len(self.ranks) < 200_000:  # cota de memoria para la mediana exacta
            self.ranks.extend(ranks.tolist())
        self.sum_entropy += float(entropy.sum())
        self.sum_margin += float(margin.sum())

        bin_idx = np.clip((confidences * 10).astype(int), 0, 9)
        for b in range(10):
            sel = bin_idx == b
            self.ece_bins[b] += sel.sum()
            self.ece_bin_correct[b] += (top1_correct & sel).sum()
            self.ece_bin_conf_sum[b] += confidences[sel].sum()

    def summary(self):
        if self.n == 0:
            return {"n_targets": 0}
        mean_nll = self.sum_nll / self.n
        ece = 0.0
        for b in range(10):
            if self.ece_bins[b] > 0:
                acc_b = self.ece_bin_correct[b] / self.ece_bins[b]
                conf_b = self.ece_bin_conf_sum[b] / self.ece_bins[b]
                ece += (self.ece_bins[b] / self.n) * abs(acc_b - conf_b)
        ranks_arr = np.array(self.ranks) if self.ranks else np.array([1])
        return {
            "n_targets": self.n,
            "loss_mean": round(mean_nll, 5),
            "perplexity": round(math.exp(min(mean_nll, 30)), 3),
            "top1": round(self.n_top1 / self.n, 5),
            "top3": round(self.n_top3 / self.n, 5),
            "top5": round(self.n_top5 / self.n, 5),
            "top10": round(self.n_top10 / self.n, 5),
            "top20": round(self.n_top20 / self.n, 5),
            "mrr": round(self.sum_reciprocal_rank / self.n, 5),
            "mean_rank": round(float(ranks_arr.mean()), 3),
            "median_rank": float(np.median(ranks_arr)),
            "ece": round(ece, 5),
            "mean_margin_top1_top2": round(self.sum_margin / self.n, 5),
            "mean_entropy_nats": round(self.sum_entropy / self.n, 5),
        }


@torch.no_grad()
def evaluate_extended(model, dataloader, vocab_size, device="cpu", freq_rare_threshold=None,
                       global_freq_map=None, max_targets=None, progress_every=50,
                       dump_per_target=False, partial_save_every_batches=None, partial_save_callback=None):
    """Recorre dataloader UNA vez, acumulando metricas globales y segmentadas
    (ply, longitud de contexto, movimiento frecuente/raro). No hace backward.

    Si dump_per_target=True, ademas devuelve una lista de registros
    (sample_id, ply[1-indexed], nll, rank, reciprocal_rank) por CADA target
    valido -- pensado para cruzar despues, offline via DuckDB, contra las
    tablas de n-gram (seen/unseen, nivel de soporte) sin cargar esas tablas
    (hasta 18M filas en K=2) enteras en memoria dentro de este loop.

    Si partial_save_every_batches se especifica, cada tantos batches se llama
    a partial_save_callback(dict_parcial) -- pensado como red de seguridad
    ante una corrida larga (horas) que se interrumpe antes de terminar: deja
    resultados intermedios reales en disco, NO implementa resume exacto desde
    ese punto (la proxima corrida arrancaria de cero el dataloader)."""
    model.eval()
    overall = RunningSegmentStats()
    by_ply = {}
    by_ctxlen = {}
    by_freq = {"frecuente": RunningSegmentStats(), "raro": RunningSegmentStats()}
    per_target_records = [] if dump_per_target else None

    t0 = time.time()
    n_batches = 0
    n_targets_seen = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn = batch["attention_mask"].to(device)

        logits, _ = model(input_ids, attention_mask=attn)
        mask = labels != IGNORE_INDEX
        if mask.sum().item() == 0:
            n_batches += 1
            continue

        log_probs = F.log_softmax(logits, dim=-1)
        flat_logits = logits[mask]
        flat_log_probs = log_probs[mask]
        flat_labels = labels[mask]

        nll = -flat_log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
        sorted_idx = flat_logits.argsort(dim=-1, descending=True)
        ranks = (sorted_idx == flat_labels.unsqueeze(1)).float().argmax(dim=1).cpu().numpy() + 1
        top1_correct = ranks == 1

        probs = flat_log_probs.exp()
        entropy = -(probs * flat_log_probs).sum(dim=-1).cpu().numpy()
        top2_vals = flat_logits.topk(2, dim=-1).values
        margin = (top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy()
        top1_conf = probs.gather(1, flat_logits.argmax(dim=-1, keepdim=True)).squeeze(1).cpu().numpy()

        nll_np = nll.cpu().numpy()
        overall.update(nll_np, ranks, top1_correct, entropy, margin, top1_conf)

        # ply por posicion: context_start[i] + j + 1 para cada posicion valida j
        ctx_starts = batch["_context_start"] if "_context_start" in batch else None
        if ctx_starts is not None:
            B, T = labels.shape
            pos_idx = torch.arange(T).unsqueeze(0).expand(B, T)
            ply_tensor = (ctx_starts.unsqueeze(1) + pos_idx + 1)
            plies = ply_tensor[mask.cpu()].numpy()
            ctxlens = pos_idx[mask.cpu()].numpy()  # cantidad de tokens previos disponibles en la ventana

            bands = np.array([ply_band_of(int(p)) for p in plies])
            for band_name in set(bands.tolist()):
                sel = bands == band_name
                if band_name not in by_ply:
                    by_ply[band_name] = RunningSegmentStats()
                by_ply[band_name].update(nll_np[sel], ranks[sel], top1_correct[sel],
                                          entropy[sel], margin[sel], top1_conf[sel])

            ctx_bands = np.array([context_len_band(int(c)) for c in ctxlens])
            for band_name in set(ctx_bands.tolist()):
                sel = ctx_bands == band_name
                if band_name not in by_ctxlen:
                    by_ctxlen[band_name] = RunningSegmentStats()
                by_ctxlen[band_name].update(nll_np[sel], ranks[sel], top1_correct[sel],
                                             entropy[sel], margin[sel], top1_conf[sel])

        if global_freq_map is not None and freq_rare_threshold is not None:
            labels_np = flat_labels.cpu().numpy()
            freqs = np.array([global_freq_map.get(int(t), 0) for t in labels_np])
            is_freq = freqs >= freq_rare_threshold
            by_freq["frecuente"].update(nll_np[is_freq], ranks[is_freq], top1_correct[is_freq],
                                         entropy[is_freq], margin[is_freq], top1_conf[is_freq])
            by_freq["raro"].update(nll_np[~is_freq], ranks[~is_freq], top1_correct[~is_freq],
                                    entropy[~is_freq], margin[~is_freq], top1_conf[~is_freq])

        if per_target_records is not None and ctx_starts is not None and "_sample_id" in batch:
            sample_id_per_window = batch["_sample_id"]
            B, T = labels.shape
            sample_id_tensor = torch.tensor(sample_id_per_window, dtype=torch.long).unsqueeze(1).expand(B, T)
            sample_ids = sample_id_tensor[mask.cpu()].numpy()
            for sid, ply, nll_v, rk in zip(sample_ids.tolist(), plies.tolist(), nll_np.tolist(), ranks.tolist()):
                per_target_records.append((int(sid), int(ply), float(nll_v), int(rk)))

        n_targets_seen += int(mask.sum().item())
        n_batches += 1
        if progress_every and n_batches % progress_every == 0:
            print(f"  batch {n_batches}, targets acumulados {n_targets_seen}, "
                  f"{time.time()-t0:.1f}s")
        if (partial_save_every_batches and partial_save_callback and
                n_batches % partial_save_every_batches == 0):
            partial_out = {
                "PARTIAL": True,
                "n_batches_done_so_far": n_batches,
                "n_targets_done_so_far": n_targets_seen,
                "elapsed_seconds_so_far": round(time.time() - t0, 2),
                "overall_so_far": overall.summary(),
                "by_ply_band_so_far": {k: v.summary() for k, v in by_ply.items()},
                "by_context_length_so_far": {k: v.summary() for k, v in by_ctxlen.items()},
                "by_frequency_so_far": {k: v.summary() for k, v in by_freq.items()},
            }
            partial_save_callback(partial_out)
        if max_targets and n_targets_seen >= max_targets:
            break

    elapsed = time.time() - t0
    out = {
        "overall": overall.summary(),
        "by_ply_band": {k: v.summary() for k, v in by_ply.items()},
        "by_context_length": {k: v.summary() for k, v in by_ctxlen.items()},
        "by_frequency": {k: v.summary() for k, v in by_freq.items()},
        "n_batches": n_batches,
        "elapsed_seconds": round(elapsed, 2),
        "targets_per_second": round(n_targets_seen / elapsed, 1) if elapsed > 0 else None,
    }
    if per_target_records is not None:
        out["per_target_records"] = per_target_records  # (sample_id, ply_1indexed, nll, rank)
    return out


@torch.no_grad()
def audit_legality(model, dataloader, device="cpu", max_games=None, progress_every=200, epoch4=None,
                    epoch4_max_games=None, partial_save_every_games=None, partial_save_callback=None):
    """Legalidad de las predicciones REALES del Transformer (no del baseline
    n-gram) -- read-only, sin backward. Reconstruye el tablero jugada a jugada
    a partir de los propios input_ids de cada ventana (teacher forcing: los
    logits en la posicion j YA son la prediccion del modelo dado el contexto
    real input_ids[0..j], no hace falta un forward autoregresivo aparte).

    Limitado a ventanas con context_start==0 (~99.9% de dev, avg_windows_per_game
    =1.001): son las unicas donde el tablero es reconstruible SOLO con los
    tokens de esa ventana (BOS + movimientos desde el inicio de la partida).
    Las pocas ventanas de continuacion (context_start>0) se saltan y se
    cuentan, no se ignoran en silencio."""
    model.eval()
    n_games_done = 0
    n_windows_skipped_continuation = 0

    n_positions = 0
    n_illegal_top1_before = 0
    n_top10_with_illegal = 0
    mass_on_illegal_total = 0.0
    n_true_covered_before = 0
    n_true_covered_after = 0
    n_illegal_after_mask = 0
    n_zero_legal_in_top10_after = 0

    # integridad del codec (gate obligatorio: se espera 0 en ambos)
    n_invalid_action_id = 0
    n_promotion_moves_seen = 0
    n_action_id_roundtrip_mismatches = 0
    n_promotion_roundtrip_mismatches = 0

    # complementariedad Transformer <-> Epoch4 (solo si epoch4 no es None), calculada
    # en el MISMO recorrido de tablero para no duplicar el costo de reconstruccion.
    n_c = 0
    c_true_in_seq = c_true_in_epoch4 = c_true_in_union = c_true_in_both = 0
    c_only_epoch4 = c_only_seq = c_both_fail = 0
    c_overlap = {1: 0, 3: 0, 5: 0, 10: 0}
    c_jaccard_sum = 0.0

    t0 = time.time()
    for batch in dataloader:
        if max_games is not None and n_games_done >= max_games:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"]
        attn = batch["attention_mask"].to(device)
        ctx_starts = batch["_context_start"]

        logits, _ = model(input_ids, attention_mask=attn)
        logits_cpu = logits.cpu()
        input_ids_cpu = batch["input_ids"]
        B = input_ids_cpu.shape[0]

        for i in range(B):
            if max_games is not None and n_games_done >= max_games:
                break
            if int(ctx_starts[i]) != 0:
                n_windows_skipped_continuation += 1
                continue
            length = int(batch["lengths"][i])
            board = chess.Board()
            for pos in range(length):
                if pos > 0:
                    tok = int(input_ids_cpu[i, pos])
                    if tok >= RAW_VOCAB_SIZE:  # EOS/PAD: el propio token de contexto ya no es una jugada
                        break
                    try:
                        move = action_id_to_move(tok)
                    except ValueError:
                        n_invalid_action_id += 1
                        break  # no se puede seguir reconstruyendo esta partida con un token invalido
                    if move.promotion is not None:
                        n_promotion_moves_seen += 1
                    if move_to_action_id(move) != tok:
                        n_action_id_roundtrip_mismatches += 1
                        if move.promotion is not None:
                            n_promotion_roundtrip_mismatches += 1
                    if move not in board.legal_moves:
                        break  # partida truncada/terminada antes de lo esperado -- no seguir
                    board.push(move)
                true_label = int(labels[i, pos]) if pos < labels.shape[1] else IGNORE_INDEX
                if true_label == IGNORE_INDEX:
                    continue
                legal_ids = legal_action_ids(board)
                n_positions += 1
                pos_logits = logits_cpu[i, pos]
                top1 = int(pos_logits.argmax().item())
                if top1 not in legal_ids:
                    n_illegal_top1_before += 1
                top10 = pos_logits.topk(10).indices.tolist()
                if any(t not in legal_ids for t in top10):
                    n_top10_with_illegal += 1
                probs = F.softmax(pos_logits, dim=-1)
                legal_idx_tensor = torch.tensor(sorted(legal_ids), dtype=torch.long)
                mass_legal = probs[legal_idx_tensor].sum().item()
                mass_on_illegal_total += (1.0 - mass_legal)
                # coverage: aparece el true_label entre las top10 ANTES/DESPUES de mascara
                if true_label in top10:
                    n_true_covered_before += 1
                masked_logits = pos_logits.clone()
                illegal_mask = torch.ones_like(masked_logits, dtype=torch.bool)
                illegal_mask[legal_idx_tensor] = False
                masked_logits[illegal_mask] = float("-inf")
                top10_after = masked_logits.topk(min(10, len(legal_ids))).indices.tolist()
                if any(t not in legal_ids for t in top10_after):
                    n_illegal_after_mask += 1
                if not legal_ids:
                    n_zero_legal_in_top10_after += 1
                elif true_label in top10_after:
                    n_true_covered_after += 1

                epoch4_active = epoch4 is not None and (epoch4_max_games is None or n_games_done < epoch4_max_games)
                if epoch4_active and legal_ids:
                    # Puente UCI: se compara por movimiento real (chess.Move), nunca por
                    # indice numerico crudo compartido -- action_id de sequence_policy_v1 y
                    # el vocabulario de 1868 UCI de Epoch4 son espacios distintos.
                    seq_rank_map = {aid: r + 1 for r, aid in enumerate(top10_after)}
                    epoch4_probs = epoch4.predict_action_id_probs(board)
                    epoch4_ranked = sorted(epoch4_probs.items(), key=lambda kv: -kv[1])[:10]
                    epoch4_rank_map = {aid: r + 1 for r, (aid, _) in enumerate(epoch4_ranked)}
                    seq_set, epoch4_set = set(seq_rank_map), set(epoch4_rank_map)

                    true_in_seq = true_label in seq_rank_map
                    true_in_epoch4 = true_label in epoch4_rank_map
                    n_c += 1
                    c_true_in_seq += true_in_seq
                    c_true_in_epoch4 += true_in_epoch4
                    c_true_in_union += (true_in_seq or true_in_epoch4)
                    c_true_in_both += (true_in_seq and true_in_epoch4)
                    c_only_epoch4 += (true_in_epoch4 and not true_in_seq)
                    c_only_seq += (true_in_seq and not true_in_epoch4)
                    c_both_fail += (not true_in_seq and not true_in_epoch4)
                    for K in (1, 3, 5, 10):
                        s_top = {a for a, r in seq_rank_map.items() if r <= K}
                        e_top = {a for a, r in epoch4_rank_map.items() if r <= K}
                        c_overlap[K] += len(s_top & e_top)
                    if seq_set | epoch4_set:
                        c_jaccard_sum += len(seq_set & epoch4_set) / len(seq_set | epoch4_set)
            n_games_done += 1
            if progress_every and n_games_done % progress_every == 0:
                print(f"  {n_games_done} partidas auditadas, {n_positions} posiciones, "
                      f"{time.time()-t0:.1f}s")
            if (partial_save_every_games and partial_save_callback and
                    n_games_done % partial_save_every_games == 0):
                partial_save_callback({
                    "PARTIAL": True,
                    "n_games_done_so_far": n_games_done,
                    "n_positions_so_far": n_positions,
                    "elapsed_seconds_so_far": round(time.time() - t0, 2),
                    "top1_illegal_rate_before_mask_so_far": round(n_illegal_top1_before / max(n_positions, 1), 5),
                    "n_illegal_after_mask_so_far": n_illegal_after_mask,
                    "n_invalid_action_id_so_far": n_invalid_action_id,
                    "n_action_id_roundtrip_mismatches_so_far": n_action_id_roundtrip_mismatches,
                })

    out = {
        "n_games_audited": n_games_done,
        "n_windows_skipped_continuation": n_windows_skipped_continuation,
        "n_positions": n_positions,
        "n_invalid_action_id_encountered": n_invalid_action_id,
        "gate_0_invalid_action_id": n_invalid_action_id == 0,
        "n_promotion_moves_seen": n_promotion_moves_seen,
        "n_action_id_roundtrip_mismatches": n_action_id_roundtrip_mismatches,
        "n_promotion_roundtrip_mismatches": n_promotion_roundtrip_mismatches,
        "gate_promotions_correct": n_promotion_roundtrip_mismatches == 0,
        "gate_action_id_roundtrip_correct": n_action_id_roundtrip_mismatches == 0,
        "top1_illegal_rate_before_mask": round(n_illegal_top1_before / max(n_positions, 1), 5),
        "top10_contains_illegal_rate_before_mask": round(n_top10_with_illegal / max(n_positions, 1), 5),
        "avg_probability_mass_on_illegal_moves_before_mask": round(mass_on_illegal_total / max(n_positions, 1), 5),
        "true_move_in_top10_before_mask": round(n_true_covered_before / max(n_positions, 1), 5),
        "true_move_in_top10_after_mask": round(n_true_covered_after / max(n_positions, 1), 5),
        "pct_positions_zero_legal_moves": round(100 * n_zero_legal_in_top10_after / max(n_positions, 1), 3),
        "n_illegal_moves_in_top10_after_mask": n_illegal_after_mask,
        "gate_0_illegal_after_mask": n_illegal_after_mask == 0,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    if epoch4 is not None:
        out["epoch4_complementarity"] = {
            "n_positions": n_c,
            "coverage_transformer_alone": round(c_true_in_seq / max(n_c, 1), 5),
            "coverage_epoch4_alone": round(c_true_in_epoch4 / max(n_c, 1), 5),
            "coverage_union": round(c_true_in_union / max(n_c, 1), 5),
            "coverage_both": round(c_true_in_both / max(n_c, 1), 5),
            "coverage_exclusive_transformer": round(c_only_seq / max(n_c, 1), 5),
            "coverage_exclusive_epoch4": round(c_only_epoch4 / max(n_c, 1), 5),
            "both_fail_rate": round(c_both_fail / max(n_c, 1), 5),
            "overlap_at_1_mean": round(c_overlap[1] / max(n_c, 1), 5),
            "overlap_at_3_mean": round(c_overlap[3] / max(n_c, 1), 5),
            "overlap_at_5_mean": round(c_overlap[5] / max(n_c, 1), 5),
            "overlap_at_10_mean": round(c_overlap[10] / max(n_c, 1), 5),
            "jaccard_10_mean": round(c_jaccard_sum / max(n_c, 1), 5),
            "note": "seq = Top-10 REAL del Transformer (post-mascara legal), no el baseline n-gram. "
                    "Puente unico con Epoch4: UCI. Sin RRF operativo.",
        }
    return out
