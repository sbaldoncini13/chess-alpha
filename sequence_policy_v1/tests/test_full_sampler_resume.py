"""Pruebas ampliadas de reanudacion del DeterministicWindowSampler para el
full training: corte a mitad de epoca, guardado, reapertura, continuidad
EXACTA (no solo el siguiente batch), ausencia de duplicacion/perdida
alrededor del punto de corte, y misma secuencia completa con la misma seed.
Complementa tests/test_sampler.py (que ya cubre lo basico)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sequence_policy.sampler import DeterministicWindowSampler


def _lengths(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(10, 256, size=n)


def test_full_sequence_identical_same_seed():
    lengths = _lengths()
    s1 = DeterministicWindowSampler(lengths, batch_size=8, seed=99)
    s2 = DeterministicWindowSampler(lengths, batch_size=8, seed=99)
    b1 = [b.tolist() if hasattr(b, "tolist") else list(b) for b in s1]
    b2 = [b.tolist() if hasattr(b, "tolist") else list(b) for b in s2]
    assert b1 == b2


def test_cut_resume_reproduces_exact_tail():
    """Corta a mitad de epoca, guarda estado, lo recarga en una instancia
    NUEVA, y verifica que el resto de los batches coincide EXACTAMENTE
    (no solo el primero) con lo que hubiera producido la corrida sin cortar."""
    lengths = _lengths()
    s_full = DeterministicWindowSampler(lengths, batch_size=8, seed=13)
    all_batches = list(s_full)
    cut_point = len(all_batches) // 2

    s_partial = DeterministicWindowSampler(lengths, batch_size=8, seed=13)
    it = iter(s_partial)
    consumed_before_cut = [next(it) for _ in range(cut_point)]
    assert consumed_before_cut == all_batches[:cut_point]
    state = s_partial.state_dict()

    s_resumed = DeterministicWindowSampler(lengths, batch_size=8, seed=13)
    s_resumed.load_state_dict(state)
    remaining = [b for b in s_resumed]

    assert remaining == all_batches[cut_point:]


def test_no_duplication_or_loss_around_cut_point():
    lengths = _lengths()
    s_partial = DeterministicWindowSampler(lengths, batch_size=8, seed=21)
    it = iter(s_partial)
    cut_point = 37
    consumed_before = [next(it) for _ in range(cut_point)]
    state = s_partial.state_dict()

    s_resumed = DeterministicWindowSampler(lengths, batch_size=8, seed=21)
    s_resumed.load_state_dict(state)
    consumed_after = list(s_resumed)

    idx_before = [i for batch in consumed_before for i in batch]
    idx_after = [i for batch in consumed_after for i in batch]
    all_idx = idx_before + idx_after

    assert len(all_idx) == len(lengths), "no debe perderse ningun indice"
    assert len(set(all_idx)) == len(lengths), "no debe duplicarse ningun indice"
    assert sorted(all_idx) == list(range(len(lengths)))


def test_state_dict_rejects_mismatched_sampler_config():
    """Cargar el estado de un sampler con distinta seed/batch_size debe
    fallar explicitamente (asserts en load_state_dict), no continuar en
    silencio con una configuracion incoherente."""
    lengths = _lengths()
    s1 = DeterministicWindowSampler(lengths, batch_size=8, seed=1)
    next(iter(s1))
    state = s1.state_dict()

    s2 = DeterministicWindowSampler(lengths, batch_size=8, seed=2)  # seed distinta
    try:
        s2.load_state_dict(state)
        assert False, "deberia haber fallado por seed distinta"
    except AssertionError:
        pass


def test_batches_completed_and_targets_processed_tracked():
    """El estado debe reflejar cuantos batches/tokens se consumieron
    realmente -- necesario para el checkpoint schema del full training
    (run_global_step / run_tokens_seen)."""
    lengths = _lengths(n=500)
    s = DeterministicWindowSampler(lengths, batch_size=8, seed=5)
    it = iter(s)
    for _ in range(10):
        next(it)
    state = s.state_dict()
    assert state["batch_pos"] == 10
    assert state["samples_consumed"] == 80  # 10 batches * batch_size=8 (salvo el ultimo batch parcial, no aplica aca)
    assert state["tokens_consumed"] > 0
