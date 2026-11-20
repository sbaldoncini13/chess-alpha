import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import torch.nn.functional as F

from sequence_policy.model import CausalMoveTransformer, RMSNorm, SwiGLU

VOCAB = 20483
TINY_KW = dict(vocab_size=VOCAB, d_model=64, num_layers=2, num_heads=4, head_dim=16,
               ffn_hidden=128, context_length=32, dropout=0.0, attention_dropout=0.0)


def make_model():
    torch.manual_seed(0)
    return CausalMoveTransformer(**TINY_KW)


# 1. Shapes de embeddings
def test_embedding_shape():
    m = make_model()
    assert m.token_embedding.weight.shape == (VOCAB, 64)


# 2. Shapes de logits
def test_logits_shape():
    m = make_model()
    x = torch.randint(0, VOCAB, (3, 10))
    logits, _ = m(x)
    assert logits.shape == (3, 10, VOCAB)


# 3+4. Causal mask / un token futuro no altera logits anteriores (TEST CRITICO)
def test_causal_no_future_leakage():
    m = make_model()
    m.eval()
    torch.manual_seed(1)
    x1 = torch.randint(0, VOCAB, (1, 12))
    x2 = x1.clone()
    t = 6
    x2[0, t:] = torch.randint(0, VOCAB, (12 - t,))  # distinto a partir de t (inclusive)
    with torch.no_grad():
        logits1, _ = m(x1)
        logits2, _ = m(x2)
    # los logits que predicen la posicion t (es decir, calculados EN la posicion t-1)
    # y todo lo anterior deben ser identicos: solo dependen de tokens < t.
    diff = (logits1[0, :t] - logits2[0, :t]).abs().max().item()
    assert diff < 1e-4, f"future leakage detectado: diff={diff}"
    # y a partir de la posicion donde cambio el input, los logits SI deben diferir
    diff_after = (logits1[0, t:] - logits2[0, t:]).abs().max().item()
    assert diff_after > 1e-4


# 5+6. PAD no contribuye a loss / loss_mask funciona
def test_pad_ignored_in_loss():
    m = make_model()
    x = torch.randint(0, VOCAB, (2, 8))
    logits, _ = m(x)
    labels = torch.randint(0, VOCAB, (2, 8))
    labels[:, 5:] = -100
    loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1), ignore_index=-100)
    assert torch.isfinite(loss)
    # cambiar los logits en zonas ignoradas no debe cambiar la loss
    logits2 = logits.clone()
    logits2[:, 5:] += 100.0
    loss2 = F.cross_entropy(logits2.view(-1, VOCAB), labels.view(-1), ignore_index=-100)
    assert torch.isclose(loss, loss2, atol=1e-5)


# 7. Tied embeddings comparten realmente pesos
def test_tied_embeddings_share_storage():
    m = make_model()
    assert m.lm_head.weight.data_ptr() == m.token_embedding.weight.data_ptr()
    with torch.no_grad():
        m.token_embedding.weight[0, 0] = 12.34
    assert abs(m.lm_head.weight[0, 0].item() - 12.34) < 1e-4


def test_untied_embeddings_do_not_share():
    kw = dict(TINY_KW)
    kw["tied_embeddings"] = False
    m = CausalMoveTransformer(**kw)
    assert m.lm_head.weight.data_ptr() != m.token_embedding.weight.data_ptr()


# 8. RoPE produce shapes correctas
def test_rope_shapes():
    from sequence_policy.rotary import build_rope_cache, apply_rope
    cos, sin = build_rope_cache(seq_len=32, head_dim=16)
    assert cos.shape == (32, 8)
    x = torch.randn(2, 4, 10, 16)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


# 9. RMSNorm estable
def test_rmsnorm_stable():
    norm = RMSNorm(64)
    x = torch.randn(4, 10, 64) * 1000
    out = norm(x)
    assert torch.isfinite(out).all()


# 10. SwiGLU estable
def test_swiglu_stable():
    ffn = SwiGLU(64, 128)
    x = torch.randn(4, 10, 64) * 1000
    out = ffn(x)
    assert torch.isfinite(out).all()


# 11. Forward sin NaN/Inf
def test_forward_no_nan_inf():
    m = make_model()
    x = torch.randint(0, VOCAB, (2, 16))
    logits, _ = m(x)
    assert torch.isfinite(logits).all()


# 12+13. Backward sin NaN/Inf, gradientes presentes
def test_backward_gradients_present_and_finite():
    m = make_model()
    x = torch.randint(0, VOCAB, (2, 16))
    labels = torch.randint(0, VOCAB, (2, 16))
    logits, _ = m(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))
    loss.backward()
    n_with_grad = 0
    for p in m.parameters():
        if p.grad is not None:
            n_with_grad += 1
            assert torch.isfinite(p.grad).all()
    assert n_with_grad > 0


# 14. Gradient clipping
def test_gradient_clipping():
    m = make_model()
    x = torch.randint(0, VOCAB, (2, 16))
    labels = torch.randint(0, VOCAB, (2, 16))
    logits, _ = m(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1)) * 1e6  # forzar norma grande
    loss.backward()
    total_norm_before = torch.norm(torch.stack([p.grad.norm() for p in m.parameters() if p.grad is not None]))
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
    total_norm_after = torch.norm(torch.stack([p.grad.norm() for p in m.parameters() if p.grad is not None]))
    assert total_norm_after <= 1.0 + 1e-3
    assert total_norm_before > total_norm_after


# 15. FP32 (default)
def test_fp32_forward():
    m = make_model()
    x = torch.randint(0, VOCAB, (2, 8))
    logits, _ = m(x)
    assert logits.dtype == torch.float32


# 21. Salida reproducible en modo eval
def test_eval_mode_deterministic():
    m = make_model()
    m.eval()
    x = torch.randint(0, VOCAB, (2, 8))
    with torch.no_grad():
        l1, _ = m(x)
        l2, _ = m(x)
    assert torch.equal(l1, l2)


# 24. BOS/EOS son ids validos dentro del vocab_size configurado
def test_bos_eos_within_vocab():
    from sequence_policy.constants import BOS_STD_ID, EOS_ID, PAD_ID, FULL_VOCAB_SIZE
    assert FULL_VOCAB_SIZE == 20483 == VOCAB
    assert PAD_ID < VOCAB and BOS_STD_ID < VOCAB and EOS_ID < VOCAB


# 25. Special tokens no colisionan entre si ni con action_id reales
def test_special_tokens_no_collision():
    from sequence_policy.constants import ACTION_ID_MAX, PAD_ID, BOS_STD_ID, EOS_ID
    specials = [PAD_ID, BOS_STD_ID, EOS_ID]
    assert len(set(specials)) == len(specials)
    assert all(s > ACTION_ID_MAX for s in specials)
