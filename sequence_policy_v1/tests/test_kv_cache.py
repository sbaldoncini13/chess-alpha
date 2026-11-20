import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch

from sequence_policy.model import CausalMoveTransformer

VOCAB = 20483
KW = dict(vocab_size=VOCAB, d_model=64, num_layers=2, num_heads=4, head_dim=16,
          ffn_hidden=128, context_length=32, dropout=0.0, attention_dropout=0.0)


def make_model():
    torch.manual_seed(0)
    m = CausalMoveTransformer(**KW)
    m.eval()
    return m


# 22. KV cache vs full-context: logits equivalentes dentro de tolerancia
def test_kv_cache_equivalent_to_full_context():
    m = make_model()
    torch.manual_seed(2)
    seq = torch.randint(0, VOCAB, (1, 10))

    with torch.no_grad():
        full_logits, _ = m(seq)

        # incremental: token a token, reusando KV cache
        past = None
        incremental_logits = []
        for t in range(seq.shape[1]):
            tok = seq[:, t:t + 1]
            logits_t, past = m(tok, past_key_values=past, use_cache=True)
            incremental_logits.append(logits_t)
        incremental_logits = torch.cat(incremental_logits, dim=1)

    diff = (full_logits - incremental_logits).abs().max().item()
    assert diff < 1e-3, f"KV cache diverge del full-context: diff={diff}"


# 23. Secuencia que supera context_length: RoPE cache tiene margen (4x), no debe crashear
def test_sequence_exceeds_context_length_no_crash():
    m = make_model()
    long_len = KW["context_length"] * 2  # supera el context_length nominal
    x = torch.randint(0, VOCAB, (1, long_len))
    with torch.no_grad():
        logits, _ = m(x)
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, long_len, VOCAB)
