"""CausalMoveTransformer: decoder-only causal, RoPE, RMSNorm, SwiGLU,
pre-norm, tied input/output embeddings, KV cache para inferencia incremental.
"""
import torch
import torch.nn as nn

from sequence_policy.attention import CausalSelfAttention
from sequence_policy.rotary import build_rope_cache


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w_up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w_down = nn.Linear(ffn_hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(nn.functional.silu(self.w_gate(x)) * self.w_up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, ffn_hidden, dropout, attention_dropout):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, head_dim, attention_dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, key_padding_mask=None, past_kv=None, use_cache=False, rope_offset=0):
        attn_out, new_kv = self.attn(self.norm1(x), cos, sin, key_padding_mask, past_kv, use_cache, rope_offset)
        x = x + self.resid_dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


class CausalMoveTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int,
                 head_dim: int, ffn_hidden: int, context_length: int,
                 dropout: float = 0.1, attention_dropout: float = 0.05,
                 tied_embeddings: bool = True, rope_base: float = 10000.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, head_dim, ffn_hidden, dropout, attention_dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tied_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        cos, sin = build_rope_cache(context_length * 4, head_dim, base=rope_base)  # margen para KV cache largo
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, exclude_tied_duplicate: bool = True) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if exclude_tied_duplicate:
                if id(p) in seen:
                    continue
                seen.add(id(p))
            total += p.numel()
        return total

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        B, T = input_ids.shape
        rope_offset = 0
        if past_key_values is not None:
            rope_offset = past_key_values[0][0].shape[2]

        x = self.embed_dropout(self.token_embedding(input_ids))
        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = block(x, self.rope_cos, self.rope_sin, attention_mask, past_kv, use_cache, rope_offset)
            if use_cache:
                new_caches.append(new_kv)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_caches
