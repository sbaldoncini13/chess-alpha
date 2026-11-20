"""Causal self-attention con RoPE, usando scaled_dot_product_attention de
PyTorch (selecciona automaticamente el backend optimizado -- flash /
memory-efficient -- con fallback seguro a la implementacion 'math' si el
backend no esta disponible en el hardware actual)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_policy.rotary import apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, attention_dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(d_model, 3 * num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)
        self.attention_dropout = attention_dropout

    def forward(self, x, cos, sin, key_padding_mask=None, past_kv=None, use_cache=False, rope_offset=0):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, head_dim)

        q = apply_rope(q, cos, sin, offset=rope_offset)
        k = apply_rope(k, cos, sin, offset=rope_offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v) if use_cache else None

        attn_mask = None
        is_causal = past_kv is None  # con KV cache incremental (T=1 nuevo), no hace falta mascara causal explicita
        if key_padding_mask is not None:
            Tk = k.shape[2]
            causal = torch.ones(T, Tk, dtype=torch.bool, device=x.device)
            if is_causal:
                causal = torch.tril(causal, diagonal=Tk - T)
            key_ok = key_padding_mask[:, None, None, :].bool()  # (B,1,1,Tk)
            attn_mask = causal[None, None, :, :] & key_ok
            is_causal = False

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.out_proj(out), new_kv
