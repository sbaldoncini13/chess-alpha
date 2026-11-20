"""RoPE (Rotary Position Embedding)."""
import torch


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                      device=None, dtype=torch.float32):
    assert head_dim % 2 == 0, "head_dim debe ser par para RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """x: (B, H, T, head_dim). cos/sin: (max_seq_len, head_dim/2). offset:
    posicion inicial (para inferencia incremental con KV cache)."""
    T = x.shape[2]
    cos_t = cos[offset:offset + T][None, None, :, :]
    sin_t = sin[offset:offset + T][None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos_t - x2 * sin_t
    rx2 = x1 * sin_t + x2 * cos_t
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)
