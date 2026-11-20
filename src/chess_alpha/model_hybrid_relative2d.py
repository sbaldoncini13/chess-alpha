"""Rama arquitectonica experimental: PolicyValueNetHybridRelative2D.

Variante aislada de PolicyValueNetHybrid (model_hybrid.py, NO modificado por
este archivo) que agrega relative positional bias 2D a la atencion de cada
bloque Transformer, manteniendo el positional embedding absoluto aprendido
existente. Es la UNICA diferencia arquitectonica: mismo stem, mismos bloques
residuales, mismos canales, misma cantidad de bloques Transformer/heads,
mismos policy/value heads.

Los 64 tokens de atencion corresponden 1:1 a los indices de casilla de
python-chess (board_to_tensor usa planes[:, row, col] = planes[:, rank,
file] con square = rank*8+file; el flatten(2) de (B,C,H,W) a (B,64,C)
preserva ese mismo orden row-major, asi que token t == chess.SQUARES[t]).
Esto permite calcular delta_rank/delta_file directamente desde el indice de
token sin tabla de traduccion adicional.

Convencion de signo: para el par (query=i, key=j),
    delta_rank(i, j) = rank(j) - rank(i)
    delta_file(i, j) = file(j) - file(i)
o sea la diferencia "hacia donde se atiende" desde la casilla que pregunta.
Ej.: a1 (query) -> h8 (key) da delta_rank=+7, delta_file=+7.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_alpha.encoding import NUM_PLANES
from chess_alpha.model_hybrid import ConvStem, ResidualBlock

ARCHITECTURE_NAME = "hybrid_cnn_transformer_relative2d"


def square_rank_file():
    """rank[i], file[i] para token/casilla i en 0..63 (rank=i//8, file=i%8)."""
    squares = torch.arange(64, dtype=torch.long)
    return squares // 8, squares % 8


def build_relative_indices(max_rank_delta: int = 7, max_file_delta: int = 7):
    """Devuelve (index_rank, index_file), cada uno (64, 64) long, listos para
    indexar una tabla de bias de forma (heads, 2*max_rank_delta+1,
    2*max_file_delta+1). index_rank[i, j] = delta_rank(i, j) + max_rank_delta."""
    rank, file = square_rank_file()
    delta_rank = rank.view(64, 1) - rank.view(1, 64)  # rank(i) - rank(j)... ver nota abajo
    delta_file = file.view(64, 1) - file.view(1, 64)
    # Queremos delta = key - query, y rank.view(64,1) es la dimension "query"
    # (fila) mientras rank.view(1,64) es "key" (columna) en la matriz 64x64.
    # Con la resta de arriba tenemos query-key; invertimos el signo para
    # obtener key-query = delta_rank(i,j) segun la convencion documentada.
    delta_rank = -delta_rank
    delta_file = -delta_file
    index_rank = delta_rank + max_rank_delta
    index_file = delta_file + max_file_delta
    return index_rank, index_file


class RelativePositionBias2D(nn.Module):
    """Tabla entrenable de bias relativo por attention head, indexada por
    (delta_rank, delta_file). Inicializada en cero de forma obligatoria
    (seccion 4 del spec): con la tabla en cero, sumarla a los logits de
    atencion no cambia nada."""

    def __init__(self, num_heads: int, max_rank_delta: int = 7, max_file_delta: int = 7):
        super().__init__()
        self.num_heads = num_heads
        self.max_rank_delta = max_rank_delta
        self.max_file_delta = max_file_delta
        num_rank = 2 * max_rank_delta + 1
        num_file = 2 * max_file_delta + 1
        self.table = nn.Parameter(torch.zeros(num_heads, num_rank, num_file))
        index_rank, index_file = build_relative_indices(max_rank_delta, max_file_delta)
        # persistent=False: son indices derivados (no pesos entrenados), no
        # deben viajar en el checkpoint ni afectar el reporte de transferencia.
        self.register_buffer("index_rank", index_rank, persistent=False)
        self.register_buffer("index_file", index_file, persistent=False)

    def forward(self) -> torch.Tensor:
        """(num_heads, 64, 64)."""
        return self.table[:, self.index_rank, self.index_file]


class MultiheadAttentionRelative2D(nn.MultiheadAttention):
    """Subclase de nn.MultiheadAttention que reimplementa el forward a mano
    para poder sumar un bias relativo 2D por head ANTES del softmax (la API
    publica de nn.MultiheadAttention no permite un bias por head que no sea
    un attn_mask compartido). Conserva exactamente los mismos parametros
    (in_proj_weight, in_proj_bias, out_proj.weight/bias) que la clase base,
    por lo que un state_dict entrenado con nn.MultiheadAttention estandar
    (el TransformerBlock original de model_hybrid.py) se carga sin remapear
    nombres via load_state_dict(strict=False)."""

    def forward(self, query, key, value, relative_bias: torch.Tensor = None, attn_mask: torch.Tensor = None):
        # batch_first=True: query/key/value son (B, N, E).
        B, N, E = query.shape
        H = self.num_heads
        head_dim = E // H

        w = self.in_proj_weight
        b = self.in_proj_bias
        q = F.linear(query, w[:E], b[:E] if b is not None else None)
        k = F.linear(key, w[E:2 * E], b[E:2 * E] if b is not None else None)
        v = F.linear(value, w[2 * E:], b[2 * E:] if b is not None else None)

        q = q.view(B, N, H, head_dim).transpose(1, 2)  # (B, H, N, hd)
        k = k.view(B, N, H, head_dim).transpose(1, 2)
        v = v.view(B, N, H, head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)

        if relative_bias is not None:
            attn_logits = attn_logits + relative_bias.unsqueeze(0).to(dtype=attn_logits.dtype)
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask

        attn_weights = torch.softmax(attn_logits, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        out = torch.matmul(attn_weights, v)  # (B, H, N, hd)
        out = out.transpose(1, 2).reshape(B, N, E)
        out = self.out_proj(out)
        return out, None


class TransformerBlockRelative2D(nn.Module):
    """Igual a TransformerBlock (model_hybrid.py) en norm1/attn/norm2/mlp -
    mismo pre-norm, misma activacion GELU, mismo mlp_ratio - salvo que
    attn es MultiheadAttentionRelative2D y recibe un bias relativo 2D
    (propio del bloque si per_layer_bias=True, o pasado desde afuera si es
    una tabla compartida entre bloques)."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        per_layer_bias: bool = True,
        max_rank_delta: int = 7,
        max_file_delta: int = 7,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MultiheadAttentionRelative2D(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels),
        )
        self.per_layer_bias = per_layer_bias
        self.relative_bias_2d = (
            RelativePositionBias2D(num_heads, max_rank_delta, max_file_delta) if per_layer_bias else None
        )

    def forward(self, x, shared_bias: torch.Tensor = None):
        bias = self.relative_bias_2d() if self.relative_bias_2d is not None else shared_bias
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, relative_bias=bias)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PolicyValueNetHybridRelative2D(nn.Module):
    """PolicyValueNetHybrid + relative positional bias 2D en la atencion.
    Mismos submodulos/nombres que PolicyValueNetHybrid para stem, conv_blocks,
    pos_embedding, transformer_norm, policy_head y value_head - un checkpoint
    de la V1 se carga directo con load_state_dict(strict=False), los unicos
    parametros nuevos son los relative positional bias 2D (cero al iniciar).

    per_layer_bias=True (recomendado, default): tabla independiente por
    bloque Transformer (~1800 parametros nuevos por bloque). per_layer_bias
    =False: una unica tabla compartida entre los `num_transformer_blocks`
    bloques (~1800 parametros nuevos en total)."""

    def __init__(
        self,
        policy_vocab_size: int,
        channels: int = 256,
        num_conv_blocks: int = 6,
        num_transformer_blocks: int = 4,
        num_heads: int = 8,
        per_layer_bias: bool = True,
        max_rank_delta: int = 7,
        max_file_delta: int = 7,
    ):
        super().__init__()
        self.channels = channels
        self.per_layer_bias = per_layer_bias
        self.max_rank_delta = max_rank_delta
        self.max_file_delta = max_file_delta

        self.stem = ConvStem(NUM_PLANES, channels)
        self.conv_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_conv_blocks)])

        self.pos_embedding = nn.Parameter(torch.zeros(1, 64, channels))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlockRelative2D(
                    channels, num_heads, per_layer_bias=per_layer_bias,
                    max_rank_delta=max_rank_delta, max_file_delta=max_file_delta,
                )
                for _ in range(num_transformer_blocks)
            ]
        )
        self.transformer_norm = nn.LayerNorm(channels)

        self.shared_relative_bias_2d = (
            None if per_layer_bias else RelativePositionBias2D(num_heads, max_rank_delta, max_file_delta)
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, policy_vocab_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def relative_bias_parameter_names(self):
        """Nombres exactos (en state_dict) de los parametros NUEVOS que no
        existen en PolicyValueNetHybrid - usados para auditar la
        transferencia de checkpoint (seccion 5)."""
        if self.per_layer_bias:
            return [f"transformer_blocks.{i}.relative_bias_2d.table" for i in range(len(self.transformer_blocks))]
        return ["shared_relative_bias_2d.table"]

    def forward(self, x: torch.Tensor):
        features = self.conv_blocks(self.stem(x))  # (B, channels, 8, 8)

        b, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)  # (B, 64, channels)
        tokens = tokens + self.pos_embedding

        shared_bias = self.shared_relative_bias_2d() if self.shared_relative_bias_2d is not None else None
        for block in self.transformer_blocks:
            tokens = block(tokens, shared_bias=shared_bias)
        tokens = self.transformer_norm(tokens)

        features_out = tokens.transpose(1, 2).reshape(b, c, h, w)

        policy_logits = self.policy_head(features_out)
        value = torch.sigmoid(self.value_head(features_out)).squeeze(-1)
        return policy_logits, value


# Conjunto FIJO de posiciones para la prueba de equivalencia (seccion 4) y
# para varios tests de la seccion 9. Deliberadamente variado: apertura,
# medio juego, jaque, final de reyes y peones, posicion con muchos
# caballos activos (deltas de caballo ejercitados en la atencion real).
FIXED_TEST_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # posicion inicial
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # apertura temprana
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",  # jaque MATE (fool's mate, Qh4#) - 0 jugadas legales, board.is_game_over()==True
    "r2qk2r/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QK2R w KQkq - 6 8",  # medio juego, muchas piezas activas
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",  # final de peon
    "8/5k2/8/8/3K4/8/8/3N4 w - - 0 1",  # final con caballo (deltas de salto)
    "4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",  # final de torres, casillas de borde
    "r1bqk2r/ppp2ppp/2n2n2/2bpp3/4P3/1B1P1N2/PPP2PPP/RNBQK2R w KQkq - 4 6",  # posicion tactica intermedia
]


def build_relative2d_from_hybrid_checkpoint(ckpt: dict, per_layer_bias: bool = True):
    """Construye PolicyValueNetHybridRelative2D a partir de un checkpoint de
    PolicyValueNetHybrid (dict con model_state/channels/num_conv_blocks/
    num_transformer_blocks/num_heads/policy_vocab_size) y transfiere todos
    los pesos compatibles. Devuelve (model, transfer_report) - NO carga con
    strict=True ciego: audita explicitamente missing/unexpected keys.

    Precondicion validada: el checkpoint debe declarar
    architecture == "hybrid_cnn_transformer" (el checkpoint original, no el
    de esta rama)."""
    if ckpt.get("architecture") != "hybrid_cnn_transformer":
        raise ValueError(
            f"Se esperaba un checkpoint de arquitectura 'hybrid_cnn_transformer', "
            f"recibi {ckpt.get('architecture')!r}. No se transfiere a ciegas."
        )

    model = PolicyValueNetHybridRelative2D(
        policy_vocab_size=ckpt["policy_vocab_size"],
        channels=ckpt["channels"],
        num_conv_blocks=ckpt["num_conv_blocks"],
        num_transformer_blocks=ckpt["num_transformer_blocks"],
        num_heads=ckpt["num_heads"],
        per_layer_bias=per_layer_bias,
    )

    old_state = ckpt["model_state"]
    new_state_keys = set(model.state_dict().keys())
    old_state_keys = set(old_state.keys())

    expected_new_keys = set(model.relative_bias_parameter_names())

    missing_keys, unexpected_keys = model.load_state_dict(old_state, strict=False)
    missing_keys = set(missing_keys)
    unexpected_keys = set(unexpected_keys)

    if unexpected_keys:
        raise ValueError(
            f"load_state_dict con strict=False devolvio unexpected_keys, algo del checkpoint "
            f"original no encontro destino en la arquitectura nueva: {sorted(unexpected_keys)}"
        )
    if missing_keys != expected_new_keys:
        raise ValueError(
            "Los parametros faltantes tras la transferencia no coinciden EXACTAMENTE con los "
            f"nuevos relative positional bias esperados.\n  faltantes reales: {sorted(missing_keys)}\n"
            f"  esperados (solo relative bias): {sorted(expected_new_keys)}\n"
            "Esto indica que alguna capa vieja quedo sin inicializar por error - abortando."
        )

    transferred_keys = old_state_keys & new_state_keys
    n_transferred = sum(old_state[k].numel() for k in transferred_keys)
    n_new = sum(model.state_dict()[k].numel() for k in expected_new_keys)
    n_total_new_model = sum(p.numel() for p in model.state_dict().values())
    n_total_old_model = sum(p.numel() for p in old_state.values())

    for k in expected_new_keys:
        if torch.count_nonzero(model.state_dict()[k]) != 0:
            raise ValueError(f"Parametro nuevo '{k}' no quedo en cero tras la inicializacion - viola seccion 4.")

    report = {
        "source_architecture": "hybrid_cnn_transformer",
        "target_architecture": ARCHITECTURE_NAME,
        "per_layer_bias": per_layer_bias,
        "n_params_source_checkpoint": n_total_old_model,
        "n_params_target_model": n_total_new_model,
        "n_params_transferred": n_transferred,
        "n_params_new": n_new,
        "n_keys_transferred": len(transferred_keys),
        "n_keys_new": len(expected_new_keys),
        "missing_keys_new_params": sorted(expected_new_keys),
        "unexpected_keys": sorted(unexpected_keys),
        "pct_transferred": round(100.0 * n_transferred / n_total_new_model, 6),
        "all_new_params_are_zero": True,
        "no_old_layer_reinitialized": transferred_keys == old_state_keys,
    }
    return model, report
