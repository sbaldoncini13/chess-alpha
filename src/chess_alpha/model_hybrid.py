import torch
import torch.nn as nn

from chess_alpha.encoding import NUM_PLANES


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Bloque residual de verdad (conv-bn-relu-conv-bn + skip + relu), a
    diferencia de los ConvBlock de model.py que se apilan sin skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class TransformerBlock(nn.Module):
    """Self-attention estandar sobre las 64 casillas como tokens, para
    capturar relaciones de largo alcance (torre/dama/alfil a lo largo del
    tablero) que a la CNN le cuesta mas ver de entrada."""

    def __init__(self, channels: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PolicyValueNetHybrid(nn.Module):
    """Tronco conv residual (patrones tacticos locales) + bloques de
    self-attention sobre las 64 casillas como tokens (relaciones de largo
    alcance), reusando la misma forma de policy/value head que PolicyValueNet
    para no tener que tocar el resto del pipeline (inference.py, search.py)."""

    def __init__(
        self,
        policy_vocab_size: int,
        channels: int = 256,
        num_conv_blocks: int = 6,
        num_transformer_blocks: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.stem = ConvStem(NUM_PLANES, channels)
        self.conv_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_conv_blocks)])

        self.pos_embedding = nn.Parameter(torch.zeros(1, 64, channels))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(channels, num_heads) for _ in range(num_transformer_blocks)]
        )
        self.transformer_norm = nn.LayerNorm(channels)

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

    def forward(self, x: torch.Tensor):
        features = self.conv_blocks(self.stem(x))  # (B, channels, 8, 8)

        b, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)  # (B, 64, channels)
        tokens = tokens + self.pos_embedding

        for block in self.transformer_blocks:
            tokens = block(tokens)
        tokens = self.transformer_norm(tokens)

        features_out = tokens.transpose(1, 2).reshape(b, c, h, w)

        policy_logits = self.policy_head(features_out)
        value = torch.sigmoid(self.value_head(features_out)).squeeze(-1)
        return policy_logits, value


class PolicyValueNetHybridV2(PolicyValueNetHybrid):
    """Variante EXPERIMENTAL, separada de PolicyValueNetHybrid: agrega una
    conexion residual global alrededor del Transformer, sumando las
    features de la CNN a la salida de atencion en vez de que la atencion
    las reemplace del todo:

        cnn_features = stem + bloques residuales
        attn_features = transformer(cnn_features)
        features_out = cnn_features + attn_features

    Mismos submodulos y nombres de parametros que la V1 (mismos
    stem/conv_blocks/transformer_blocks/heads) - un checkpoint de V1 se
    puede cargar directo con load_state_dict, solo cambia el forward.
    Se mantiene como clase aparte a proposito: cambia el grafo de computo,
    asi que requiere reentrenar (al menos parcialmente) y no es comparable
    1:1 con un checkpoint ya entrenado de la V1."""

    def forward(self, x: torch.Tensor):
        cnn_features = self.conv_blocks(self.stem(x))  # (B, channels, 8, 8)

        b, c, h, w = cnn_features.shape
        tokens = cnn_features.flatten(2).transpose(1, 2)  # (B, 64, channels)
        tokens = tokens + self.pos_embedding

        for block in self.transformer_blocks:
            tokens = block(tokens)
        tokens = self.transformer_norm(tokens)

        attn_features = tokens.transpose(1, 2).reshape(b, c, h, w)
        features_out = cnn_features + attn_features  # residual global CNN + atencion

        policy_logits = self.policy_head(features_out)
        value = torch.sigmoid(self.value_head(features_out)).squeeze(-1)
        return policy_logits, value
