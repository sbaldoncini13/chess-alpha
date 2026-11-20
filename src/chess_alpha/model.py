import torch
import torch.nn as nn

from chess_alpha.encoding import NUM_PLANES


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PolicyValueNet(nn.Module):
    """Red chica policy+value sobre el tensor 8x8xN de la posición.
    No hace búsqueda ni conoce jugadas legales: eso lo maneja el módulo de
    inferencia (chess_alpha.inference), que filtra/renormaliza sobre el
    vocabulario de jugadas observadas."""

    def __init__(self, policy_vocab_size: int, channels: int = 64, num_blocks: int = 4):
        super().__init__()
        self.stem = ConvBlock(NUM_PLANES, channels)
        self.blocks = nn.Sequential(*[ConvBlock(channels, channels) for _ in range(num_blocks)])

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
        features = self.blocks(self.stem(x))
        policy_logits = self.policy_head(features)
        value = torch.sigmoid(self.value_head(features)).squeeze(-1)
        return policy_logits, value
