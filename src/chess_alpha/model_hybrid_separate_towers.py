"""Variante EXPERIMENTAL de PolicyValueNetHybrid: agrega dos torres livianas
("Separate Towers") post-Transformer, una para policy y otra para value,
antes de que cada rama entre a su head existente. Hipotesis unica de este
experimento ("Especializacion estable de Policy y Value"): dejar que cada
tarea transforme un poco mas las features compartidas del tronco/Transformer
antes de proyectar a policy_logits/value, sin tocar el tronco ni los heads.

NO modifica chess_alpha/model_hybrid.py - clase nueva, separada. No se activa
automaticamente desde ningun import operativo: solo se instancia
explicitamente desde scripts experimentales (train_separate_towers_microablation.py,
chess_alpha/separate_towers_conversion.py).

Se mantienen intactos (mismos nombres/formas de parametro que
PolicyValueNetHybrid, por eso se hereda de ella en vez de reescribir):
- input 23x8x8 (NUM_PLANES, sin cambios);
- stem convolucional;
- 6 bloques residuales (conv_blocks);
- 4 bloques Transformer (transformer_blocks) + transformer_norm;
- dimension 256 (channels);
- vocabulario de policy_vocab_size jugadas (parametro, sin cambios);
- policy_head y value_head existentes (Conv2d -> ... -> Linear), intactos;
- interfaz de salida (policy_logits, value).

Unico cambio: entre transformer_norm y los heads, cada rama (policy/value)
pasa por su propia SeparateTower antes de reshape a mapa espacial 8x8.

Inicializacion identidad SIN doble supresion: fc2 (peso y bias) se
inicializa en CERO, y LayerScale se inicializa en 1.0 (NO en un valor chico
tipo 1e-4) y queda entrenable. Con fc2=0, la salida inicial de la torre ya es
exactamente 0 sin importar el valor de layer_scale (0 * cualquier_escala = 0),
asi que usar layer_scale=1.0 no cambia la equivalencia inicial (sigue siendo
identidad exacta, bit a bit) - pero SI evita que el gradiente de fc2 quede
escalado por un factor adicional de 1e-4 una vez que fc2 empieza a aprender
(ver docstring de SeparateTower.forward): con layer_scale=1e-4 habria DOS
mecanismos de supresion simultaneos (fc2 arrancando en cero + un multiplicador
fijo muy chico), ralentizando el aprendizaje de la torre sin necesidad."""

import torch
import torch.nn as nn

from chess_alpha.model_hybrid import PolicyValueNetHybrid

DEFAULT_LAYER_SCALE_INIT = 1.0  # entrenable, NO 1e-4 - ver docstring del modulo (evita doble supresion con fc2=0)


class SeparateTower(nn.Module):
    """Torre ligera aplicada punto a punto sobre cada uno de los 64 tokens
    (misma convencion de tokens que TransformerBlock: (B, 64, channels)).

    Pre-Norm + MLP angosto (channels -> hidden_dim -> channels) + GELU +
    LayerScale entrenable + conexion residual. Sin dropout (deshabilitado a
    proposito en esta primera version).

    Dinamica de gradientes esperada (verificada en tests/test_separate_towers_dynamic.py):
    - Primer backward (fc2 y layer_scale en su init, fc2=0): el gradiente
      dL/d(fc2) = dL/d(output) * layer_scale (no depende de fc1/norm, fluye
      directo) - con layer_scale=1.0 este gradiente NO esta artificialmente
      suprimido. El gradiente dL/d(fc1)/d(norm)/d(layer_scale) SI depende de
      fc2.weight (via la regla de la cadena hacia atras: d(fc2(h))/dh
      involucra fc2.weight, que es 0), asi que estos reciben gradiente
      cero o casi cero en el primer step - esperado, no un bug.
    - Despues de que fc2 se actualiza una vez (deja de ser exactamente 0):
      fc1, LayerNorm y layer_scale empiezan a recibir gradiente no-nulo."""

    def __init__(
        self,
        channels: int = 256,
        hidden_dim: int = 128,
        layer_scale_init: float = DEFAULT_LAYER_SCALE_INIT,
        pre_norm: bool = True,
        residual: bool = True,
        final_projection_zero_init: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.residual = residual

        self.norm = nn.LayerNorm(channels) if pre_norm else nn.Identity()
        self.fc1 = nn.Linear(channels, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), float(layer_scale_init)))

        if final_projection_zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = h * self.layer_scale
        return x + h if self.residual else h


class PolicyValueNetHybridSeparateTowers(PolicyValueNetHybrid):
    """PolicyValueNetHybrid + torres separadas policy/value post-Transformer.

    Un checkpoint de PolicyValueNetHybrid (mismos policy_vocab_size/
    channels/num_conv_blocks/num_transformer_blocks/num_heads) transfiere
    directo via load_state_dict(strict=False) a esta clase: todas las claves
    de PolicyValueNetHybrid coinciden en nombre y forma (se hereda, no se
    redefine nada existente); solo faltan policy_tower.*/value_tower.* -
    exactamente las claves nuevas esperadas (ver
    chess_alpha/separate_towers_conversion.py, que audita esto explicitamente
    y NUNCA usa strict=False en silencio)."""

    def __init__(
        self,
        policy_vocab_size: int,
        channels: int = 256,
        num_conv_blocks: int = 6,
        num_transformer_blocks: int = 4,
        num_heads: int = 8,
        tower_hidden_dim: int = 128,
        tower_layer_scale_init: float = DEFAULT_LAYER_SCALE_INIT,
        tower_pre_norm: bool = True,
        tower_residual: bool = True,
        tower_final_projection_zero_init: bool = True,
    ):
        super().__init__(
            policy_vocab_size=policy_vocab_size, channels=channels, num_conv_blocks=num_conv_blocks,
            num_transformer_blocks=num_transformer_blocks, num_heads=num_heads,
        )
        tower_kwargs = dict(
            channels=channels, hidden_dim=tower_hidden_dim, layer_scale_init=tower_layer_scale_init,
            pre_norm=tower_pre_norm, residual=tower_residual,
            final_projection_zero_init=tower_final_projection_zero_init,
        )
        self.policy_tower = SeparateTower(**tower_kwargs)
        self.value_tower = SeparateTower(**tower_kwargs)

    def forward(self, x: torch.Tensor):
        features = self.conv_blocks(self.stem(x))  # (B, channels, 8, 8)

        b, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)  # (B, 64, channels)
        tokens = tokens + self.pos_embedding

        for block in self.transformer_blocks:
            tokens = block(tokens)
        tokens = self.transformer_norm(tokens)

        policy_tokens = self.policy_tower(tokens)
        value_tokens = self.value_tower(tokens)

        policy_features = policy_tokens.transpose(1, 2).reshape(b, c, h, w)
        value_features = value_tokens.transpose(1, 2).reshape(b, c, h, w)

        policy_logits = self.policy_head(policy_features)
        value = torch.sigmoid(self.value_head(value_features)).squeeze(-1)
        return policy_logits, value


def count_new_parameters(model: "PolicyValueNetHybridSeparateTowers") -> dict:
    """Cuenta parametros SOLO de las torres nuevas (policy_tower +
    value_tower), para reportar el incremento sin necesidad de instanciar
    tambien la clase original y restar (evita cargar dos modelos)."""
    tower_names = ("policy_tower", "value_tower")
    new_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.split(".")[0] in tower_names
    )
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "new_tower_params": new_params,
        "total_params": total_params,
        "base_params": total_params - new_params,
        "pct_increase": new_params / (total_params - new_params) * 100,
    }
