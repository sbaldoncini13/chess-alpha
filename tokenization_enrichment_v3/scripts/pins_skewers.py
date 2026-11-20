"""#10, #11 -- pins absolutos/relativos, skewers, x-rays. Definiciones:

  absolute_pin  := board.is_pinned(color, square) de python-chess (motor
                    ya validado -- pin formal contra el rey, unica fuente
                    de verdad para este caso, no se reimplementa).
  relative_pin  := via geometry.ray_pieces: un deslizante enemigo, luego
                    EXACTAMENTE una pieza propia, luego OTRA pieza propia
                    de valor MAYOR en la misma linea, sin bloqueos entre
                    medio. La primera pieza (menor valor) esta
                    "relativamente clavada" a la segunda (mayor valor).
                    NO es un pin formal (no restringe legalidad), es una
                    observacion geometrica -- documentado explicitamente.
  x_ray_attack  := deslizante enemigo -> pieza cualquiera -> OTRA pieza
                    cualquiera en la misma linea, sin bloqueos entre medio
                    (observacion geometrica pura, sin importar valores).
  skewer        := caso de x-ray donde la primera pieza es del bando
                    atacado y su valor es >= al de la segunda (o es el
                    rey) -- mover la primera para salvarla expone la
                    segunda.

Todas las conteos se reportan por color (piezas propias que SUFREN el pin/
skewer -- "propio" = color del defensor, no del atacante).
"""
import chess

from geometry import PIECE_VALUES, ray_pieces, slider_directions, sliders_of


def compute_absolute_pins(board: chess.Board, color: bool) -> dict:
    pinned_squares = []
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece.color == color and piece.piece_type != chess.KING and board.is_pinned(color, sq):
            pinned_squares.append(sq)
    return {"absolute_pins": len(pinned_squares), "absolute_pinned_squares": pinned_squares}


def compute_relative_geometry(board: chess.Board, attacker_color: bool) -> dict:
    """Escanea desde cada deslizante de `attacker_color` -- devuelve
    relative_pins, x_ray_attacks y skewers SUFRIDOS por el bando defensor
    (not attacker_color)."""
    defender_color = not attacker_color
    relative_pins = 0
    x_rays = 0
    skewers = 0

    for sq, ptype in sliders_of(board, attacker_color):
        for d in slider_directions(ptype):
            hits = ray_pieces(board, sq, d, max_pieces=2)
            if len(hits) < 2:
                continue
            (sq1, p1), (sq2, p2) = hits
            if p1.color != defender_color:
                continue  # la primera pieza debe ser del defensor para pin/skewer
            x_rays += 1  # geometria pura: dos piezas alineadas tras el deslizante
            v1 = PIECE_VALUES[p1.piece_type]
            v2 = PIECE_VALUES[p2.piece_type]
            # pin/skewer solo tienen sentido tactico si AMBAS piezas alineadas
            # son del defensor (si p2 fuera del atacante, no hay nada que "ganar"
            # alineando la pieza propia detras de la pieza enemiga)
            if p2.color == defender_color:
                if v2 > v1:
                    relative_pins += 1
                elif v1 >= v2 or p1.piece_type == chess.KING:
                    skewers += 1

    return {"x_ray_attacks_on_defender": x_rays, "relative_pins_on_defender": relative_pins,
            "skewers_on_defender": skewers}


def compute_pins_skewers_both(board: chess.Board) -> dict:
    out = {}
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        abs_pins = compute_absolute_pins(board, color)
        out[f"{name}_absolute_pins"] = abs_pins["absolute_pins"]
        # geometria: piezas del color `name` como DEFENSORAS, atacante = color contrario
        rel = compute_relative_geometry(board, attacker_color=not color)
        out[f"{name}_relative_pins"] = rel["relative_pins_on_defender"]
        out[f"{name}_x_ray_attacks_against"] = rel["x_ray_attacks_on_defender"]
        out[f"{name}_skewers_against"] = rel["skewers_on_defender"]
    return out
