"""Parser de argumentos comun para todos los scripts de SEARCH TREE DATASET
V1.1 (Fase 16 del pedido: todos los scripts deben aceptar el mismo conjunto
de flags, aunque un script concreto no use todos). Cada script llama a
`build_common_parser()`, agrega los argumentos propios que necesite, y usa
solo los flags relevantes a su logica - los demas quedan disponibles con
default seguro para mantener una CLI uniforme en todo el pipeline."""

import argparse


def build_common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None)
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--max-positions", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--stockfish-path", default=None)
    p.add_argument("--stockfish-depth", type=int, default=16)
    p.add_argument("--stockfish-nodes", type=int, default=None)
    p.add_argument("--movetime-ms", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--elo-min", type=int, default=2200)
    p.add_argument("--elo-max", type=int, default=2300)
    p.add_argument("--require-both-ratings", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--exclude-bots", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--novelty-manifest", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--walk-length", type=int, default=4)
    p.add_argument("--trajectory-seed", type=int, default=None)
    return p
