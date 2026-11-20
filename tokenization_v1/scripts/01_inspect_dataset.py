"""Inspeccion liviana del dataset de entrada (sin parsear las 4.47M partidas).

Qué hace:
- Confirma ruta, tamaño, fecha de modificación y un identificador liviano del
  archivo (no un hash SHA256 completo de 3.5GB: eso violaria el requisito de
  "no parsees/hashees las 4.471.784 partidas antes de la prueba" en espiritu,
  ya que recorrer el archivo completo por bytes para un solo hash es un costo
  comparable a indexarlo. Se usa un identificador compuesto: tamaño + mtime +
  hash de los primeros y ultimos 1MB, que detecta con alta confianza si el
  archivo cambio entre corridas).
- Cuenta partidas totales (conteo de '[Event ' en modo binario, streaming).
- Toma una muestra de las primeras N partidas (rápida, no aleatoria: es solo
  para caracterizar el formato, el muestreo real de la prueba lo hace 03 con
  la semilla fija) y mide: presencia de cabeceras, presencia de Elo,
  distribución de Result, distribución de PlyCount declarado, codificación.
"""
import hashlib
import json
import os
import time

SOURCE_PGN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "AJ-OTB-PGN-000.pgn"))
REPORT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "dataset_inspection.json"))
SAMPLE_SIZE = 2000
CHUNK_SIZE = 8 * 1024 * 1024


def lightweight_file_identifier(path: str) -> dict:
    stat = os.stat(path)
    with open(path, "rb") as f:
        head = f.read(1024 * 1024)
        f.seek(max(0, stat.st_size - 1024 * 1024))
        tail = f.read(1024 * 1024)
    h = hashlib.sha256()
    h.update(head)
    h.update(tail)
    h.update(str(stat.st_size).encode())
    return {
        "path": path,
        "size_bytes": stat.st_size,
        "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
        "identifier_method": "sha256(first_1MB + last_1MB + size_bytes) -- NO es hash del archivo completo",
        "identifier_sha256": h.hexdigest(),
    }


def detect_encoding(path: str) -> dict:
    with open(path, "rb") as f:
        head = f.read(4)
    has_bom = head.startswith(b"\xef\xbb\xbf")
    return {"has_utf8_bom": has_bom, "declared_encoding": "utf-8-sig" if has_bom else "utf-8"}


def count_total_games(path: str) -> int:
    marker = b"[Event "
    count = 0
    with open(path, "rb") as f:
        carry = b""
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            data = carry + chunk
            count += data.count(marker)
            carry = data[-len(marker):]
    return count


def sample_first_games(path: str, n: int) -> list:
    games = []
    current_lines = []
    with open(path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("utf-8-sig", errors="replace")
            stripped = line.strip()
            if stripped.startswith("[Event ") and current_lines:
                games.append("".join(current_lines))
                current_lines = [line]
                if len(games) >= n:
                    break
            else:
                current_lines.append(line)
        else:
            if current_lines:
                games.append("".join(current_lines))
    return games[:n]


def parse_headers_only(game_text: str) -> dict:
    headers = {}
    for line in game_text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try:
                key, rest = line[1:-1].split(" ", 1)
                headers[key] = rest.strip('"')
            except ValueError:
                continue
    return headers


def main():
    print(f"Archivo de entrada: {SOURCE_PGN}")
    assert os.path.isfile(SOURCE_PGN), f"No existe: {SOURCE_PGN}"

    t0 = time.time()
    file_id = lightweight_file_identifier(SOURCE_PGN)
    encoding_info = detect_encoding(SOURCE_PGN)
    print(f"Identificador liviano calculado en {time.time()-t0:.1f}s")

    t0 = time.time()
    total_games = count_total_games(SOURCE_PGN)
    count_elapsed = time.time() - t0
    print(f"Conteo total de partidas: {total_games} ({count_elapsed:.1f}s)")

    t0 = time.time()
    sample = sample_first_games(SOURCE_PGN, SAMPLE_SIZE)
    sample_elapsed = time.time() - t0
    print(f"Muestra de {len(sample)} partidas leida en {sample_elapsed:.1f}s")

    n_with_result_star = 0
    n_with_plycount_zero = 0
    n_with_elo_both = 0
    n_with_setup_fen = 0
    n_missing_headers = 0
    result_counts = {}
    plycounts = []

    for g in sample:
        h = parse_headers_only(g)
        if not h:
            n_missing_headers += 1
            continue
        result = h.get("Result", "")
        result_counts[result] = result_counts.get(result, 0) + 1
        if result == "*":
            n_with_result_star += 1
        pc = h.get("PlyCount")
        if pc is not None:
            try:
                pc_int = int(pc)
                plycounts.append(pc_int)
                if pc_int == 0:
                    n_with_plycount_zero += 1
            except ValueError:
                pass
        if h.get("WhiteElo") and h.get("BlackElo"):
            n_with_elo_both += 1
        if h.get("SetUp") == "1" and h.get("FEN"):
            n_with_setup_fen += 1

    report = {
        "source_file": file_id,
        "encoding": encoding_info,
        "total_games_counted": total_games,
        "count_elapsed_seconds": round(count_elapsed, 2),
        "sample_size": len(sample),
        "sample_elapsed_seconds": round(sample_elapsed, 2),
        "sample_stats": {
            "n_missing_headers": n_missing_headers,
            "n_with_result_star": n_with_result_star,
            "n_with_plycount_zero": n_with_plycount_zero,
            "n_with_elo_both": n_with_elo_both,
            "n_with_setup_fen": n_with_setup_fen,
            "result_distribution": result_counts,
            "plycount_mean": (sum(plycounts) / len(plycounts)) if plycounts else None,
            "plycount_max": max(plycounts) if plycounts else None,
        },
        "note": (
            "Este es un pre-diagnostico sobre una muestra de las primeras "
            f"{SAMPLE_SIZE} partidas del archivo (no aleatoria, solo para "
            "caracterizar formato). El muestreo real reproducible con seed=20260803 "
            "se hace en 03_generate_candidate_order.py sobre el indice completo "
            "de 02_build_game_index.py. Duplicados exhaustivos solo se miden "
            "sobre los candidatos efectivamente procesados en el piloto, no sobre "
            "las 4.471.784 partidas totales."
        ),
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Reporte escrito en {REPORT_PATH}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
