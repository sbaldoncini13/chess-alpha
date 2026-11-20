"""Cache persistente (SQLite) de requests al motor, para el full enrichment.
Fase 1/2/3 de la auditoria de reutilizacion.

PRINCIPIO (no negociable): la clave de cache (`engine_request_key`) se
construye a partir del FEN EXACTO enviado al motor (no del Zobrist
position_id -- la auditoria en 15_cache_audit.py demostro empiricamente que
9226 grupos de position_id comparten el mismo hash con FEN distinto por
halfmove_clock/fullmove_number) + toda la configuracion que puede cambiar
el resultado (version de motor, hash del binario, depth, multipv, threads,
hash_mb, tipo de request). Dos requests con la MISMA clave son, por
construccion, exactamente la misma llamada UCI.

Solo se cachea el RESULTADO CRUDO del motor (topk/best_move para root,
eval unificado para child) -- toda la logica de construccion de filas
(engine_records.py) y de engine_criticality corre IGUAL sobre datos de
cache que sobre datos frescos, asi que cache no puede alterar ninguna
formula ya validada.
"""
import hashlib
import json
import sqlite3
import time

REQUEST_TYPE_ROOT = "root_multipv"
REQUEST_TYPE_CHILD = "child_single_pv"


def compute_request_key(fen: str, engine_version: str, engine_binary_sha256: str,
                         depth: int, multipv: int, threads: int, hash_mb: int,
                         request_type: str) -> str:
    payload = (
        f"{fen}|{engine_version}|{engine_binary_sha256}|depth={depth}|"
        f"multipv={multipv}|threads={threads}|hash={hash_mb}|type={request_type}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ConfigMismatchError(Exception):
    """La cache existente pertenece a una configuracion de motor distinta --
    nunca se reutiliza silenciosamente en ese caso."""


class EngineCache:
    def __init__(self, db_path: str, config_meta: dict):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()
        self._check_or_write_meta(config_meta)
        self.root_hits = 0
        self.root_misses = 0
        self.child_hits = 0
        self.child_misses = 0

    def _ensure_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_meta (
                key TEXT PRIMARY KEY, value TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS root_cache (
                request_key TEXT PRIMARY KEY, fen TEXT, payload_json TEXT, created_utc TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS child_cache (
                request_key TEXT PRIMARY KEY, fen TEXT, payload_json TEXT, created_utc TEXT
            )
        """)
        self.conn.commit()

    def _check_or_write_meta(self, config_meta: dict):
        existing = {}
        for row in self.conn.execute("SELECT key, value FROM cache_meta"):
            existing[row[0]] = row[1]
        new_meta = {k: json.dumps(v, sort_keys=True) for k, v in config_meta.items()}
        if existing:
            mismatches = {
                k: (existing.get(k), new_meta.get(k))
                for k in set(existing) | set(new_meta)
                if existing.get(k) != new_meta.get(k)
            }
            if mismatches:
                raise ConfigMismatchError(
                    f"La cache en {self.db_path} pertenece a una configuracion distinta. "
                    f"Diferencias (existente vs nueva): {mismatches}. No se reutiliza "
                    f"silenciosamente -- borrar la cache o usar un run_id nuevo."
                )
        else:
            self.conn.executemany(
                "INSERT INTO cache_meta (key, value) VALUES (?, ?)", list(new_meta.items())
            )
            self.conn.commit()

    def get_root(self, request_key: str):
        row = self.conn.execute(
            "SELECT payload_json FROM root_cache WHERE request_key = ?", (request_key,)
        ).fetchone()
        if row is None:
            self.root_misses += 1
            return None
        self.root_hits += 1
        return json.loads(row[0])

    def put_root(self, request_key: str, fen: str, payload: dict):
        self.conn.execute(
            "INSERT OR IGNORE INTO root_cache (request_key, fen, payload_json, created_utc) VALUES (?, ?, ?, ?)",
            (request_key, fen, json.dumps(payload), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
        self.conn.commit()

    def get_child(self, request_key: str):
        row = self.conn.execute(
            "SELECT payload_json FROM child_cache WHERE request_key = ?", (request_key,)
        ).fetchone()
        if row is None:
            self.child_misses += 1
            return None
        self.child_hits += 1
        return json.loads(row[0])

    def put_child(self, request_key: str, fen: str, payload: dict):
        self.conn.execute(
            "INSERT OR IGNORE INTO child_cache (request_key, fen, payload_json, created_utc) VALUES (?, ?, ?, ?)",
            (request_key, fen, json.dumps(payload), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
        self.conn.commit()

    def counts(self) -> dict:
        n_root = self.conn.execute("SELECT COUNT(*) FROM root_cache").fetchone()[0]
        n_child = self.conn.execute("SELECT COUNT(*) FROM child_cache").fetchone()[0]
        return {
            "n_root_entries": n_root, "n_child_entries": n_child,
            "root_hits": self.root_hits, "root_misses": self.root_misses,
            "child_hits": self.child_hits, "child_misses": self.child_misses,
        }

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
