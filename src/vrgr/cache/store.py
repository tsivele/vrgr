"""
Επίμονο TTL cache για αποκρίσεις HikerAPI.

Γιατί sqlite και όχι in-memory: κάθε κλήση HikerAPI κοστίζει credits.
Το cache πρέπει να επιβιώνει restart, αλλιώς κάθε δοκιμή ξοδεύει ξανά.
Οι εγγραφές κρατιούνται και μετά τη λήξη τους (`stale`) ώστε σε outage
να μπορούμε να σερβίρουμε παλιά δεδομένα αντί να αποτύχουμε — με ρητή
σήμανση ότι είναι παλιά.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key         TEXT PRIMARY KEY,
    endpoint    TEXT NOT NULL,
    params      TEXT NOT NULL,
    body        TEXT NOT NULL,
    status      INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cache_endpoint ON http_cache(endpoint);
CREATE INDEX IF NOT EXISTS idx_cache_expires  ON http_cache(expires_at);
"""


def cache_key(endpoint: str, params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{endpoint}|{canonical}".encode("utf-8")).hexdigest()


class HttpCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, endpoint: str, params: dict,
            allow_stale: bool = False) -> Optional[Tuple[Any, bool]]:
        """Επιστρέφει `(payload, is_stale)` ή `None`."""
        key = cache_key(endpoint, params)
        with self._lock:
            row = self._conn.execute(
                "SELECT body, expires_at FROM http_cache WHERE key=?", (key,)
            ).fetchone()
            if not row:
                return None
            body, expires_at = row
            stale = time.time() > expires_at
            if stale and not allow_stale:
                return None
            self._conn.execute(
                "UPDATE http_cache SET hits = hits + 1 WHERE key=?", (key,))
            self._conn.commit()
        try:
            return json.loads(body), stale
        except json.JSONDecodeError:
            return None

    def put(self, endpoint: str, params: dict, payload: Any,
            ttl_s: float, status: int = 200) -> None:
        key = cache_key(endpoint, params)
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO http_cache "
                "(key, endpoint, params, body, status, created_at, expires_at, hits) "
                "VALUES (?,?,?,?,?,?,?,COALESCE((SELECT hits FROM http_cache WHERE key=?),0))",
                (key, endpoint,
                 json.dumps(params, sort_keys=True, ensure_ascii=False, default=str),
                 json.dumps(payload, ensure_ascii=False), status, now, now + ttl_s, key),
            )
            self._conn.commit()

    def purge_expired(self, older_than_days: float = 30.0) -> int:
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM http_cache WHERE expires_at < ?", (cutoff,))
            self._conn.commit()
            return cur.rowcount

    def vacuum(self) -> None:
        """Επιστρέφει τον ελεύθερο χώρο στο λειτουργικό μετά από διαγραφές."""
        with self._lock:
            self._conn.execute("VACUUM")
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(hits),0), "
                "COALESCE(SUM(LENGTH(body)),0) FROM http_cache").fetchone()
            fresh = self._conn.execute(
                "SELECT COUNT(*) FROM http_cache WHERE expires_at > ?",
                (time.time(),)).fetchone()[0]
        return {"entries": row[0], "fresh": fresh,
                "total_hits": row[1], "bytes": row[2]}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
