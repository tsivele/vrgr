"""
Σχήμα και σύνδεση βάσης.

Αρχιτεκτονική τριών επιπέδων:

  ΕΠΙΠΕΔΟ 1 — evidence   : ωμά γεγονότα, αμετάβλητα, με provenance
  ΕΠΙΠΕΔΟ 2 — derived    : ό,τι υπολογίζουμε από αυτά
  ΕΠΙΠΕΔΟ 3 — patterns   : κανόνες με Beta(α,β) βεβαιότητα

Κρίσιμη απόφαση: τα posts αποθηκεύονται ως ΣΤΙΓΜΙΟΤΥΠΑ (`post_snapshots`),
όχι ως μοναδικές γραμμές. Το ίδιο media_id ξαναμετρημένο μετά από μια
εβδομάδα δεν αντικαθιστά το παλιό — προστίθεται. Έτσι το HikerAPI, που
δίνει μόνο στιγμιότυπα, μας χτίζει δωρεάν ΙΣΤΟΡΙΚΟ: ταχύτητα ανάπτυξης
views, καμπύλες, και ανίχνευση τι μεγάλωσε γρήγορα.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);

-- ═══ ΕΠΙΠΕΔΟ 1: ΓΕΓΟΝΟΤΑ ═══════════════════════════════════════════

CREATE TABLE IF NOT EXISTS creators (
    pk               TEXT PRIMARY KEY,
    username         TEXT NOT NULL,
    full_name        TEXT DEFAULT '',
    followers        INTEGER DEFAULT 0,
    following        INTEGER DEFAULT 0,
    media_count      INTEGER DEFAULT 0,
    biography        TEXT DEFAULT '',
    category         TEXT DEFAULT '',
    is_private       INTEGER DEFAULT 0,
    is_verified      INTEGER DEFAULT 0,
    greek_confidence REAL DEFAULT 0,
    niche            TEXT DEFAULT '',
    is_benchmark     INTEGER DEFAULT 0,
    first_seen       REAL NOT NULL,
    last_seen        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_creators_username  ON creators(username);
CREATE INDEX IF NOT EXISTS idx_creators_greek     ON creators(greek_confidence);
CREATE INDEX IF NOT EXISTS idx_creators_followers ON creators(followers);

CREATE TABLE IF NOT EXISTS posts (
    media_id        TEXT PRIMARY KEY,
    code            TEXT DEFAULT '',
    creator_pk      TEXT DEFAULT '',
    username        TEXT DEFAULT '',
    caption         TEXT DEFAULT '',
    caption_body    TEXT DEFAULT '',
    hashtags_json   TEXT DEFAULT '[]',
    taken_at        INTEGER,
    duration_s      REAL,
    product_type    TEXT DEFAULT '',
    music_title     TEXT DEFAULT '',
    music_artist    TEXT DEFAULT '',
    is_original_audio INTEGER,
    location_name   TEXT DEFAULT '',
    language        TEXT DEFAULT '',
    greek_confidence REAL DEFAULT 0,
    niche           TEXT DEFAULT '',
    sub_niche       TEXT DEFAULT '',
    hook_type       TEXT DEFAULT '',
    emotional_angle TEXT DEFAULT '',
    caption_structure TEXT DEFAULT '',
    thumbnail_url   TEXT DEFAULT '',
    first_seen      REAL NOT NULL,
    last_seen       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_creator ON posts(creator_pk);
CREATE INDEX IF NOT EXISTS idx_posts_greek   ON posts(greek_confidence);
CREATE INDEX IF NOT EXISTS idx_posts_niche   ON posts(niche);
CREATE INDEX IF NOT EXISTS idx_posts_taken   ON posts(taken_at);

-- Στιγμιότυπα: ΠΟΤΕ δεν αντικαθίστανται. Εδώ χτίζεται το ιστορικό.
CREATE TABLE IF NOT EXISTS post_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id        TEXT NOT NULL REFERENCES posts(media_id) ON DELETE CASCADE,
    observed_at     REAL NOT NULL,
    followers       INTEGER,
    views           INTEGER,
    likes           INTEGER,
    comments        INTEGER,
    vf_ratio        REAL,
    engagement_rate REAL,
    comment_rate    REAL,
    viral_multiplier REAL,
    outlier_score   REAL,
    source_endpoint TEXT DEFAULT '',
    UNIQUE(media_id, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_snap_media   ON post_snapshots(media_id);
CREATE INDEX IF NOT EXISTS idx_snap_outlier ON post_snapshots(outlier_score);
CREATE INDEX IF NOT EXISTS idx_snap_time    ON post_snapshots(observed_at);

CREATE TABLE IF NOT EXISTS hashtag_stats (
    tag                TEXT PRIMARY KEY,
    media_count        INTEGER,
    tier               TEXT DEFAULT 'unknown',
    is_greek           INTEGER DEFAULT 0,
    sample_size        INTEGER DEFAULT 0,
    median_views_top   REAL,
    median_followers_top REAL,
    small_account_share REAL,
    recency_days_median REAL,
    difficulty         REAL,
    observed_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tag_greek ON hashtag_stats(is_greek);
CREATE INDEX IF NOT EXISTS idx_tag_tier  ON hashtag_stats(tier);

-- Σε πόσα (και ποια) επιτυχημένα posts εμφανίστηκε κάθε hashtag.
CREATE TABLE IF NOT EXISTS hashtag_evidence (
    tag           TEXT NOT NULL,
    media_id      TEXT NOT NULL,
    outlier_score REAL,
    niche         TEXT DEFAULT '',
    observed_at   REAL NOT NULL,
    PRIMARY KEY (tag, media_id)
);
CREATE INDEX IF NOT EXISTS idx_htev_tag   ON hashtag_evidence(tag);
CREATE INDEX IF NOT EXISTS idx_htev_niche ON hashtag_evidence(niche);

-- ═══ ΕΠΙΠΕΔΟ 2: ΣΗΜΑΣΙΟΛΟΓΙΚΗ ΜΝΗΜΗ ═══════════════════════════════

CREATE TABLE IF NOT EXISTS embeddings (
    ref_type   TEXT NOT NULL,          -- post | run | angle
    ref_id     TEXT NOT NULL,
    provider   TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,          -- float32 little-endian
    text_hash  TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (ref_type, ref_id, provider)
);

CREATE VIRTUAL TABLE IF NOT EXISTS posts_fts USING fts5(
    media_id UNINDEXED,
    caption_body,
    hashtags,
    niche,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- ═══ ΕΠΙΠΕΔΟ 3: ΜΟΤΙΒΑ ΠΟΥ ΜΑΘΑΙΝΟΥΝ ═════════════════════════════

CREATE TABLE IF NOT EXISTS patterns (
    key             TEXT NOT NULL,
    kind            TEXT NOT NULL,
    niche           TEXT DEFAULT '',
    description_el  TEXT DEFAULT '',
    alpha           REAL NOT NULL DEFAULT 1.0,
    beta            REAL NOT NULL DEFAULT 1.0,
    last_seen       REAL NOT NULL,
    last_decayed    REAL NOT NULL,
    sample_ids_json TEXT DEFAULT '[]',
    PRIMARY KEY (key, niche)
);
CREATE INDEX IF NOT EXISTS idx_patterns_kind  ON patterns(kind);
CREATE INDEX IF NOT EXISTS idx_patterns_niche ON patterns(niche);

-- ═══ ΕΚΤΕΛΕΣΕΙΣ & ΑΝΑΤΡΟΦΟΔΟΤΗΣΗ ═════════════════════════════════

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    video_path      TEXT DEFAULT '',
    niche           TEXT DEFAULT '',
    sub_niche       TEXT DEFAULT '',
    angle_name      TEXT DEFAULT '',
    angle_strategy  TEXT DEFAULT '',
    caption         TEXT DEFAULT '',
    hashtags_json   TEXT DEFAULT '[]',
    predicted_score REAL,
    confidence      TEXT DEFAULT '',
    api_calls       INTEGER DEFAULT 0,
    duration_s      REAL DEFAULT 0,
    result_json     TEXT DEFAULT '{}',
    pattern_keys_json TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_niche   ON runs(niche);

-- Ο βρόχος που κάνει το σύστημα να μαθαίνει πραγματικά.
CREATE TABLE IF NOT EXISTS outcomes (
    run_id           TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    posted_url       TEXT DEFAULT '',
    media_id         TEXT DEFAULT '',
    measured_at      REAL NOT NULL,
    followers        INTEGER,
    views            INTEGER,
    likes            INTEGER,
    comments         INTEGER,
    vf_ratio         REAL,
    outlier_score    REAL,
    predicted_score  REAL,
    error            REAL,
    notes            TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS api_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT DEFAULT '',
    endpoint   TEXT NOT NULL,
    calls      INTEGER DEFAULT 0,
    cache_hits INTEGER DEFAULT 0,
    errors     INTEGER DEFAULT 0,
    cost_units INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);
"""


class Database:
    """Thread-safe wrapper γύρω από sqlite3."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    def migrate(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),))
            self._conn.commit()

    # ── βοηθητικά ────────────────────────────────────────────────────
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def executemany(self, sql: str, rows: list) -> None:
        if not rows:
            return
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def counts(self) -> dict:
        tables = ("creators", "posts", "post_snapshots", "hashtag_stats",
                  "hashtag_evidence", "patterns", "runs", "outcomes", "embeddings")
        out = {}
        for t in tables:
            try:
                out[t] = self.one(f"SELECT COUNT(*) c FROM {t}")["c"]
            except sqlite3.Error:
                out[t] = -1
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()
