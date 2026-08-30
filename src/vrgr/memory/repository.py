"""
Αποθήκευση και ανάκτηση γεγονότων.

Κανόνας deduplication: ένα `media_id` = μία γραμμή στο `posts` (τα στατικά
του χαρακτηριστικά) + Ν γραμμές στο `post_snapshots` (οι μετρήσεις του σε
διαφορετικές στιγμές). Έτσι δεν χάνουμε ιστορικό ούτε δημιουργούμε διπλά.
"""
from __future__ import annotations

import json
import time
from typing import Iterable, Optional

from ..clients.llm import embeddings as EMB
from ..logging_setup import get_logger
from ..schemas import Creator, HashtagStat, ObservedPost
from .db import Database

log = get_logger("memory")

# Ένα νέο snapshot γράφεται μόνο αν πέρασε αρκετή ώρα από το προηγούμενο ή
# αν τα views άλλαξαν ουσιαστικά — αλλιώς η βάση γεμίζει με θόρυβο.
MIN_SNAPSHOT_GAP_S = 6 * 3600
MIN_VIEW_DELTA = 0.02


class Repository:
    def __init__(self, db: Database, embedder=None):
        self.db = db
        self.embedder = embedder

    # ── creators ─────────────────────────────────────────────────────
    def upsert_creator(self, c: Creator, niche: str = "",
                       is_benchmark: bool = False) -> None:
        if not c.pk:
            return
        now = time.time()
        self.db.execute(
            """INSERT INTO creators (pk, username, full_name, followers, following,
                   media_count, biography, category, is_private, is_verified,
                   greek_confidence, niche, is_benchmark, first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(pk) DO UPDATE SET
                   username=excluded.username, full_name=excluded.full_name,
                   followers=excluded.followers, following=excluded.following,
                   media_count=excluded.media_count, biography=excluded.biography,
                   category=excluded.category, is_private=excluded.is_private,
                   is_verified=excluded.is_verified,
                   greek_confidence=MAX(creators.greek_confidence, excluded.greek_confidence),
                   niche=CASE WHEN excluded.niche != '' THEN excluded.niche ELSE creators.niche END,
                   is_benchmark=MAX(creators.is_benchmark, excluded.is_benchmark),
                   last_seen=excluded.last_seen""",
            (c.pk, c.username, c.full_name, c.followers, c.following, c.media_count,
             c.biography, c.category, int(c.is_private), int(c.is_verified),
             c.greek_confidence, niche, int(is_benchmark), now, now))

    def get_creator(self, username: str) -> Optional[dict]:
        row = self.db.one("SELECT * FROM creators WHERE username=?", (username,))
        return dict(row) if row else None

    def benchmark_creators(self, limit: int = 100) -> list:
        rows = self.db.query(
            "SELECT * FROM creators WHERE is_benchmark=1 ORDER BY followers DESC LIMIT ?",
            (limit,))
        return [dict(r) for r in rows]

    # ── posts ────────────────────────────────────────────────────────
    def save_posts(self, posts: Iterable[ObservedPost], niche: str = "",
                   sub_niche: str = "") -> dict:
        stats = {"new_posts": 0, "new_snapshots": 0, "skipped_snapshots": 0}
        to_embed = []
        for p in posts:
            if not p.media_id:
                continue
            existing = self.db.one(
                "SELECT media_id FROM posts WHERE media_id=?", (p.media_id,))
            now = time.time()
            self.db.execute(
                """INSERT INTO posts (media_id, code, creator_pk, username, caption,
                       caption_body, hashtags_json, taken_at, duration_s, product_type,
                       music_title, music_artist, is_original_audio, location_name,
                       language, greek_confidence, niche, sub_niche, thumbnail_url,
                       first_seen, last_seen)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(media_id) DO UPDATE SET
                       caption=CASE WHEN excluded.caption != '' THEN excluded.caption
                                    ELSE posts.caption END,
                       niche=CASE WHEN excluded.niche != '' THEN excluded.niche
                                  ELSE posts.niche END,
                       last_seen=excluded.last_seen""",
                (p.media_id, p.code, p.creator_pk, p.username, p.caption,
                 p.caption_body, json.dumps(p.hashtags, ensure_ascii=False),
                 p.taken_at, p.duration_s, p.product_type, p.music_title,
                 p.music_artist,
                 None if p.is_original_audio is None else int(p.is_original_audio),
                 p.location_name, p.language, p.greek_confidence,
                 niche or "", sub_niche or "", p.thumbnail_url, now, now))

            if not existing:
                stats["new_posts"] += 1
                self.db.execute(
                    "INSERT INTO posts_fts (media_id, caption_body, hashtags, niche) "
                    "VALUES (?,?,?,?)",
                    (p.media_id, p.caption_body, " ".join(p.hashtags), niche or ""))
                to_embed.append(p)

            if self._should_snapshot(p):
                n = p.normalized
                self.db.execute(
                    """INSERT OR IGNORE INTO post_snapshots
                       (media_id, observed_at, followers, views, likes, comments,
                        vf_ratio, engagement_rate, comment_rate, viral_multiplier,
                        outlier_score, source_endpoint)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (p.media_id, p.observed_at, p.followers_at_observation,
                     p.metrics.views, p.metrics.likes, p.metrics.comments,
                     n.vf_ratio, n.engagement_rate, n.comment_rate,
                     n.viral_multiplier, n.outlier_score, p.source_endpoint))
                stats["new_snapshots"] += 1
            else:
                stats["skipped_snapshots"] += 1

            for tag in p.hashtags:
                self.db.execute(
                    "INSERT OR REPLACE INTO hashtag_evidence "
                    "(tag, media_id, outlier_score, niche, observed_at) VALUES (?,?,?,?,?)",
                    (tag, p.media_id, p.normalized.outlier_score, niche or "", now))

        if to_embed and self.embedder is not None:
            self._embed_posts(to_embed)
        return stats

    def _should_snapshot(self, p: ObservedPost) -> bool:
        """Αποφυγή θορύβου: νέο snapshot μόνο αν προσθέτει πληροφορία."""
        row = self.db.one(
            "SELECT observed_at, views FROM post_snapshots WHERE media_id=? "
            "ORDER BY observed_at DESC LIMIT 1", (p.media_id,))
        if not row:
            return True
        if time.time() - row["observed_at"] >= MIN_SNAPSHOT_GAP_S:
            return True
        old, new = row["views"], p.metrics.views
        if old and new:
            return abs(new - old) / max(1, old) >= MIN_VIEW_DELTA
        return False

    def _embed_posts(self, posts: list) -> None:
        texts, ids = [], []
        for p in posts:
            blob = f"{p.caption_body} {' '.join(p.hashtags)}".strip()
            if blob:
                texts.append(blob)
                ids.append(p.media_id)
        if not texts:
            return
        try:
            vectors = self.embedder.embed(texts)
        except Exception as exc:                       # noqa: BLE001
            log.warning("Αποτυχία embedding (%s) — τα posts μένουν χωρίς διάνυσμα",
                        type(exc).__name__)
            return
        now = time.time()
        rows = [
            (("post"), mid, self.embedder.name, int(vectors.shape[1]),
             EMB.pack(vectors[i]), str(hash(texts[i])), now)
            for i, mid in enumerate(ids)
        ]
        self.db.executemany(
            "INSERT OR REPLACE INTO embeddings "
            "(ref_type, ref_id, provider, dim, vector, text_hash, created_at) "
            "VALUES (?,?,?,?,?,?,?)", rows)

    def growth_rate(self, media_id: str) -> Optional[float]:
        """
        Views ανά ημέρα από διαδοχικά snapshots.

        Αυτό ΔΕΝ το δίνει το HikerAPI — προκύπτει επειδή κρατάμε ιστορικό.
        """
        rows = self.db.query(
            "SELECT observed_at, views FROM post_snapshots WHERE media_id=? "
            "AND views IS NOT NULL ORDER BY observed_at", (media_id,))
        if len(rows) < 2:
            return None
        first, last = rows[0], rows[-1]
        days = (last["observed_at"] - first["observed_at"]) / 86400.0
        if days < 0.2:
            return None
        return round((last["views"] - first["views"]) / days, 1)

    # ── hashtags ─────────────────────────────────────────────────────
    def save_hashtag_stat(self, s: HashtagStat) -> None:
        self.db.execute(
            """INSERT INTO hashtag_stats (tag, media_count, tier, is_greek, sample_size,
                   median_views_top, median_followers_top, small_account_share,
                   recency_days_median, difficulty, observed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(tag) DO UPDATE SET
                   media_count=COALESCE(excluded.media_count, hashtag_stats.media_count),
                   tier=excluded.tier, is_greek=excluded.is_greek,
                   sample_size=MAX(hashtag_stats.sample_size, excluded.sample_size),
                   median_views_top=COALESCE(excluded.median_views_top, hashtag_stats.median_views_top),
                   median_followers_top=COALESCE(excluded.median_followers_top, hashtag_stats.median_followers_top),
                   small_account_share=COALESCE(excluded.small_account_share, hashtag_stats.small_account_share),
                   recency_days_median=COALESCE(excluded.recency_days_median, hashtag_stats.recency_days_median),
                   difficulty=COALESCE(excluded.difficulty, hashtag_stats.difficulty),
                   observed_at=excluded.observed_at""",
            (s.tag, s.media_count, s.tier, int(s.is_greek), s.sample_size,
             s.median_views_top, s.median_followers_top, s.small_account_share,
             s.recency_days_median, s.difficulty, s.observed_at))

    def get_hashtag_stat(self, tag: str, max_age_days: float = 14.0) -> Optional[dict]:
        row = self.db.one("SELECT * FROM hashtag_stats WHERE tag=?", (tag,))
        if not row:
            return None
        if (time.time() - row["observed_at"]) / 86400.0 > max_age_days:
            return None
        return dict(row)

    def hashtag_evidence_count(self, tag: str, niche: str = "",
                               min_outlier: float = 45.0) -> int:
        """Σε πόσα ΕΠΙΤΥΧΗΜΕΝΑ posts εμφανίστηκε αυτό το hashtag."""
        if niche:
            row = self.db.one(
                "SELECT COUNT(*) c FROM hashtag_evidence WHERE tag=? AND niche=? "
                "AND outlier_score >= ?", (tag, niche, min_outlier))
        else:
            row = self.db.one(
                "SELECT COUNT(*) c FROM hashtag_evidence WHERE tag=? "
                "AND outlier_score >= ?", (tag, min_outlier))
        return row["c"] if row else 0

    def top_hashtags_for_niche(self, niche: str, limit: int = 60,
                               min_outlier: float = 45.0) -> list:
        rows = self.db.query(
            """SELECT tag, COUNT(*) n, AVG(outlier_score) avg_score
               FROM hashtag_evidence
               WHERE outlier_score >= ? AND (? = '' OR niche = ?)
               GROUP BY tag ORDER BY n DESC, avg_score DESC LIMIT ?""",
            (min_outlier, niche, niche, limit))
        return [(r["tag"], r["n"], round(r["avg_score"] or 0, 1)) for r in rows]

    # ── runs & outcomes ──────────────────────────────────────────────
    def save_run(self, run_id: str, payload: dict) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO runs (run_id, created_at, video_path, niche,
                   sub_niche, angle_name, angle_strategy, caption, hashtags_json,
                   predicted_score, confidence, api_calls, duration_s, result_json,
                   pattern_keys_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, payload.get("created_at", time.time()),
             payload.get("video_path", ""), payload.get("niche", ""),
             payload.get("sub_niche", ""), payload.get("angle_name", ""),
             payload.get("angle_strategy", ""), payload.get("caption", ""),
             json.dumps(payload.get("hashtags", []), ensure_ascii=False),
             payload.get("predicted_score"), payload.get("confidence", ""),
             payload.get("api_calls", 0), payload.get("duration_s", 0.0),
             json.dumps(payload.get("result", {}), ensure_ascii=False, default=str),
             json.dumps(payload.get("pattern_keys", []), ensure_ascii=False)))

    def get_run(self, run_id: str) -> Optional[dict]:
        row = self.db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return dict(row) if row else None

    def recent_runs(self, limit: int = 20) -> list:
        rows = self.db.query(
            "SELECT run_id, created_at, niche, angle_name, caption, predicted_score, "
            "confidence FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def save_outcome(self, run_id: str, data: dict) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO outcomes (run_id, posted_url, media_id, measured_at,
                   followers, views, likes, comments, vf_ratio, outlier_score,
                   predicted_score, error, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, data.get("posted_url", ""), data.get("media_id", ""),
             data.get("measured_at", time.time()), data.get("followers"),
             data.get("views"), data.get("likes"), data.get("comments"),
             data.get("vf_ratio"), data.get("outlier_score"),
             data.get("predicted_score"), data.get("error"), data.get("notes", "")))

    def outcomes(self) -> list:
        return [dict(r) for r in self.db.query(
            "SELECT * FROM outcomes ORDER BY measured_at DESC")]

    def log_api(self, run_id: str, per_endpoint: dict) -> None:
        now = time.time()
        rows = [(run_id, path, d.get("calls", 0), d.get("cache", 0),
                 d.get("errors", 0), d.get("calls", 0), now)
                for path, d in per_endpoint.items()]
        self.db.executemany(
            "INSERT INTO api_log (run_id, endpoint, calls, cache_hits, errors, "
            "cost_units, created_at) VALUES (?,?,?,?,?,?,?)", rows)
