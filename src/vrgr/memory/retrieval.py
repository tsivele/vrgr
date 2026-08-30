"""
Υβριδική ανάκτηση από τη μνήμη.

Γιατί υβριδική και όχι σκέτο vector search:

  • Σημασιολογική αναζήτηση βρίσκει «σχέση θέματος» αλλά χάνει ακριβείς
    όρους (ένα σπάνιο hashtag, ένα συγκεκριμένο brand).
  • Λεξιλογική (FTS5) βρίσκει ακριβείς όρους αλλά χάνει παραφράσεις.
  • Στα ελληνικά το πρόβλημα οξύνεται: πλούσια μορφολογία σπάει το lexical,
    και τα μικρά κείμενα δίνουν φτωχό σήμα στο σημασιολογικό.

Η σύντηξη γίνεται με Reciprocal Rank Fusion — δουλεύει με ΘΕΣΕΙΣ αντί για
σκορ, οπότε δεν χρειάζεται να κάνουμε κοινή κλίμακα δύο ασύμβατα μέτρα.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import numpy as np

from .. import greek as G
from ..clients.llm import embeddings as EMB
from ..logging_setup import get_logger
from .db import Database

log = get_logger("retrieval")

RRF_K = 60.0          # σταθερά εξομάλυνσης RRF — τυπική τιμή βιβλιογραφίας
RECENCY_HALFLIFE_D = 120.0
_FTS_UNSAFE = re.compile(r'[^\w\sά-ώΆ-ΏϊϋΐΰΪΫ]', re.UNICODE)


class MemoryRetriever:
    def __init__(self, db: Database, embedder=None):
        self.db = db
        self.embedder = embedder

    # ── συστατικές αναζητήσεις ───────────────────────────────────────
    def _lexical(self, query: str, limit: int, niche: str = "") -> list:
        terms = [t for t in G.tokenize(query, drop_stopwords=True) if len(t) > 2]
        if not terms:
            return []
        expr = " OR ".join(f'"{t}"' for t in terms[:24])
        sql = ("SELECT media_id, bm25(posts_fts) AS rank FROM posts_fts "
               "WHERE posts_fts MATCH ? ")
        params: tuple = (expr,)
        if niche:
            sql += "AND niche = ? "
            params = (expr, niche)
        sql += "ORDER BY rank LIMIT ?"
        try:
            rows = self.db.query(sql, params + (limit,))
        except Exception as exc:                      # noqa: BLE001
            log.debug("FTS απέτυχε: %s", exc)
            return []
        return [r["media_id"] for r in rows]

    def _semantic(self, query: str, limit: int) -> list:
        if self.embedder is None:
            return []
        rows = self.db.query(
            "SELECT ref_id, dim, vector FROM embeddings "
            "WHERE ref_type='post' AND provider=?", (self.embedder.name,))
        if not rows:
            return []
        try:
            qv = self.embedder.embed([query])[0]
        except Exception as exc:                      # noqa: BLE001
            log.warning("Embedding ερωτήματος απέτυχε: %s", type(exc).__name__)
            return []
        dim = rows[0]["dim"]
        matrix = np.vstack([EMB.unpack(r["vector"], dim) for r in rows])
        sims = EMB.cosine(qv, matrix)
        order = np.argsort(-sims)[:limit]
        return [rows[int(i)]["ref_id"] for i in order]

    # ── σύντηξη ──────────────────────────────────────────────────────
    @staticmethod
    def _rrf(rankings: list, weights: Optional[list] = None) -> dict:
        weights = weights or [1.0] * len(rankings)
        fused: dict = {}
        for ranking, w in zip(rankings, weights):
            for pos, key in enumerate(ranking):
                fused[key] = fused.get(key, 0.0) + w / (RRF_K + pos + 1)
        return fused

    def search(self, query: str, limit: int = 20, niche: str = "",
               greek_only: bool = True, min_outlier: float = 0.0,
               max_age_days: Optional[float] = None) -> list:
        """
        Επιστρέφει ιστορικά posts όμοια με το ερώτημα, ταξινομημένα κατά
        συνδυασμό ομοιότητας, αποδεδειγμένης επιτυχίας και φρεσκάδας.
        """
        pool = max(limit * 6, 90)
        lex = self._lexical(query, pool, niche)
        sem = self._semantic(query, pool)
        if not lex and not sem:
            return []
        # Το σημασιολογικό βαραίνει λίγο περισσότερο: για λεζάντες, η
        # παράφραση είναι ο κανόνας, όχι η εξαίρεση.
        fused = self._rrf([sem, lex], [1.0, 0.85])
        ids = sorted(fused, key=lambda k: -fused[k])[:pool]
        if not ids:
            return []

        placeholders = ",".join("?" * len(ids))
        sql = f"""
            SELECT p.*, s.views, s.likes, s.comments, s.followers, s.vf_ratio,
                   s.outlier_score, s.observed_at
            FROM posts p
            LEFT JOIN (
                SELECT ps.* FROM post_snapshots ps
                JOIN (SELECT media_id, MAX(observed_at) mo
                      FROM post_snapshots GROUP BY media_id) last
                  ON ps.media_id = last.media_id AND ps.observed_at = last.mo
            ) s ON s.media_id = p.media_id
            WHERE p.media_id IN ({placeholders})
        """
        rows = [dict(r) for r in self.db.query(sql, tuple(ids))]

        now = time.time()
        results = []
        for row in rows:
            if greek_only and (row.get("greek_confidence") or 0) < 0.5:
                continue
            outlier = row.get("outlier_score") or 0.0
            if outlier < min_outlier:
                continue
            if max_age_days and row.get("taken_at"):
                if (now - row["taken_at"]) / 86400.0 > max_age_days:
                    continue
            sim = fused.get(row["media_id"], 0.0)
            # Κανονικοποίηση RRF σε ~0-1 (max θεωρητικό ≈ 2/(K+1)).
            sim_n = min(1.0, sim * (RRF_K + 1) / 1.85)
            success = min(1.0, outlier / 80.0)
            age_d = ((now - row["taken_at"]) / 86400.0) if row.get("taken_at") else 180.0
            recency = 0.5 ** (age_d / RECENCY_HALFLIFE_D)
            row["similarity"] = round(sim_n, 4)
            row["retrieval_score"] = round(
                0.50 * sim_n + 0.35 * success + 0.15 * recency, 4)
            try:
                row["hashtags"] = json.loads(row.get("hashtags_json") or "[]")
            except json.JSONDecodeError:
                row["hashtags"] = []
            results.append(row)

        results.sort(key=lambda r: -r["retrieval_score"])
        return results[:limit]

    def similar_to_video(self, content, angle=None, limit: int = 20,
                         niche: str = "") -> list:
        """
        Χτίζει ερώτημα από την ΑΝΑΛΥΣΗ του βίντεο, όχι από τη λεζάντα.

        Έτσι βρίσκουμε «όμοιες viral καταστάσεις»: ίδιο συναίσθημα, ίδιο
        κοινό, ίδια δυναμική — ακόμη κι αν τα λόγια είναι εντελώς άλλα.
        """
        parts = [content.summary, content.main_subject, content.mood,
                 content.niche, content.sub_niche, content.target_audience,
                 content.humor, content.aesthetic,
                 " ".join(content.actions or []),
                 " ".join(content.emotions or []),
                 " ".join(content.on_screen_text or []),
                 " ".join(content.cultural_markers or []),
                 (content.spoken_transcript or "")[:400]]
        if angle is not None:
            parts += [angle.name, angle.strategy, angle.caption_should_add,
                      angle.target_segment]
        query = " ".join(p for p in parts if p).strip()
        if not query:
            return []
        return self.search(query, limit=limit, niche=niche or content.niche,
                           greek_only=True, min_outlier=35.0)

    def corpus_stats(self) -> dict:
        row = self.db.one(
            """SELECT COUNT(*) n,
                      SUM(CASE WHEN greek_confidence >= 0.5 THEN 1 ELSE 0 END) gr
               FROM posts""")
        snaps = self.db.one("SELECT COUNT(*) n FROM post_snapshots")
        outl = self.db.one(
            "SELECT COUNT(DISTINCT media_id) n FROM post_snapshots WHERE outlier_score >= 45")
        return {"posts": row["n"] if row else 0,
                "greek_posts": (row["gr"] or 0) if row else 0,
                "snapshots": snaps["n"] if snaps else 0,
                "outliers": outl["n"] if outl else 0}
