"""
Μοτίβα που μαθαίνουν (απαίτηση #6).

Κάθε μοτίβο είναι Beta(α, β):
    α = φορές που συνδέθηκε με επιτυχία   (+1 prior)
    β = φορές που δεν συνδέθηκε           (+1 prior)

Τρεις λόγοι για Beta αντί για ποσοστό:

  1. Ξεχωρίζει «2/2 = 100%» από «41/60 = 68%». Το πρώτο δεν σημαίνει τίποτα.
  2. Δίνει ΚΑΤΩ ΦΡΑΓΜΑ. Το σκορ χρησιμοποιεί το συντηρητικό άκρο, όχι τη
     μέση τιμή — έτσι ένα μοτίβο πρέπει να ΚΕΡΔΙΣΕΙ την επιρροή του.
  3. Η ενημέρωση με νέα δεδομένα είναι απλή πρόσθεση.

Χρονική απόσβεση: half-life 90 ημερών. Το Instagram του 2024 δεν είναι το
Instagram του 2026 — ένα μοτίβο που δεν επιβεβαιώνεται ξανά ξεθωριάζει.
"""
from __future__ import annotations

import json
import math
import time
from typing import Optional

from ..logging_setup import get_logger
from ..schemas import Pattern
from .db import Database

log = get_logger("patterns")

HALFLIFE_DAYS = 90.0
MIN_N_FOR_USE = 4.0          # κάτω από 4 δείγματα το μοτίβο δεν επηρεάζει σκορ


class PatternStore:
    def __init__(self, db: Database):
        self.db = db

    # ── ανάγνωση ─────────────────────────────────────────────────────
    def get(self, key: str, niche: str = "") -> Optional[Pattern]:
        row = self.db.one(
            "SELECT * FROM patterns WHERE key=? AND niche=?", (key, niche))
        if not row:
            return None
        return self._decayed(row)

    def by_kind(self, kind: str, niche: str = "", min_n: float = MIN_N_FOR_USE,
                limit: int = 40) -> list:
        rows = self.db.query(
            "SELECT * FROM patterns WHERE kind=? AND (niche=? OR niche='') "
            "ORDER BY (alpha + beta) DESC LIMIT ?", (kind, niche, limit * 3))
        out = [self._decayed(r) for r in rows]
        out = [p for p in out if p.n >= min_n]
        out.sort(key=lambda p: -p.lower_bound())
        return out[:limit]

    def _decayed(self, row) -> Pattern:
        """
        Εφαρμόζει χρονική απόσβεση κατά την ΑΝΑΓΝΩΣΗ.

        Έτσι δεν χρειάζεται περιοδική εργασία συντήρησης, και η απόσβεση
        είναι πάντα σωστή ανεξάρτητα από το πότε έτρεξε τελευταία φορά κάτι.
        """
        days = max(0.0, (time.time() - row["last_decayed"]) / 86400.0)
        factor = 0.5 ** (days / HALFLIFE_DAYS) if days > 1.0 else 1.0
        alpha = 1.0 + (row["alpha"] - 1.0) * factor
        beta = 1.0 + (row["beta"] - 1.0) * factor
        try:
            samples = json.loads(row["sample_ids_json"] or "[]")
        except json.JSONDecodeError:
            samples = []
        return Pattern(key=row["key"], kind=row["kind"], niche=row["niche"],
                       description_el=row["description_el"], alpha=alpha,
                       beta=beta, last_seen=row["last_seen"], sample_ids=samples)

    # ── ενημέρωση ────────────────────────────────────────────────────
    def observe(self, key: str, kind: str, success: bool, niche: str = "",
                description_el: str = "", weight: float = 1.0,
                sample_id: str = "") -> None:
        """Μία παρατήρηση. `weight` < 1 για ασθενέστερα τεκμήρια."""
        now = time.time()
        existing = self.db.one(
            "SELECT * FROM patterns WHERE key=? AND niche=?", (key, niche))
        if existing:
            cur = self._decayed(existing)
            alpha = cur.alpha + (weight if success else 0.0)
            beta = cur.beta + (0.0 if success else weight)
            samples = cur.sample_ids
            if sample_id and sample_id not in samples:
                samples = (samples + [sample_id])[-50:]
            self.db.execute(
                "UPDATE patterns SET alpha=?, beta=?, last_seen=?, last_decayed=?, "
                "sample_ids_json=?, description_el=CASE WHEN ?='' THEN description_el "
                "ELSE ? END WHERE key=? AND niche=?",
                (alpha, beta, now, now, json.dumps(samples, ensure_ascii=False),
                 description_el, description_el, key, niche))
        else:
            self.db.execute(
                "INSERT INTO patterns (key, kind, niche, description_el, alpha, beta, "
                "last_seen, last_decayed, sample_ids_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (key, kind, niche, description_el,
                 1.0 + (weight if success else 0.0),
                 1.0 + (0.0 if success else weight), now, now,
                 json.dumps([sample_id] if sample_id else [], ensure_ascii=False)))

    def observe_batch(self, observations: list) -> int:
        for o in observations:
            self.observe(**o)
        return len(observations)

    # ── χρήση από το σκορ ────────────────────────────────────────────
    def support(self, keys: list, niche: str = "") -> dict:
        """
        Πόσο υποστηρίζει η μνήμη ένα σύνολο χαρακτηριστικών.

        Επιστρέφει 0–1 σκορ + τα μοτίβα που το στηρίζουν, για το report.
        Χρησιμοποιεί ΚΑΤΩ ΦΡΑΓΜΑ: ένα μοτίβο με λίγα δείγματα δεν μπορεί
        να ανεβάσει το σκορ, όσο εντυπωσιακό κι αν φαίνεται το ποσοστό του.
        """
        hits, total_w, weighted = [], 0.0, 0.0
        for key in keys:
            p = self.get(key, niche) or self.get(key, "")
            if p is None or p.n < MIN_N_FOR_USE:
                continue
            # Βάρος = πόσα δείγματα, με κορεσμό στα ~25.
            w = min(1.0, math.log1p(p.n) / math.log1p(25))
            weighted += p.lower_bound() * w
            total_w += w
            hits.append({
                "key": key, "n": round(p.n, 1),
                "mean": round(p.mean, 3),
                "lower": round(p.lower_bound(), 3),
                "confidence": p.confidence,
                "description": p.description_el,
            })
        if total_w == 0:
            return {"score": 0.0, "coverage": 0.0, "patterns": []}
        hits.sort(key=lambda h: -h["lower"])
        return {
            "score": round(weighted / total_w, 3),
            "coverage": round(total_w / max(1, len(keys)), 3),
            "patterns": hits,
        }

    def stats(self) -> dict:
        row = self.db.one("SELECT COUNT(*) n, AVG(alpha+beta-2) avg_n FROM patterns")
        strong = self.db.one(
            "SELECT COUNT(*) n FROM patterns WHERE (alpha + beta - 2) >= ?",
            (MIN_N_FOR_USE,))
        return {"patterns": row["n"] if row else 0,
                "usable": strong["n"] if strong else 0,
                "avg_samples": round(row["avg_n"] or 0, 1) if row else 0}


# ── κλειδιά μοτίβων ───────────────────────────────────────────────────
# Σταθερά, αναγνώσιμα κλειδιά ώστε ένα μοτίβο να είναι συγκρίσιμο διαχρονικά.

def caption_key(feature: str, value) -> str:
    return f"caption:{feature}={value}"


def hashtag_key(tag: str) -> str:
    return f"hashtag:{tag}"


def structure_key(name: str) -> str:
    return f"structure:{name}"


def hook_key(name: str) -> str:
    return f"hook:{name}"


def length_bucket(n_chars: int) -> str:
    for limit, label in ((40, "πολύ σύντομη"), (90, "σύντομη"),
                         (180, "μεσαία"), (400, "μεγάλη")):
        if n_chars <= limit:
            return label
    return "πολύ μεγάλη"
