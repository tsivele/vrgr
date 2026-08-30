"""
Κανονικοποίηση απόδοσης ως προς μέγεθος λογαριασμού (απαίτηση #14).

Η κεντρική ιδέα:

    1.500.000 views από λογαριασμό 5.000 followers   → V/F = 300
    2.000.000 views από λογαριασμό 10.000.000        → V/F = 0.2

Το πρώτο αποδεικνύει ΦΟΡΜΑΤ που ο αλγόριθμος έσπρωξε σε άγνωστο κοινό.
Το δεύτερο αποδεικνύει μόνο ότι ο λογαριασμός έχει ήδη κοινό.
Για να μάθουμε τι δουλεύει, μας ενδιαφέρει αποκλειστικά το πρώτο.
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Iterable, Optional

from ..schemas import NormalizedMetrics, ObservedPost

# Κατώφλια V/F βαθμονομημένα στη συμπεριφορά των Reels (2025-2026):
# ο αλγόριθμος σπρώχνει σχεδόν κάθε Reel σε ~1-2x του κοινού· το ενδιαφέρον
# ξεκινά όταν ξεπεράσει σαφώς αυτό το φυσικό όριο.
VF_BASELINE = 2.0
VF_STRONG = 8.0
VF_VIRAL = 25.0
VF_EXPLOSIVE = 100.0


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or not b:
        return None
    try:
        return a / b
    except ZeroDivisionError:
        return None


def compute(post: ObservedPost,
            creator_median_views: Optional[float] = None) -> NormalizedMetrics:
    """Μετρικές ενός post. Ό,τι δεν μπορεί να υπολογιστεί μένει `None`."""
    m = post.metrics
    followers = post.followers_at_observation
    views = m.views

    nm = NormalizedMetrics(
        vf_ratio=_safe_div(views, followers),
        lf_ratio=_safe_div(m.likes, followers),
        cf_ratio=_safe_div(m.comments, followers),
        like_rate=_safe_div(m.likes, views),
        comment_rate=_safe_div(m.comments, views),
        comment_to_like=_safe_div(m.comments, m.likes),
    )
    if m.likes is not None and m.comments is not None and views:
        nm.engagement_rate = (m.likes + m.comments) / views
    if creator_median_views:
        nm.viral_multiplier = _safe_div(views, creator_median_views)
    nm.outlier_score = outlier_score(nm, followers)
    return nm


def outlier_score(nm: NormalizedMetrics, followers: Optional[int]) -> Optional[float]:
    """
    0–100: «πόσο ξέφυγε αυτό το post από ό,τι θα περίμενε κανείς».

    Σύνθεση τριών ανεξάρτητων σημάτων:
      • V/F  — απήχηση εκτός υπάρχοντος κοινού (το βαρύτερο)
      • viral multiplier — breakout σε σχέση με τον ΙΔΙΟ creator
      • comment rate — ένταση αντίδρασης, όχι απλή κατανάλωση

    Bonus μικρού λογαριασμού: όταν ένας λογαριασμός <50K πετυχαίνει υψηλό V/F,
    το φορμά είναι η εξήγηση — δεν υπάρχει προϋπάρχον κοινό να το εξηγήσει.
    """
    if nm.vf_ratio is None:
        return None

    vf = nm.vf_ratio
    if vf <= VF_BASELINE:
        vf_pts = 22.0 * (vf / VF_BASELINE)
    else:
        # Λογαριθμική κλίμακα: το 300x δεν είναι «10x καλύτερο» από το 30x.
        vf_pts = 22.0 + 45.0 * min(1.0, math.log10(vf / VF_BASELINE) /
                                   math.log10(VF_EXPLOSIVE / VF_BASELINE))

    mult_pts = 0.0
    if nm.viral_multiplier:
        mult_pts = 20.0 * min(1.0, math.log10(max(1.0, nm.viral_multiplier)) / math.log10(20))

    eng_pts = 0.0
    if nm.comment_rate:
        # 1% comment rate σε Reel είναι ήδη πολύ υψηλό.
        eng_pts = 13.0 * min(1.0, nm.comment_rate / 0.01)

    score = vf_pts + mult_pts + eng_pts
    if followers and followers < 50_000 and vf >= VF_STRONG:
        score *= 1.08
    return round(min(100.0, score), 1)


def tier(nm: NormalizedMetrics) -> str:
    """Ετικέτα στα ελληνικά για αναφορές."""
    vf = nm.vf_ratio
    if vf is None:
        return "άγνωστο"
    if vf >= VF_EXPLOSIVE:
        return "εκρηκτικό"
    if vf >= VF_VIRAL:
        return "viral"
    if vf >= VF_STRONG:
        return "δυνατό"
    if vf >= VF_BASELINE:
        return "πάνω από τη βάση"
    return "κανονικό"


def creator_baseline(posts: Iterable[ObservedPost]) -> Optional[float]:
    """
    Διάμεσο views του creator — η βάση σύγκρισης για breakout.

    Διάμεσο και όχι μέσος όρος: ένα και μόνο viral post ανεβάζει τον μέσο
    όρο τόσο ώστε κάθε επόμενο breakout να μοιάζει φυσιολογικό.
    """
    vals = [p.metrics.views for p in posts
            if p.metrics.views and p.metrics.views > 0]
    return float(statistics.median(vals)) if len(vals) >= 3 else None


def enrich(posts: list, per_creator_baseline: bool = True) -> list:
    """Γεμίζει τα `normalized` για μια λίστα posts, με βάση ανά creator."""
    baselines = {}
    if per_creator_baseline:
        by_creator = {}
        for p in posts:
            if p.creator_pk:
                by_creator.setdefault(p.creator_pk, []).append(p)
        for pk, group in by_creator.items():
            base = creator_baseline(group)
            if base:
                baselines[pk] = base
    for p in posts:
        p.normalized = compute(p, baselines.get(p.creator_pk))
    return posts


RECENCY_HALFLIFE_D = 150.0


def rank_outliers(posts: list, min_score: float = 45.0,
                  greek_only: bool = True, min_greek: float = 0.5,
                  limit: int = 60, apply_recency: bool = True) -> list:
    """
    Τα posts που αξίζει να μάθουμε από αυτά.

    Δύο αποφάσεις ταξινόμησης:

    1. Φιλτράρουμε ΠΡΙΝ ταξινομήσουμε. Αν κρατούσαμε τα «top 60 σε views» και
       μετά φιλτράραμε, η λίστα θα γέμιζε με μεγάλους λογαριασμούς και θα
       χάναμε ακριβώς τα μικρά breakouts που μας ενδιαφέρουν.

    2. Χρονική στάθμιση κατά την ΚΑΤΑΤΑΞΗ (half-life 150 ημέρες).
       Μετρήθηκε σε πραγματική εκτέλεση: τα outliers είχαν διάμεσο ηλικίας
       96 ημερών αλλά εύρος έως 767 — και ένα post δυόμισι ετών εμφανιζόταν
       ως τεκμήριο για το «τι δουλεύει τώρα». Το αποθηκευμένο `outlier_score`
       μένει καθαρό (είναι γεγονός για τη στιγμή του)· η στάθμιση εφαρμόζεται
       μόνο εδώ, όπου απαντάμε «τι να μιμηθούμε σήμερα».
    """
    now = time.time()
    pool = []
    for p in posts:
        if greek_only and p.greek_confidence < min_greek:
            continue
        s = p.normalized.outlier_score
        if s is None or s < min_score:
            continue
        rank_key = s
        if apply_recency and p.taken_at:
            age_d = max(0.0, (now - p.taken_at) / 86400.0)
            # Πάτωμα 0.45: ένα παλιό αλλά εκρηκτικό post παραμένει διδακτικό,
            # απλώς δεν εκτοπίζει ένα πρόσφατο ισοδύναμο.
            rank_key = s * max(0.45, 0.5 ** (age_d / RECENCY_HALFLIFE_D))
        pool.append((rank_key, p))
    pool.sort(key=lambda t: -t[0])
    return [p for _, p in pool[:limit]]


def percentile_of(value: float, population: list) -> Optional[float]:
    """Σε ποιο εκατοστημόριο του πληθυσμού βρίσκεται η τιμή."""
    vals = sorted(v for v in population if v is not None)
    if not vals:
        return None
    below = sum(1 for v in vals if v < value)
    return round(100.0 * below / len(vals), 1)


def corpus_summary(posts: list) -> dict:
    """Συγκεντρωτικά για το report — όλα DERIVED από OBSERVED."""
    vf = [p.normalized.vf_ratio for p in posts if p.normalized.vf_ratio is not None]
    views = [p.metrics.views for p in posts if p.metrics.views]
    fol = [p.followers_at_observation for p in posts if p.followers_at_observation]
    small = [p for p in posts
             if p.followers_at_observation and p.followers_at_observation < 50_000]
    return {
        "n": len(posts),
        "n_greek": sum(1 for p in posts if p.greek_confidence >= 0.5),
        "median_vf": round(statistics.median(vf), 2) if vf else None,
        "max_vf": round(max(vf), 1) if vf else None,
        "median_views": int(statistics.median(views)) if views else None,
        "median_followers": int(statistics.median(fol)) if fol else None,
        "small_account_share": round(len(small) / len(posts), 3) if posts else None,
    }
