"""Μάθηση: Beta μοτίβα, χρονική απόσβεση, ανατροφοδότηση."""
import time

from vrgr.memory.patterns import PatternStore, structure_key
from vrgr.learning.feedback import _spearman


def test_many_samples_beat_perfect_but_tiny(db):
    """
    Ο λόγος ύπαρξης της Beta: το «2/2 = 100%» δεν είναι απόδειξη.
    """
    ps = PatternStore(db)
    for i in range(60):
        ps.observe(structure_key("ερώτηση"), "caption_structure",
                   success=(i % 10 < 7), niche="χιούμορ")
    for _ in range(2):
        ps.observe(structure_key("emoji"), "caption_structure", True, "χιούμορ")

    strong = ps.get(structure_key("ερώτηση"), "χιούμορ")
    weak = ps.get(structure_key("emoji"), "χιούμορ")
    assert weak.mean > strong.mean               # ο μέσος όρος λέει ψέματα
    assert strong.lower_bound() > weak.lower_bound()   # το κάτω φράγμα όχι
    assert strong.confidence > weak.confidence * 2


def test_support_ignores_undersampled_patterns(db):
    ps = PatternStore(db)
    for i in range(30):
        ps.observe(structure_key("τεκμηριωμένο"), "caption_structure",
                   success=True, niche="n")
    ps.observe(structure_key("ατεκμηρίωτο"), "caption_structure", True, "n")
    support = ps.support([structure_key("τεκμηριωμένο"),
                          structure_key("ατεκμηρίωτο")], "n")
    keys = [p["key"] for p in support["patterns"]]
    assert structure_key("τεκμηριωμένο") in keys
    assert structure_key("ατεκμηρίωτο") not in keys
    assert support["coverage"] == 0.5


def test_time_decay_fades_unconfirmed_patterns(db):
    """Το Instagram του 2024 δεν είναι του 2026 — τα παλιά μοτίβα ξεθωριάζουν."""
    ps = PatternStore(db)
    for _ in range(40):
        ps.observe(structure_key("παλιό"), "caption_structure", True, "n")
    fresh = ps.get(structure_key("παλιό"), "n")
    old_time = time.time() - 180 * 86400          # 2 ημιζωές πίσω
    db.execute("UPDATE patterns SET last_decayed=? WHERE key=?",
               (old_time, structure_key("παλιό")))
    decayed = ps.get(structure_key("παλιό"), "n")
    assert decayed.n < fresh.n * 0.35
    assert decayed.confidence < fresh.confidence


def test_spearman_measures_ranking_not_value():
    """Το σκορ υπόσχεται ΣΕΙΡΑ, όχι τιμή — άρα Spearman, όχι Pearson."""
    assert _spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == 1.0
    assert _spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == -1.0
    assert _spearman([1, 2], [1, 2]) is None       # πολύ λίγα δείγματα
