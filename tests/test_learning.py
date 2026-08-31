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


# ── κανονικοποίηση niche ──────────────────────────────────────────────

def test_free_text_niches_collapse_to_one_key():
    """
    Το σφάλμα που ακύρωνε ολόκληρη τη μάθηση.

    Τέσσερις αναλύσεις ΤΟΥ ΙΔΙΟΥ βίντεο έδωσαν τέσσερα διαφορετικά niche από
    το μοντέλο. Επειδή το niche είναι μέρος του κλειδιού των μοτίβων, κάθε
    εκτέλεση έφτιαχνε ΝΕΑ μοτίβα αντί να ενισχύει τα υπάρχοντα: 32 μοτίβα,
    μέσο n=0,56, κανένα αξιοποιήσιμο.
    """
    from vrgr.niches import canonical
    real_outputs = [
        ("Lifestyle & Personality Creators", "Talking-head thirst trap"),
        ("Lifestyle & Attention-Based Creator Content", "Talking-head κοπέλας"),
        ("Lifestyle και Personality Content", "Talking-head storytime"),
        ("Lifestyle / Προσωπικό Brand", "Talking-head με thirst-trap"),
    ]
    keys = {canonical(n, s) for n, s in real_outputs}
    assert keys == {"lifestyle"}, keys


def test_canonical_niche_handles_greek_english_and_greeklish():
    from vrgr.niches import canonical, OTHER
    assert canonical("Greek Comedy & Memes", "") == "χιούμορ"
    assert canonical("χιούμορ", "") == "χιούμορ"
    assert canonical("Relationship humor", "ζευγάρια") == "σχέσεις"
    assert canonical("Food & Recipes", "ελληνική κουζίνα") == "φαγητό"
    assert canonical("Fitness motivation", "προπόνηση") == "γυμναστική"
    # Χωρίς πραγματική ένδειξη → «άλλο», ποτέ λάθος κατηγορία που θα
    # μόλυνε τα μοτίβα άλλου niche.
    assert canonical("qwerty zxcvbn", "") == OTHER
    assert canonical("", "") == OTHER


def test_patterns_accumulate_across_runs_with_same_niche(db):
    """Με κανονικό κλειδί, τρεις εκτελέσεις δίνουν n=3, όχι 3× n=1."""
    from vrgr.memory.patterns import PatternStore, structure_key
    from vrgr.niches import canonical
    ps = PatternStore(db)
    for label in ("Lifestyle & Personality Creators",
                  "Lifestyle και Personality Content",
                  "Lifestyle / Προσωπικό Brand"):
        ps.observe(structure_key("ερώτηση"), "caption_structure",
                   success=True, niche=canonical(label))
    p = ps.get(structure_key("ερώτηση"), "lifestyle")
    assert p is not None and p.n == 3.0
