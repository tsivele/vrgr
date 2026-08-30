"""
Κοινή βελτιστοποίηση λεζάντας × hashtags (απαίτηση #11).

ΓΙΑΤΙ ΔΕΝ ΒΕΛΤΙΣΤΟΠΟΙΟΥΝΤΑΙ ΧΩΡΙΣΤΑ:

Η καλύτερη λεζάντα και το καλύτερο σετ hashtags δεν δίνουν απαραίτητα τον
καλύτερο συνδυασμό. Παράδειγμα:

  Λεζάντα «περιέργειας» («δεν θα μαντέψεις τι έγινε μετά») χρειάζεται στενά
  hashtags — το κοινό πρέπει να έχει ήδη ενδιαφέρον για το θέμα, αλλιώς
  η υπόσχεση πέφτει στο κενό και το retention καταρρέει.

  Λεζάντα «ταύτισης» («αυτό είμαστε εμείς») δουλεύει με πλατύτερα hashtags —
  η ταύτιση λειτουργεί σε άγνωστο κοινό, δεν χρειάζεται προϋπάρχον ενδιαφέρον.

Γι' αυτό σκοράρουμε ΖΕΥΓΗ και προσθέτουμε ρητό όρο συνέργειας.
"""
from __future__ import annotations

from typing import Optional

from ..logging_setup import get_logger
from ..schemas import (CaptionCandidate, HashtagSet, MinedPatterns,
                       ResearchBundle, ScoredCombo, VideoAnalysis, ViralAngle)
from .viral_score import ViralScorer

log = get_logger("ranking")

# Ποια χαρτοφυλάκια ταιριάζουν σε ποια στρατηγική λεζάντας, και γιατί.
SYNERGY = {
    "περιέργεια": {"ανακάλυψη": 5.0, "ελληνική_στόχευση": 3.0,
                   "ισορροπημένο": 1.0, "εμβέλεια": -3.5},
    "ανοιχτός βρόχος": {"ανακάλυψη": 5.0, "ελληνική_στόχευση": 3.0,
                        "ισορροπημένο": 1.0, "εμβέλεια": -3.5},
    "ταύτιση": {"εμβέλεια": 4.0, "ισορροπημένο": 3.0,
                "ελληνική_στόχευση": 3.5, "ανακάλυψη": 0.5},
    "κοινωνική παρατήρηση": {"εμβέλεια": 3.5, "ελληνική_στόχευση": 4.0,
                             "ισορροπημένο": 2.5, "ανακάλυψη": 0.5},
    "χιούμορ": {"ελληνική_στόχευση": 4.5, "ισορροπημένο": 2.5,
                "εμβέλεια": 2.0, "ανακάλυψη": 1.0},
    "συναίσθημα": {"ισορροπημένο": 3.0, "ελληνική_στόχευση": 3.5,
                   "ανακάλυψη": 1.5, "εμβέλεια": 1.0},
    "αντιπαράθεση": {"ανακάλυψη": 4.0, "ισορροπημένο": 2.0,
                     "ελληνική_στόχευση": 2.5, "εμβέλεια": -2.0},
    "ερώτηση": {"ανακάλυψη": 3.5, "ελληνική_στόχευση": 3.0,
                "ισορροπημένο": 2.0, "εμβέλεια": -1.0},
    "αφήγηση": {"ανακάλυψη": 3.0, "ισορροπημένο": 2.0,
                "ελληνική_στόχευση": 2.5, "εμβέλεια": -1.5},
    "εκτενής αφήγηση": {"ανακάλυψη": 3.0, "ισορροπημένο": 2.0,
                        "ελληνική_στόχευση": 2.5, "εμβέλεια": -1.5},
    "σύντομη κοφτή": {"εμβέλεια": 3.0, "ισορροπημένο": 2.5,
                      "ελληνική_στόχευση": 2.0, "ανακάλυψη": 1.0},
    "απρόσμενη οπτική": {"ανακάλυψη": 3.5, "ισορροπημένο": 2.5,
                         "ελληνική_στόχευση": 2.0, "εμβέλεια": -1.0},
}


def synergy(caption: CaptionCandidate, hset: HashtagSet) -> float:
    return SYNERGY.get(caption.strategy, {}).get(hset.strategy, 0.0)


def rank(analysis: VideoAnalysis, angle: ViralAngle, captions: list,
         hashtag_sets: list, scorer: ViralScorer,
         research: Optional[ResearchBundle] = None,
         mined: Optional[MinedPatterns] = None,
         memory_support: Optional[dict] = None,
         memory_hits: int = 0, top_n: int = 6) -> list:
    """
    Σκοράρει ΚΑΘΕ ζεύγος και επιστρέφει τα κορυφαία, με εγγυημένη ποικιλία.

    Χωρίς τον κανόνα ποικιλίας, οι «εναλλακτικές» θα ήταν η ίδια λεζάντα με
    τέσσερα διαφορετικά σετ hashtags — που δεν είναι εναλλακτική.
    """
    if not captions or not hashtag_sets:
        return []

    combos = []
    for caption in captions:
        for hset in hashtag_sets:
            score = scorer.score(analysis, angle, caption, hset, research,
                                 mined, memory_support, memory_hits)
            syn = synergy(caption, hset)
            score.total = round(max(0.0, min(100.0, score.total + syn)), 1)
            score.interval = [round(max(0.0, score.interval[0] + syn), 1),
                              round(min(100.0, score.interval[1] + syn), 1)]
            combos.append(ScoredCombo(caption=caption, hashtag_set=hset,
                                      score=score, synergy=round(syn, 2)))

    combos.sort(key=lambda c: -c.score.total)
    log.info("Σκοράρισμα %d ζευγών (%d λεζάντες × %d σετ) — κορυφή %.1f",
             len(combos), len(captions), len(hashtag_sets), combos[0].score.total)

    # Κανόνας ποικιλίας: μία εμφάνιση ανά λεζάντα στα τελικά αποτελέσματα.
    selected, used_captions = [], set()
    for combo in combos:
        if combo.caption.id in used_captions:
            continue
        used_captions.add(combo.caption.id)
        combo.rank = len(selected) + 1
        selected.append(combo)
        if len(selected) >= top_n:
            break
    return selected


def alternative_sets(winner: ScoredCombo, hashtag_sets: list,
                     analysis: VideoAnalysis, angle: ViralAngle,
                     scorer: ViralScorer, research=None, mined=None,
                     memory_support=None, memory_hits: int = 0,
                     limit: int = 3) -> list:
    """Εναλλακτικά σετ hashtags ΓΙΑ ΤΗ ΝΙΚΗΤΡΙΑ λεζάντα."""
    out = []
    for hset in hashtag_sets:
        if hset.id == winner.hashtag_set.id:
            continue
        score = scorer.score(analysis, angle, winner.caption, hset, research,
                             mined, memory_support, memory_hits)
        syn = synergy(winner.caption, hset)
        score.total = round(max(0.0, min(100.0, score.total + syn)), 1)
        out.append(ScoredCombo(caption=winner.caption, hashtag_set=hset,
                               score=score, synergy=round(syn, 2)))
    out.sort(key=lambda c: -c.score.total)
    return out[:limit]


def explain(combo: ScoredCombo) -> str:
    """Σύντομη ελληνική αιτιολόγηση με βάση τους ισχυρότερους πυλώνες."""
    pillars = sorted(combo.score.pillars, key=lambda p: -p.weighted)
    top = [p for p in pillars[:3] if p.raw >= 55]
    weak = [p for p in pillars if p.raw < 45]
    parts = []
    if top:
        parts.append("Κέρδισε κυρίως σε: " + ", ".join(
            f"{p.label_el} ({p.raw:.0f}/100)" for p in top) + ".")
    if combo.synergy > 1.0:
        parts.append(
            f"Η στρατηγική «{combo.caption.strategy}» ταιριάζει με το "
            f"χαρτοφυλάκιο «{combo.hashtag_set.strategy}» (+{combo.synergy:.1f}).")
    elif combo.synergy < -1.0:
        parts.append(
            f"Παρά την ασυμβατότητα λεζάντας–hashtags ({combo.synergy:.1f}), "
            f"παρέμεινε το ισχυρότερο σύνολο.")
    if weak:
        parts.append("Αδύνατα σημεία: " + ", ".join(
            f"{p.label_el} ({p.raw:.0f})" for p in weak[:2]) + ".")
    return " ".join(parts)
