"""
Κανονικοποίηση niche σε σταθερό λεξιλόγιο.

ΤΟ ΣΦΑΛΜΑ ΠΟΥ ΛΥΝΕΙ — μετρημένο σε πραγματικά δεδομένα:

Τέσσερις αναλύσεις ΤΟΥ ΙΔΙΟΥ βίντεο παρήγαγαν τέσσερα διαφορετικά niche:

    «Lifestyle & Personality Creators»
    «Lifestyle & Attention-Based Creator Content»
    «Lifestyle και Personality Content»
    «Lifestyle / Προσωπικό Brand»

Το niche είναι ελεύθερο κείμενο από το μοντέλο, και χρησιμοποιείται ως μέρος
του πρωτεύοντος κλειδιού των μοτίβων `(key, niche)`. Αποτέλεσμα: κάθε
εκτέλεση δημιουργούσε ΝΕΟ σύνολο μοτίβων αντί να ενισχύει τα υπάρχοντα.

Μετά από 4 αναλύσεις: 32 μοτίβα, μέσο n=0,56, **κανένα αξιοποιήσιμο**
(χρειάζονται n≥4). Δηλαδή η μνήμη δεν επηρέαζε ποτέ το σκορ — ο πυλώνας
«Ιστορικά τεκμήρια» έμενε κολλημένος στην ουδέτερη τιμή 35 για πάντα.

Η λύση δεν είναι να δεσμεύσουμε το μοντέλο σε λίστα: η ελεύθερη περιγραφή
είναι χρήσιμη στο report. Κρατάμε και τα δύο — η αρχική περιγραφή μένει στο
`runs.sub_niche`, ενώ το ΚΛΕΙΔΙ γίνεται κανονικό.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from . import greek as G

OTHER = "άλλο"

# Συνώνυμα ανά κανονικό niche. Καλύπτουν ελληνικά, αγγλικά και greeklish,
# γιατί το μοντέλο γράφει και τα τρία.
SYNONYMS = {
    "χιούμορ": ("χιουμορ", "αστει", "κωμ", "meme", "funny", "comedy", "humor",
                "γελι", "σατιρ", "παρωδ", "joke"),
    "σχέσεις": ("σχεσ", "ζευγαρ", "ερωτ", "relationship", "couple", "dating",
                "love", "γκομεν", "συντροφ"),
    "lifestyle": ("lifestyle", "καθημερινοτητ", "personality", "vlog", "grwm",
                  "day in", "routine", "ρουτιν", "προσωπικο brand", "creator",
                  "talking head", "talkinghead", "storytime", "στοριταιμ"),
    "ομορφιά": ("ομορφ", "μακιγιαζ", "makeup", "beauty", "skincare", "μαλλι",
                "hair", "nails", "νυχι", "περιποιησ"),
    "φαγητό": ("φαγητ", "μαγειρ", "συνταγ", "food", "recipe", "cooking",
               "γλυκ", "ζαχαροπλαστ", "εστιατορ"),
    "ταξίδια": ("ταξιδ", "travel", "προορισμ", "νησ", "island", "διακοπ",
                "vacation", "παραλ", "beach", "τουρισμ"),
    "γυμναστική": ("γυμναστ", "fitness", "προπον", "workout", "gym", "ασκησ",
                   "αθλητ", "sport", "διατροφ", "bodybuild"),
    "χορός": ("χορ", "dance", "dancing", "χορευ", "choreograph"),
    "μουσική": ("μουσικ", "music", "τραγουδ", "song", "singer", "singing",
                "ραπ", "rap", "dj"),
    "εκπαίδευση": ("εκπαιδευ", "μαθ", "tips", "συμβουλ", "tutorial", "how to",
                   "πως να", "γνωσ", "educat", "explain", "διδα"),
}


@lru_cache(maxsize=1)
def _vocabulary() -> tuple:
    """Κανονικά niches από το config, με fallback στα συνώνυμα."""
    path = Path(__file__).resolve().parents[2] / "config" / "niches.json"
    names = list(SYNONYMS)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for name in (data.get("niches") or {}):
                if name not in names:
                    names.append(name)
        except (json.JSONDecodeError, OSError):
            pass
    return tuple(names)


def _score(text_norm: str, name: str) -> float:
    """Πόσο ταιριάζει ένα ελεύθερο niche σε ένα κανονικό όνομα."""
    score = 0.0
    if G.normalize(name) in text_norm:
        score += 3.0
    for token in SYNONYMS.get(name, ()):
        if G.normalize(token) in text_norm:
            # Μεγαλύτερα συνώνυμα είναι πιο ειδικά, άρα πιο αξιόπιστα.
            score += 1.0 + min(1.0, len(token) / 12.0)
    return score


def canonical(niche: str, sub_niche: str = "") -> str:
    """
    Ελεύθερο κείμενο → σταθερό κλειδί niche.

    Το `sub_niche` μετράει λιγότερο: περιγράφει το συγκεκριμένο βίντεο, ενώ
    το `niche` περιγράφει την κατηγορία. Χρησιμοποιείται μόνο ως tiebreaker.
    """
    if not niche and not sub_niche:
        return OTHER
    main = G.normalize(niche or "")
    sub = G.normalize(sub_niche or "")

    best, best_score = OTHER, 0.0
    for name in _vocabulary():
        score = _score(main, name) + 0.4 * _score(sub, name)
        if score > best_score:
            best, best_score = name, score
    # Κάτω από 1.0 δεν υπάρχει πραγματική ένδειξη — καλύτερα «άλλο» παρά
    # λάθος κατηγοριοποίηση που θα μόλυνε τα μοτίβα άλλου niche.
    return best if best_score >= 1.0 else OTHER


def all_niches() -> list:
    return list(_vocabulary()) + [OTHER]
