"""
Εργαλεία ελληνικής γλώσσας.

Γιατί χρειάζεται δικό μας layer αντί για έτοιμη βιβλιοθήκη:
  • Ο τονισμός στα ελληνικά είναι θόρυβος για ταίριασμα κειμένου
    («καφές» / «καφες» πρέπει να είναι το ίδιο token).
  • Το τελικό σίγμα («ς» vs «σ») σπάει κάθε naive matching.
  • Στο Instagram το μισό ελληνικό κοινό γράφει greeklish — πρέπει να
    το αναγνωρίζουμε ως ελληνικό περιεχόμενο, όχι ως αγγλικό.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

GREEK_RANGE = re.compile(r"[Ͱ-Ͽἀ-῿]")
LATIN_RANGE = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_WS = re.compile(r"\s+")
HASHTAG_RE = re.compile(r"#([^\s#.,!?;:()\[\]{}\"'«»…]+)", re.UNICODE)
MENTION_RE = re.compile(r"@([A-Za-z0-9_.]+)")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF❤️]+"
)

# Λέξεις-δείκτες greeklish: συχνές ελληνικές λέξεις σε λατινικούς χαρακτήρες.
GREEKLISH_MARKERS = {
    "einai", "eimai", "exei", "exo", "kai", "gia", "apo", "sto", "stin", "ston",
    "mou", "sou", "tou", "tis", "poly", "poli", "kala", "kalo", "kali", "oxi",
    "nai", "tora", "meta", "prin", "otan", "epeidi", "giati", "ti", "pou",
    "pws", "pos", "afto", "auto", "auti", "afti", "ola", "oloi", "kanena",
    "tipota", "panta", "pote", "mono", "akoma", "akomh", "telos", "arxi",
    "zwh", "zoi", "agapi", "filos", "fili", "paidia", "koritsia", "gynaikes",
    "andres", "spiti", "douleia", "leftá", "lefta", "xrimata", "trela",
    "gamw", "vre", "re", "mwre", "ade", "ela", "opa", "malaka", "file",
}

# Ελληνικά stopwords — αφαιρούνται από την εξόρυξη μοτίβων γιατί εμφανίζονται
# παντού και δεν διακρίνουν επιτυχημένο από αποτυχημένο περιεχόμενο.
GREEK_STOPWORDS = {
    "ο", "η", "το", "οι", "τα", "του", "της", "των", "τον", "την", "και", "κι",
    "να", "θα", "με", "σε", "για", "από", "απο", "ως", "στο", "στη", "στην",
    "στον", "στα", "στις", "στους", "που", "πως", "πώς", "τι", "αν", "ή", "αλλά",
    "αλλα", "όμως", "ομως", "γιατί", "γιατι", "είναι", "ειναι", "ήταν", "ηταν",
    "έχω", "εχω", "έχει", "εχει", "μου", "σου", "του", "μας", "σας", "τους",
    "ένα", "ενα", "μια", "μία", "ένας", "ενας", "δεν", "μην", "μη", "ναι", "όχι",
    "οχι", "πολύ", "πολυ", "πιο", "πάλι", "παλι", "εγώ", "εγω", "εσύ", "εσυ",
    "αυτό", "αυτο", "αυτή", "αυτη", "αυτός", "αυτος", "όλα", "ολα", "κάθε",
    "καθε", "ήδη", "ηδη", "ακόμα", "ακομα", "όταν", "οταν",
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "this", "that", "it",
    "you", "your", "my", "me", "i", "we", "they", "he", "she", "not", "no",
    "so", "just", "do", "does", "did", "have", "has", "at", "by", "from",
}

STOPWORDS = GREEK_STOPWORDS | ENGLISH_STOPWORDS


def strip_accents(text: str) -> str:
    """Αφαιρεί τόνους/διαλυτικά διατηρώντας τα γράμματα."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    )


def normalize(text: str) -> str:
    """Κανονικοποίηση για ταίριασμα: πεζά, χωρίς τόνους, τελικό σίγμα → σ."""
    if not text:
        return ""
    return strip_accents(text).lower().replace("ς", "σ")


def normalize_tag(text: str) -> str:
    """
    Κανονικοποίηση για HASHTAG — διαφορετική από την `normalize`.

    Το Instagram αγνοεί τόνους αλλά ΟΧΙ το τελικό σίγμα: το «#σχεσεις»
    και το «#σχεσεισ» είναι δύο διαφορετικά hashtags, και το δεύτερο δεν
    υπάρχει. Η `normalize` (που κάνει ς→σ για ταίριασμα κειμένου) θα
    παρήγαγε άκυρα hashtags, γι' αυτό υπάρχει ξεχωριστή συνάρτηση.
    """
    if not text:
        return ""
    cleaned = strip_accents(text).lower()
    return "".join(c for c in cleaned if c.isalnum())


def word_tokens(text: str, min_len: int = 3) -> list:
    """Λέξεις με διατηρημένο τελικό σίγμα — για παραγωγή hashtags."""
    if not text:
        return []
    cleaned = MENTION_RE.sub(" ", HASHTAG_RE.sub(" ", text))
    cleaned = EMOJI_RE.sub(" ", cleaned)
    stop = {normalize(w) for w in STOPWORDS}
    out, seen = [], set()
    for m in _WORD.finditer(cleaned):
        raw = m.group(0)
        if len(raw) < min_len or normalize(raw) in stop:
            continue
        tag = normalize_tag(raw)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def tokenize(text: str, drop_stopwords: bool = True, min_len: int = 2) -> list:
    """Λέξεις κανονικοποιημένες, χωρίς hashtags/mentions/emoji."""
    if not text:
        return []
    cleaned = MENTION_RE.sub(" ", HASHTAG_RE.sub(" ", text))
    cleaned = EMOJI_RE.sub(" ", cleaned)
    toks = [normalize(m.group(0)) for m in _WORD.finditer(cleaned)]
    out = []
    for t in toks:
        if len(t) < min_len:
            continue
        if drop_stopwords and t in {normalize(s) for s in STOPWORDS}:
            continue
        out.append(t)
    return out


def char_ngrams(text: str, n: int = 4) -> list:
    """Χαρακτηρο-n-grams — ανθεκτικά στην ελληνική κλίση.

    «καφέδες» και «καφέ» μοιράζονται n-grams· με word-matching θα ήταν
    δύο άσχετα tokens. Αυτό μετράει πολύ σε γλώσσα με πλούσια μορφολογία.
    """
    collapsed = _WS.sub(" ", normalize(text))
    s = "  " + collapsed + "  "
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]


def greek_ratio(text: str) -> float:
    """Ποσοστό ελληνικών γραμμάτων επί των αλφαβητικών χαρακτήρων."""
    if not text:
        return 0.0
    gr = len(GREEK_RANGE.findall(text))
    la = len(LATIN_RANGE.findall(text))
    total = gr + la
    return gr / total if total else 0.0


def greeklish_score(text: str) -> float:
    """0–1: πόσο πιθανό είναι το λατινικό κείμενο να είναι ελληνικά."""
    if not text:
        return 0.0
    words = {normalize(m.group(0)) for m in _WORD.finditer(text)}
    if not words:
        return 0.0
    hits = len(words & GREEKLISH_MARKERS)
    return min(1.0, hits / max(3.0, len(words) * 0.25))


def greek_confidence(text: str, extra_signals: Iterable = ()) -> float:
    """
    Συνολική βεβαιότητα 0–1 ότι το περιεχόμενο απευθύνεται σε ελληνικό κοινό.

    Συνδυάζει: ελληνικό αλφάβητο, greeklish, και εξωτερικά σήματα
    (ελληνικό hashtag, ελληνική τοποθεσία, ελληνικό όνομα λογαριασμού).
    """
    base = greek_ratio(text)
    if base >= 0.30:
        score = 0.60 + min(0.40, base * 0.45)
    else:
        score = greeklish_score(text) * 0.55
    for sig in extra_signals:
        if sig:
            score = min(1.0, score + 0.12)
    return round(min(1.0, score), 3)


def extract_hashtags(text: str) -> list:
    """Hashtags με τη σειρά εμφάνισης, χωρίς δίεση, πεζά, χωρίς διπλά."""
    seen, out = set(), []
    for m in HASHTAG_RE.finditer(text or ""):
        tag = m.group(1).lower().rstrip(".")
        if not tag or tag.isdigit() or len(tag) > 60:
            continue
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def caption_body(text: str) -> str:
    """Η λεζάντα χωρίς το «σύννεφο» hashtags — μόνο το κείμενο που διαβάζεται."""
    if not text:
        return ""
    body = HASHTAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", body).strip()


def count_emoji(text: str) -> int:
    return sum(len(m.group(0)) for m in EMOJI_RE.finditer(text or ""))


def is_greek_hashtag(tag: str) -> bool:
    """Ελληνικό hashtag: ελληνικοί χαρακτήρες ή γνωστός ελληνικός λατινικός όρος."""
    if GREEK_RANGE.search(tag):
        return True
    t = tag.lower()
    return any(k in t for k in (
        "greek", "greece", "ellada", "hellas", "athens", "athina", "thessaloniki",
        "salonica", "cyprus", "kypros", "gr_", "_gr", "grstyle", "greekgirl",
        "greekboy", "greeks",
    ))
