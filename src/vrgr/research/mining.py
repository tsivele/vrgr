"""
Εξόρυξη μοτίβων από corpus πραγματικών posts (απαίτηση #15).

ΜΕΘΟΔΟΣ: ανάλυση ΑΝΥΨΩΣΗΣ (lift), όχι απλή συχνότητα.

Η συχνότητα λέει ψέματα. Η λέξη «μου» εμφανίζεται στο 60% των επιτυχημένων
ελληνικών λεζαντών — και στο 60% των αποτυχημένων. Δεν εξηγεί τίποτα.

Η ανύψωση συγκρίνει δύο ομάδες:
    ΟΜΑΔΑ Α: outliers          (post που ξέφυγαν)
    ΟΜΑΔΑ Β: όλα τα υπόλοιπα   (η βάση)

    lift = P(χαρακτηριστικό | outlier) / P(χαρακτηριστικό | βάση)

lift = 2.4 σημαίνει «αυτό εμφανίζεται 2,4 φορές συχνότερα στα επιτυχημένα».
Αυτό είναι μάθημα. Το «εμφανίζεται 60 φορές» δεν είναι.

Επιπλέον εφαρμόζεται ΕΞΟΜΑΛΥΝΣΗ και ελάχιστο πλήθος: χωρίς αυτά, ένα
χαρακτηριστικό που εμφανίζεται 2 φορές στα outliers και 0 στη βάση θα
έβγαζε άπειρο lift και θα κυριαρχούσε λανθασμένα.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Optional

from .. import greek as G
from ..logging_setup import get_logger
from ..schemas import MinedPatterns, ObservedPost

log = get_logger("research.mining")

MIN_COUNT_OUTLIER = 3        # ελάχιστες εμφανίσεις στα outliers για να μετρήσει
SMOOTHING = 0.5              # εξομάλυνση Laplace

CTA_PATTERNS = [
    (r"\bσχολ[ιί]ασ", "ζητά σχόλιο"),
    (r"\bγρ[άα]ψ", "ζητά γραπτή απάντηση"),
    (r"\bπες\s+μου\b", "ζητά απάντηση"),
    (r"\bστε[ίι]λ", "ζητά αποστολή"),
    (r"\bκ[άα]νε\s+tag\b|\btag\b|\bταγκ[άα]ρ", "ζητά tag φίλου"),
    (r"\bαποθ[ήη]κευσ|\bσ[ώω]σε\b|\bsave\b", "ζητά αποθήκευση"),
    (r"\bακολο[υύ]θησ|\bfollow\b", "ζητά follow"),
    (r"\bσυμφωνε[ίι]τε\b|\bποιος\s+[άα]λλος\b|\bποια\s+[άα]λλη\b", "ζητά ταύτιση"),
    (r"\bλ[ίι]νκ\b|\blink\s+in\s+bio\b|\bστο\s+bio\b", "παραπέμπει στο bio"),
]
FIRST_PERSON = re.compile(r"\b(εγ[ώω]|μου|μας|[έε]χω|ε[ίι]μαι|νι[ώω]θω|θυμ[άα]μαι)\b")
SECOND_PERSON = re.compile(r"\b(εσ[ύυ]|σου|σας|[έε]χεις|ε[ίι]σαι|ξ[έε]ρεις|θυμ[άα]σαι|σκ[έε]ψου)\b")


def _has_cta(text: str) -> Optional[str]:
    low = text.lower()
    for pattern, label in CTA_PATTERNS:
        if re.search(pattern, low):
            return label
    return None


def _lift(feature_counts_a: Counter, n_a: int,
          feature_counts_b: Counter, n_b: int,
          min_count: int = MIN_COUNT_OUTLIER, limit: int = 25) -> list:
    """[(χαρακτηριστικό, lift, εμφανίσεις_στα_outliers)] ταξινομημένα."""
    if n_a == 0 or n_b == 0:
        return []
    out = []
    for feature, count_a in feature_counts_a.items():
        if count_a < min_count:
            continue
        p_a = (count_a + SMOOTHING) / (n_a + 2 * SMOOTHING)
        p_b = (feature_counts_b.get(feature, 0) + SMOOTHING) / (n_b + 2 * SMOOTHING)
        lift = p_a / p_b if p_b > 0 else 1.0
        # Ποινή σε σπάνια χαρακτηριστικά: εμπιστευόμαστε λιγότερο το lift τους.
        confidence = min(1.0, math.log1p(count_a) / math.log1p(12))
        out.append((feature, round(lift * confidence + (1 - confidence), 2), count_a))
    out.sort(key=lambda t: -t[1])
    return out[:limit]


def _features(post: ObservedPost) -> dict:
    body = post.caption_body
    tokens = G.tokenize(body, drop_stopwords=True)
    return {
        "words": set(t for t in tokens if len(t) >= 3),
        "bigrams": set(f"{a} {b}" for a, b in zip(tokens, tokens[1:])),
        "hashtags": set(post.hashtags),
        "length": len(body),
        "emoji": G.count_emoji(post.caption),
        "n_hashtags": len(post.hashtags),
        "question": ("?" in body) or (";" in body),
        "cta": _has_cta(body),
        "first_person": bool(FIRST_PERSON.search(body.lower())),
        "second_person": bool(SECOND_PERSON.search(body.lower())),
        "hour": (post.taken_at // 3600 % 24) if post.taken_at else None,
        "duration": post.duration_s,
    }


def mine(posts: list, outliers: list,
         greek_only: bool = True) -> MinedPatterns:
    """Σύγκριση outliers vs βάσης → μοτίβα με ανύψωση."""
    if greek_only:
        posts = [p for p in posts if p.greek_confidence >= 0.5]
        outliers = [p for p in outliers if p.greek_confidence >= 0.5]

    outlier_ids = {p.media_id for p in outliers}
    baseline = [p for p in posts if p.media_id not in outlier_ids]

    result = MinedPatterns(sample_size=len(posts), outlier_sample_size=len(outliers))
    if len(outliers) < 3:
        log.info("Πολύ λίγα outliers (%d) — δεν εξάγονται μοτίβα ανύψωσης",
                 len(outliers))
        if posts:
            _fill_distributions(result, posts)
        return result

    feats_o = [_features(p) for p in outliers]
    feats_b = [_features(p) for p in baseline] or feats_o

    n_o, n_b = len(feats_o), len(feats_b)
    # Προσαρμοστικό κατώφλι: με 14 outliers, η απαίτηση «3 εμφανίσεις» αφήνει
    # σχεδόν τίποτα, γιατί επιτυχημένες λεζάντες σπάνια μοιράζονται λέξεις.
    # Κλιμακώνεται με το δείγμα ώστε να μη χαλαρώνει όταν υπάρχουν δεδομένα.
    min_count = max(2, n_o // 7)
    for key, target in (("words", "top_words"), ("bigrams", "top_bigrams"),
                        ("hashtags", "top_hashtags")):
        ca = Counter(f for feat in feats_o for f in feat[key])
        cb = Counter(f for feat in feats_b for f in feat[key])
        setattr(result, target, _lift(ca, n_o, cb, n_b, min_count=min_count))

    _fill_distributions(result, outliers)

    # Δομικά χαρακτηριστικά: ποσοστά ΣΤΑ OUTLIERS (αυτό μιμείται ο generator)
    result.question_share = round(
        sum(1 for f in feats_o if f["question"]) / n_o, 3)
    result.cta_share = round(sum(1 for f in feats_o if f["cta"]) / n_o, 3)
    result.first_person_share = round(
        sum(1 for f in feats_o if f["first_person"]) / n_o, 3)
    result.second_person_share = round(
        sum(1 for f in feats_o if f["second_person"]) / n_o, 3)

    result.hashtag_cooccurrence = _cooccurrence(outliers)
    result.greek_expressions = _greek_expressions(outliers, baseline)
    result.posting_hours = _posting_hours(outliers)
    result.duration_sweet_spot = _duration_range(outliers)
    log.info("Μοτίβα: %d λέξεις, %d hashtags, %d ζεύγη — από %d outliers",
             len(result.top_words), len(result.top_hashtags),
             len(result.hashtag_cooccurrence), n_o)
    return result


def _fill_distributions(result: MinedPatterns, posts: list) -> None:
    lengths = [len(p.caption_body) for p in posts if p.caption_body]
    if lengths:
        lengths.sort()
        result.caption_len_median = float(statistics.median(lengths))
        result.caption_len_p25 = float(lengths[len(lengths) // 4])
        result.caption_len_p75 = float(lengths[(3 * len(lengths)) // 4])
    emojis = [G.count_emoji(p.caption) for p in posts]
    if emojis:
        result.emoji_median = float(statistics.median(emojis))
    tags = [len(p.hashtags) for p in posts]
    if tags:
        result.hashtag_count_median = float(statistics.median(tags))


def _cooccurrence(posts: list, min_pairs: int = 2, limit: int = 20) -> list:
    """
    Ζεύγη hashtags που εμφανίζονται ΜΑΖΙ σε επιτυχημένα posts.

    Πιο χρήσιμο από τη δημοτικότητα ενός μεμονωμένου tag: δείχνει πώς
    συνθέτουν οι creators που πετυχαίνουν, όχι απλώς τι είναι δημοφιλές.
    """
    pairs = Counter()
    for p in posts:
        tags = sorted(set(p.hashtags))[:30]
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                pairs[(a, b)] += 1
    return [{"pair": [a, b], "count": n}
            for (a, b), n in pairs.most_common(limit) if n >= min_pairs]


def _greek_expressions(outliers: list, baseline: list, limit: int = 15) -> list:
    """
    Χαρακτηριστικές ελληνικές εκφράσεις των επιτυχημένων.

    Ψάχνουμε φράσεις 2-4 λέξεων ΧΩΡΙΣ αφαίρεση stopwords — γιατί εδώ
    ακριβώς η καθομιλουμένη ζει: «ρε συ», «δεν παίζεσαι», «μου την έδωσε».
    """
    def phrases(posts):
        c = Counter()
        for p in posts:
            toks = G.tokenize(p.caption_body, drop_stopwords=False, min_len=1)
            for n in (2, 3, 4):
                for i in range(len(toks) - n + 1):
                    ph = " ".join(toks[i:i + n])
                    if 6 <= len(ph) <= 45:
                        c[ph] += 1
        return c

    co, cb = phrases(outliers), phrases(baseline)
    ranked = _lift(co, len(outliers) or 1, cb, len(baseline) or 1,
                   min_count=2, limit=limit * 3)
    seen, out = set(), []
    for phrase, lift, n in ranked:
        # Απόρριψη φράσεων που περιέχονται σε ήδη επιλεγμένη (θόρυβος)
        if any(phrase in s for s in seen):
            continue
        seen.add(phrase)
        out.append({"phrase": phrase, "lift": lift, "count": n})
        if len(out) >= limit:
            break
    return out


def _posting_hours(posts: list) -> list:
    """
    Ώρες δημοσίευσης — ΠΡΟΣΟΧΗ στην ερμηνεία.

    Το `taken_at` είναι UTC. Επιστρέφουμε ώρα Ελλάδας (UTC+3 θερινή /
    UTC+2 χειμερινή· χρησιμοποιούμε +3 ως προσέγγιση και το δηλώνουμε).
    Είναι ασθενές σήμα και σημαίνεται ως τέτοιο στο report.
    """
    hours = Counter(((p.taken_at // 3600 + 3) % 24) for p in posts if p.taken_at)
    return [{"hour_gr": h, "count": n} for h, n in hours.most_common(6)]


def _duration_range(posts: list) -> Optional[list]:
    durations = sorted(p.duration_s for p in posts if p.duration_s)
    if len(durations) < 4:
        return None
    lo = durations[len(durations) // 4]
    hi = durations[(3 * len(durations)) // 4]
    return [round(lo, 1), round(hi, 1)]


def patterns_to_observations(mined: MinedPatterns, niche: str) -> list:
    """
    Μετατροπή εξαγμένων μοτίβων σε παρατηρήσεις για το `PatternStore`.

    Έτσι κλείνει ο βρόχος: η εξόρυξη τροφοδοτεί τη μακροπρόθεσμη μνήμη,
    και ένα μοτίβο που επιβεβαιώνεται σε πολλές εκτελέσεις κερδίζει
    σταδιακά βεβαιότητα.
    """
    from ..memory.patterns import hashtag_key, structure_key
    obs = []
    for tag, lift, n in mined.top_hashtags[:20]:
        obs.append({"key": hashtag_key(tag), "kind": "hashtag",
                    "success": lift > 1.25, "niche": niche,
                    "description_el": f"#{tag} — ανύψωση {lift}× σε {n} επιτυχημένα posts",
                    "weight": min(1.0, n / 8.0)})
    if mined.question_share is not None and mined.outlier_sample_size >= 5:
        obs.append({"key": structure_key("ερώτηση"), "kind": "caption_structure",
                    "success": mined.question_share > 0.35, "niche": niche,
                    "description_el": f"Ερώτηση στη λεζάντα "
                                      f"({mined.question_share:.0%} των επιτυχημένων)",
                    "weight": min(1.0, mined.outlier_sample_size / 15.0)})
    if mined.cta_share is not None and mined.outlier_sample_size >= 5:
        obs.append({"key": structure_key("cta"), "kind": "caption_structure",
                    "success": mined.cta_share > 0.30, "niche": niche,
                    "description_el": f"Ρητό κάλεσμα δράσης "
                                      f"({mined.cta_share:.0%} των επιτυχημένων)",
                    "weight": min(1.0, mined.outlier_sample_size / 15.0)})
    if mined.second_person_share is not None and mined.outlier_sample_size >= 5:
        obs.append({"key": structure_key("β_πρόσωπο"), "kind": "caption_structure",
                    "success": mined.second_person_share > 0.40, "niche": niche,
                    "description_el": f"Απεύθυνση σε β' πρόσωπο "
                                      f"({mined.second_person_share:.0%} των επιτυχημένων)",
                    "weight": min(1.0, mined.outlier_sample_size / 15.0)})
    if mined.caption_len_median:
        from ..memory.patterns import length_bucket
        bucket = length_bucket(int(mined.caption_len_median))
        obs.append({"key": f"caption:length={bucket}", "kind": "caption_structure",
                    "success": True, "niche": niche,
                    "description_el": f"Μήκος λεζάντας «{bucket}» "
                                      f"(διάμεσο {int(mined.caption_len_median)} χαρακτ.)",
                    "weight": min(1.0, mined.outlier_sample_size / 15.0)})
    return obs
