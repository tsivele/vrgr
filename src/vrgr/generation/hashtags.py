"""
Μηχανή hashtags: ΧΑΡΤΟΦΥΛΑΚΙΟ, όχι λίστα.

Το λάθος που κάνουν όλα τα εργαλεία: διαλέγουν «τα καλύτερα 15 hashtags».
Αυτό δεν έχει νόημα, γιατί τα hashtags δεν είναι ανεξάρτητες επιλογές —
είναι κατανομή ρίσκου, ακριβώς όπως ένα χαρτοφυλάκιο επενδύσεων:

  #ελλαδα (95M posts)      → σχεδόν μηδενική πιθανότητα κατάταξης, αλλά
                             τεράστιο ταβάνι αν πετύχει                  → λοταρία
  #ελληνικοχιουμορ (85K)   → ρεαλιστική πιθανότητα κατάταξης              → ο στόχος
  #ζευγαρακια_αθηνα (900)  → σχεδόν σίγουρη κατάταξη, μικρό κοινό         → σίγουρο

Ένα σετ μόνο από πλατιά = μηδέν εμφανίσεις. Ένα σετ μόνο από στενά =
εμφανίσεις σε 200 άτομα. Η αξία είναι στη ΣΥΝΘΕΣΗ.

Τα βάρη του σκορ κάθε υποψηφίου δίνουν προτεραιότητα στα ΤΕΚΜΗΡΙΑ
(σε πόσα επιτυχημένα ελληνικά posts εμφανίστηκε πραγματικά) πάνω από
τη γενική δημοτικότητα.
"""
from __future__ import annotations

import math
from typing import Optional

from .. import greek as G
from ..logging_setup import get_logger
from ..research.collector import classify_tier
from ..schemas import (HashtagCandidate, HashtagSet, MinedPatterns,
                       VideoContent, ViralAngle)

log = get_logger("generation.hashtags")

# Σύνθεση-στόχος ανά στρατηγική: tier → (ελάχιστα, μέγιστα)
PORTFOLIOS = {
    "ισορροπημένο": {
        "broad": (1, 2), "mid": (3, 4), "niche": (4, 6), "micro": (2, 3),
        "_περιγραφή": "Κατανομή ρίσκου σε όλα τα επίπεδα. Η προεπιλογή.",
        "_greek_min": 0.55},
    "ανακάλυψη": {
        "broad": (0, 1), "mid": (2, 3), "niche": (5, 7), "micro": (3, 5),
        "_περιγραφή": "Βάρος στα στενά — μέγιστη πιθανότητα πραγματικής κατάταξης. "
                      "Για μικρούς λογαριασμούς είναι συνήθως η σωστή επιλογή.",
        "_greek_min": 0.60},
    "εμβέλεια": {
        "broad": (2, 3), "mid": (4, 6), "niche": (3, 4), "micro": (1, 2),
        "_περιγραφή": "Βάρος στα πλατιά. Λογικό μόνο όταν το περιεχόμενο έχει "
                      "ήδη αποδεδειγμένα υψηλό retention."},
    "ελληνική_στόχευση": {
        "broad": (1, 2), "mid": (3, 4), "niche": (4, 5), "micro": (2, 3),
        "_περιγραφή": "Ίδια κατανομή με το ισορροπημένο αλλά με απαίτηση "
                      "τουλάχιστον 70% ελληνικά hashtags.", "_greek_min": 0.70},
}

MIN_SET, MAX_SET = 8, 18

# Hashtags άλλων πλατφορμών. Σε Instagram Reel δεν κάνουν τίποτα — απλώς
# καταλαμβάνουν θέση και δηλώνουν ερασιτεχνισμό.
CROSS_PLATFORM = {
    "tiktok", "tiktokviral", "viraltiktok", "tiktokgreece", "foryoutiktok",
    "youtube", "youtubeshorts", "shorts", "twitter", "xtwitter", "facebook",
    "snapchat", "pinterest", "capcut", "duet", "stitch",
}

# Κενά «σήματα» χωρίς κοινό-στόχο: υπάρχουν σε δεκάδες εκατομμύρια posts
# χωρίς καμία θεματική συνοχή, άρα μηδενική πιθανότητα κατάταξης και μηδενική
# πληροφορία για τον αλγόριθμο.
EMPTY_SIGNALS = {
    "viral", "viralpost", "viralvideo", "trending", "trend", "explore",
    "explorepage", "foryou", "foryoupage", "follow", "followme", "like4like",
    "likeforlike", "l4l", "f4f", "instagood", "instadaily", "photooftheday",
}


# Ονόματα πλατφορμών ως ΥΠΟΣΥΜΒΟΛΟΣΕΙΡΕΣ. Η ακριβής αντιστοίχιση δεν αρκεί:
# σε πραγματική εκτέλεση πέρασε το «#youtubeshortsελληνικα», που είναι το ίδιο
# άχρηστο σε Instagram με το «#youtubeshorts».
PLATFORM_MARKERS = ("tiktok", "youtube", "yt shorts", "ytshorts", "twitter",
                    "facebook", "snapchat", "pinterest", "capcut", "threads",
                    "onlyfans", "twitch")


def is_cross_platform(tag: str) -> bool:
    t = tag.lower()
    if t in CROSS_PLATFORM:
        return True
    return any(marker.replace(" ", "") in t for marker in PLATFORM_MARKERS)


def is_branded(tag: str, creator_usernames: set) -> bool:
    """
    Προσωπικό hashtag άλλου creator.

    Μετρήθηκε σε πραγματική εκτέλεση: η ανάλυση ανύψωσης πρότεινε
    «#chryssanthemis» και «#singers» επειδή ο ίδιος creator εμφανιζόταν σε
    πολλά outliers. Το προσωπικό brand hashtag κάποιου ΑΛΛΟΥ δεν πρόκειται
    ποτέ να φέρει κοινό σε εσένα — και δείχνει ότι αντέγραψες τη λίστα του.
    """
    t = tag.lower()
    for u in creator_usernames:
        u = u.lower().replace("_", "").replace(".", "")
        if len(u) < 5:
            continue
        if t == u or (len(t) >= 6 and (t in u or u in t)):
            return True
    return False


def build_candidates(content: VideoContent, angle: Optional[ViralAngle],
                     mined: Optional[MinedPatterns], hashtag_stats: dict,
                     memory_tags: Optional[list] = None,
                     planner_terms: Optional[dict] = None,
                     repo=None, niche: str = "",
                     creator_usernames: Optional[set] = None) -> list:
    """
    Συγκέντρωση υποψηφίων από ΟΛΕΣ τις πηγές, με σκορ ανά υποψήφιο.

    Πηγές κατά σειρά αξιοπιστίας:
      1. Ανύψωση από φρέσκα δεδομένα HikerAPI  (ισχυρότερο — μετρήθηκε τώρα)
      2. Μνήμη προηγούμενων αναλύσεων           (αποδεδειγμένο διαχρονικά)
      3. Όροι του planner                        (σχετικό αλλά ατεκμηρίωτο)
      4. Παράγωγα από το περιεχόμενο             (σχετικό, καθόλου τεκμηριωμένο)
    """
    pool: dict = {}
    creators = creator_usernames or set()
    rejected: list = []

    def add(tag: str, source: str, relevance: float, evidence: int = 0,
            reason: str = "") -> None:
        tag = G.normalize_tag(tag.lstrip("#"))
        if not tag or len(tag) < 2 or len(tag) > 40 or tag.isdigit():
            return
        if is_cross_platform(tag):
            rejected.append((tag, "άλλη πλατφόρμα"))
            return
        if tag in EMPTY_SIGNALS:
            rejected.append((tag, "κενό σήμα"))
            return
        if is_branded(tag, creators):
            rejected.append((tag, "προσωπικό brand άλλου creator"))
            return
        existing = pool.get(tag)
        if existing:
            existing.relevance = max(existing.relevance, relevance)
            existing.evidence_count = max(existing.evidence_count, evidence)
            if reason and reason not in existing.reason:
                existing.reason = f"{existing.reason}· {reason}".strip("· ")
            return
        pool[tag] = HashtagCandidate(
            tag=tag, relevance=relevance, evidence_count=evidence,
            is_greek=G.is_greek_hashtag(tag), category=source, reason=reason)

    # 1) Φρέσκα δεδομένα — ανύψωση
    if mined:
        for tag, lift, n in (mined.top_hashtags or []):
            rel = min(1.0, 0.55 + 0.15 * math.log1p(max(0.0, lift - 1.0)))
            add(tag, "content-specific", rel, n,
                f"ανύψωση {lift}× σε {n} επιτυχημένα ελληνικά Reels")

    # 2) Μνήμη
    for tag, n, avg in (memory_tags or []):
        add(tag, "community", min(1.0, 0.50 + 0.04 * n), n,
            f"{n} επιτυχημένα posts στη μνήμη (μέσο outlier {avg})")

    # 3) Planner
    terms = planner_terms or {}
    for key, rel, cat in (("hashtags_niche", 0.72, "micro-niche"),
                          ("hashtags_mid", 0.62, "topic"),
                          ("hashtags_broad", 0.42, "broad")):
        for tag in terms.get(key, []):
            add(tag, cat, rel, 0, "από σχεδιασμό έρευνας")

    # 4) Παράγωγα από το περιεχόμενο
    for word in _content_terms(content, angle):
        add(word, "topic", 0.50, 0, "από το περιεχόμενο του βίντεο")

    # Εμπλουτισμός με OBSERVED μεγέθη + DERIVED δυσκολία
    for tag, cand in pool.items():
        stat = hashtag_stats.get(tag)
        if stat is None and repo is not None:
            cached = repo.get_hashtag_stat(tag)
            if cached:
                cand.media_count = cached.get("media_count")
                cand.difficulty = cached.get("difficulty")
                cand.tier = cached.get("tier") or classify_tier(cand.media_count)
        elif stat is not None:
            cand.media_count = stat.media_count
            cand.difficulty = stat.difficulty
            cand.tier = stat.tier or classify_tier(stat.media_count)
        if cand.tier in ("", "unknown"):
            cand.tier = classify_tier(cand.media_count)
        if repo is not None and not cand.evidence_count:
            cand.evidence_count = repo.hashtag_evidence_count(tag, niche)
        cand.score = score_candidate(cand)

    ranked = sorted(pool.values(), key=lambda c: -c.score)
    log.info("Υποψήφια hashtags: %d (με μέγεθος από API: %d· απορρίφθηκαν %d)",
             len(ranked), sum(1 for c in ranked if c.media_count is not None),
             len(rejected))
    if rejected:
        log.debug("Απορρίφθηκαν: %s", ", ".join(f"#{t} ({r})" for t, r in rejected[:12]))
    return ranked


def _content_terms(content: VideoContent, angle: Optional[ViralAngle]) -> list:
    """Ουσιαστικοί όροι από την ανάλυση — τελευταία γραμμή, χαμηλή αξιοπιστία."""
    blob = " ".join(filter(None, [
        content.niche, content.sub_niche, content.main_subject,
        content.mood, content.aesthetic,
        " ".join(content.cultural_markers or []),
        " ".join(content.actions or [])[:120],
        angle.name if angle else "",
    ]))
    # `word_tokens` και όχι `tokenize`: το τελικό σίγμα πρέπει να διατηρηθεί,
    # αλλιώς παράγονται ανύπαρκτα hashtags («#σχεσεισ»).
    return [t for t in G.word_tokens(blob) if 4 <= len(t) <= 22][:12]


def score_candidate(c: HashtagCandidate) -> float:
    """
    Σκορ 0-100 ενός hashtag.

    Η βαρύτητα των τεκμηρίων (35%) είναι σκόπιμα υψηλότερη από τη
    δημοτικότητα: προτιμούμε ένα tag που εμφανίστηκε σε 6 πραγματικά
    ελληνικά viral από ένα με 40 εκατομμύρια posts και καμία απόδειξη
    ότι δουλεύει για μας.
    """
    relevance = c.relevance * 100.0
    evidence = min(100.0, 22.0 * math.log1p(c.evidence_count) / math.log1p(3))
    greek = 100.0 if c.is_greek else 45.0
    if c.difficulty is not None:
        # Καμπύλη με κορυφή γύρω στο 45: ούτε αδιάφορο, ούτε άπιαστο.
        openness = 100.0 * math.exp(-((c.difficulty - 45.0) ** 2) / (2 * 26.0 ** 2))
    elif c.media_count is not None:
        openness = {"micro": 80.0, "niche": 95.0,
                    "mid": 70.0, "broad": 30.0}.get(c.tier, 55.0)
    else:
        openness = 50.0            # άγνωστο → ουδέτερο, χωρίς μπόνους
    total = (0.32 * relevance + 0.35 * evidence + 0.18 * openness + 0.15 * greek)
    return round(total, 1)


def build_sets(candidates: list, strategies: Optional[list] = None,
               mined: Optional[MinedPatterns] = None,
               target_size: int = 14) -> list:
    """Ένα σετ ανά στρατηγική χαρτοφυλακίου."""
    strategies = strategies or list(PORTFOLIOS.keys())
    by_tier: dict = {"broad": [], "mid": [], "niche": [], "micro": [], "unknown": []}
    for c in candidates:
        by_tier.setdefault(c.tier or "unknown", by_tier["unknown"]).append(c)
    for group in by_tier.values():
        group.sort(key=lambda c: -c.score)

    cooc = {tuple(p["pair"]): p["count"] for p in (mined.hashtag_cooccurrence or [])} \
        if mined else {}

    sets = []
    for name in strategies:
        spec = PORTFOLIOS.get(name)
        if not spec:
            continue
        chosen = _fill_portfolio(by_tier, spec, target_size, cooc)
        if len(chosen) < MIN_SET:
            continue
        sets.append(_make_set(chosen, name, spec))
    sets.sort(key=lambda s: -s.score)
    return sets


def _fill_portfolio(by_tier: dict, spec: dict, target: int, cooc: dict) -> list:
    greek_min = spec.get("_greek_min", 0.0)
    chosen: list = []
    used = set()

    for tier in ("niche", "micro", "mid", "broad"):
        lo, hi = spec.get(tier, (0, 0))
        pool = [c for c in by_tier.get(tier, []) if c.tag not in used]
        if greek_min:
            pool.sort(key=lambda c: (-int(c.is_greek), -c.score))
        take = pool[:hi]
        for c in take:
            chosen.append(c)
            used.add(c.tag)

    # Συμπλήρωση από «άγνωστο μέγεθος».
    #
    # ΦΡΑΓΗ ΥΠΟ ΟΡΟΥΣ: τα tags χωρίς μετρημένο μέγεθος δεν επιτρέπεται να
    # ΕΚΤΟΠΙΣΟΥΝ μετρημένα — αλλά δεν επιτρέπεται και να ΜΠΛΟΚΑΡΟΥΝ τη
    # λειτουργία. Αν το HikerAPI δεν απάντησε καθόλου, όλα τα υποψήφια είναι
    # «άγνωστα»· τότε η φραγή θα παρήγαγε κενό σετ, δηλαδή θα μετέτρεπε μια
    # υποβαθμισμένη λειτουργία σε ολική αποτυχία.
    #
    # Κανόνας: η φραγή 35% ισχύει ΜΟΝΟ όταν έχουμε ήδη αρκετά μετρημένα tags
    # για ένα βιώσιμο σετ. Αλλιώς τα άγνωστα συμπληρώνουν ελεύθερα και το
    # σετ σημαίνεται ρητά ως λιγότερο τεκμηριωμένο.
    measured_taken = len(chosen)
    unknown_cap = (max(2, int(target * 0.35)) if measured_taken >= MIN_SET
                   else target)
    unknown_taken = 0
    if len(chosen) < target:
        # Πρώτα όσα έχουν τουλάχιστον τεκμήριο από πραγματικό post.
        pool = sorted(by_tier.get("unknown", []),
                      key=lambda c: (-int(c.evidence_count > 0), -c.score))
        for c in pool:
            if c.tag in used or unknown_taken >= unknown_cap:
                continue
            chosen.append(c)
            used.add(c.tag)
            unknown_taken += 1
            if len(chosen) >= target:
                break

    # Μπόνους συνεμφάνισης: tags που οι επιτυχημένοι creators βάζουν ΜΑΖΙ
    if cooc:
        tagset = {c.tag for c in chosen}
        for c in chosen:
            partners = sum(n for (a, b), n in cooc.items()
                           if (a == c.tag and b in tagset) or (b == c.tag and a in tagset))
            if partners:
                c.score = round(c.score + min(8.0, 2.0 * partners), 1)
                c.reason = f"{c.reason}· εμφανίζεται μαζί με άλλα επιλεγμένα".strip("· ")

    chosen.sort(key=lambda c: -c.score)
    if greek_min:
        # Πάτωμα ελληνικών: το προϊόν στοχεύει ελληνικό κοινό, οπότε ένα σετ
        # με 38% ελληνικά hashtags δουλεύει εναντίον του σκοπού του. Αν όμως
        # δεν υπάρχουν αρκετά ελληνικά υποψήφια, ΔΕΝ κόβουμε το σετ —
        # συμπληρώνουμε με τα καλύτερα διαθέσιμα και το σκορ το καταγράφει.
        greek = [c for c in chosen if c.is_greek]
        other = [c for c in chosen if not c.is_greek]
        need_greek = min(len(greek), int(math.ceil(target * greek_min)))
        chosen = greek[:need_greek] + other[:max(0, target - need_greek)]
        chosen.sort(key=lambda c: -c.score)
    return chosen[:min(target, MAX_SET)]


def _make_set(chosen: list, name: str, spec: dict) -> HashtagSet:
    dist: dict = {}
    for c in chosen:
        dist[c.tier] = dist.get(c.tier, 0) + 1
    greek_share = sum(1 for c in chosen if c.is_greek) / len(chosen)
    evidence_share = sum(1 for c in chosen if c.evidence_count > 0) / len(chosen)
    measured_share = sum(1 for c in chosen if c.media_count is not None) / len(chosen)
    base = sum(c.score for c in chosen) / len(chosen)
    # Επιβράβευση πραγματικής διασποράς επιπέδων: ένα σετ με 4 επίπεδα
    # καλύπτει περισσότερα σενάρια από ένα με 1.
    spread = min(1.0, len([t for t, n in dist.items() if n and t != "unknown"]) / 4.0)
    score = round(base * (0.80 + 0.12 * spread + 0.08 * evidence_share), 1)
    rationale = spec.get("_περιγραφή", "")
    if measured_share < 0.4:
        rationale += (f"  ⚠ Μόνο για {measured_share:.0%} των hashtags μετρήθηκε "
                      f"πραγματικό μέγεθος — η κατανομή επιπέδων είναι εκτίμηση.")
    return HashtagSet(
        tags=[c.tag for c in chosen], candidates=chosen, strategy=name,
        tier_distribution=dist, greek_share=round(greek_share, 3),
        evidence_share=round(evidence_share, 3), score=score,
        rationale=rationale,
    ).fill()


def format_for_instagram(tags: list, on_new_line: bool = True) -> str:
    """Έτοιμο για επικόλληση κάτω από τη λεζάντα."""
    line = " ".join(f"#{t}" for t in tags)
    return ("\n\n" + line) if on_new_line else line
