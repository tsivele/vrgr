"""
Μηχανή Viral Score (απαίτηση #13).

ΔΥΟ ΜΗΧΑΝΙΣΜΟΙ ΚΑΝΟΥΝ ΤΟ ΣΚΟΡ ΕΙΛΙΚΡΙΝΕΣ — χωρίς αυτούς θα ήταν διακόσμηση:

1. EVIDENCE MULTIPLIER
   Το τελικό σκορ πολλαπλασιάζεται με 0,72-1,0 ανάλογα με το πόσα ΠΡΑΓΜΑΤΙΚΑ
   ελληνικά posts στήριξαν την απόφαση. Χωρίς δεδομένα HikerAPI το σύστημα
   ΔΕΝ ΜΠΟΡΕΙ να βγάλει 90. Αυτό είναι σκόπιμο: ένα υψηλό σκορ χωρίς τεκμήρια
   θα ήταν ισχυρισμός που δεν μπορούμε να στηρίξουμε.

2. ΔΙΑΣΤΗΜΑ ΕΜΠΙΣΤΟΣΥΝΗΣ
   Επιστρέφεται «78 (εύρος 68-85)», όχι «78,4». Η ψευδο-ακρίβεια σε πρόβλεψη
   συμπεριφοράς αλγορίθμου είναι παραπλανητική.

ΤΙ ΔΕΝ ΕΙΝΑΙ ΤΟ ΣΚΟΡ: πρόβλεψη προβολών. Είναι ΣΧΕΤΙΚΗ ΚΑΤΑΤΑΞΗ — «αυτός ο
συνδυασμός είναι ισχυρότερος από εκείνον, με βάση τα διαθέσιμα στοιχεία».
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Optional

from .. import greek as G
from ..logging_setup import get_logger
from ..schemas import (CaptionCandidate, HashtagSet, MinedPatterns, Origin,
                       PillarScore, ResearchBundle, VideoAnalysis, ViralAngle,
                       ViralScore)

log = get_logger("scoring")

DEFAULT_WEIGHTS = {
    "content_angle_fit": 0.18, "caption_strength": 0.20, "hashtag_strength": 0.13,
    "greek_audience_fit": 0.16, "historical_evidence": 0.14,
    "competitor_evidence": 0.11, "trend_signal": 0.08,
}
LABELS = {
    "content_angle_fit": "Ταίριασμα βίντεο–γωνίας",
    "caption_strength": "Δύναμη λεζάντας",
    "hashtag_strength": "Δύναμη hashtags",
    "greek_audience_fit": "Ταίριασμα με ελληνικό κοινό",
    "historical_evidence": "Ιστορικά τεκμήρια (μνήμη)",
    "competitor_evidence": "Τεκμήρια ανταγωνισμού (φρέσκα)",
    "trend_signal": "Σήμα τάσης",
}


def load_weights(config_dir: Path) -> dict:
    path = config_dir / "weights.json"
    if not path.is_file():
        return {"pillars": DEFAULT_WEIGHTS, "raw": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    pillars = {k: v["weight"] for k, v in (data.get("pillars") or {}).items()}
    total = sum(pillars.values()) or 1.0
    return {"pillars": {k: v / total for k, v in pillars.items()}, "raw": data}


class ViralScorer:
    def __init__(self, config_dir: Path):
        cfg = load_weights(config_dir)
        self.weights = cfg["pillars"] or DEFAULT_WEIGHTS
        self.raw = cfg["raw"]

    # ── πυλώνες ──────────────────────────────────────────────────────
    def _content_angle_fit(self, analysis: VideoAnalysis,
                           angle: ViralAngle) -> PillarScore:
        signals = analysis.signals
        # Ο μέσος όρος των σημάτων του βίντεο, βαρυμένος προς τη διατήρηση
        # προσοχής: αν ο χρήστης δεν μείνει στα πρώτα 3 δευτ., τίποτα άλλο
        # δεν προλαβαίνει να μετρήσει.
        base = (0.30 * signals.attention_hold + 0.20 * signals.curiosity_gap +
                0.20 * signals.relatability + 0.15 * signals.shareability +
                0.15 * signals.rewatch_potential)
        raw = 0.60 * base + 0.40 * angle.strength
        note = ""
        # Ποινή όταν η γωνία απλώς περιγράφει το βίντεο.
        overlap = G.char_ngrams(angle.caption_should_add or "", 4)
        summary_grams = set(G.char_ngrams(analysis.content.summary or "", 4))
        if overlap and summary_grams:
            shared = len(set(overlap) & summary_grams) / len(set(overlap))
            if shared > 0.45:
                raw *= 0.80
                note = ("Η γωνία επικαλύπτεται πολύ με την περιγραφή του βίντεο — "
                        "κίνδυνος η λεζάντα να επαναλαμβάνει αντί να προσθέτει.")
        return PillarScore(name="content_angle_fit", label_el=LABELS["content_angle_fit"],
                           raw=_clamp(raw), weight=self.weights.get("content_angle_fit", 0),
                           weighted=0.0, origin=Origin.INFERRED, note=note)

    def _caption_strength(self, caption: CaptionCandidate, angle: ViralAngle,
                          mined: Optional[MinedPatterns],
                          memory_support: Optional[dict]) -> PillarScore:
        score = 50.0
        notes = []
        n_evidence = 0

        # (α) Μήκος έναντι ΜΕΤΡΗΜΕΝΟΥ εύρους των επιτυχημένων
        if mined and mined.caption_len_p25 and mined.caption_len_p75:
            lo, hi = mined.caption_len_p25, mined.caption_len_p75
            n_evidence += mined.outlier_sample_size
            length = caption.length_chars
            if lo <= length <= hi:
                score += 12.0
            else:
                distance = (lo - length) if length < lo else (length - hi)
                span = max(20.0, hi - lo)
                score += max(-10.0, 12.0 - 16.0 * (distance / span))
                notes.append(
                    f"Μήκος {length} χαρ. εκτός του εύρους των επιτυχημένων "
                    f"({int(lo)}-{int(hi)}).")
        else:
            # Χωρίς δεδομένα: ήπια προτίμηση σε 60-220 χαρακτήρες.
            if 60 <= caption.length_chars <= 220:
                score += 6.0

        # (β) Δομικά στοιχεία, βαρυμένα από το ΠΟΣΟ συχνά εμφανίζονται στα viral
        if mined and mined.outlier_sample_size >= 5:
            if caption.has_question:
                score += 14.0 * (mined.question_share or 0.3)
            if caption.has_cta:
                score += 12.0 * (mined.cta_share or 0.3)
        else:
            score += 5.0 if caption.has_question else 0.0
            score += 4.0 if caption.has_cta else 0.0

        # (γ) Ταίριασμα στρατηγικής με τη γωνία
        if caption.strategy and angle.strategy:
            if G.normalize(caption.strategy) == G.normalize(angle.strategy):
                score += 7.0
            else:
                score += 2.0

        # (δ) Μοτίβα μνήμης
        if memory_support and memory_support.get("score"):
            score += 14.0 * memory_support["score"] * min(1.0, memory_support.get("coverage", 0))
            n_evidence += sum(int(p.get("n", 0)) for p in memory_support.get("patterns", []))

        # (ε) Ποινή λογοκλοπής (κάτω από το κατώφλι απόρριψης, αλλά ύποπτη)
        if caption.similarity_to_corpus > 0.42:
            penalty = 22.0 * (caption.similarity_to_corpus - 0.42) / 0.20
            score -= penalty
            notes.append(f"Μοιάζει με υπάρχουσα λεζάντα (ομοιότητα "
                         f"{caption.similarity_to_corpus}) — μειώθηκε το σκορ.")

        # (στ) Emoji σύμφωνα με το μετρημένο διάμεσο
        if mined and mined.emoji_median is not None:
            if abs(caption.emoji_count - mined.emoji_median) <= 2:
                score += 4.0
            elif caption.emoji_count > mined.emoji_median + 4:
                score -= 5.0
                notes.append("Περισσότερα emoji από ό,τι συνηθίζουν τα επιτυχημένα.")

        return PillarScore(name="caption_strength", label_el=LABELS["caption_strength"],
                           raw=_clamp(score), weight=self.weights.get("caption_strength", 0),
                           weighted=0.0, evidence_n=n_evidence,
                           origin=Origin.DERIVED if n_evidence else Origin.INFERRED,
                           note=" ".join(notes))

    def _hashtag_strength(self, hset: HashtagSet) -> PillarScore:
        if not hset.tags:
            return PillarScore(name="hashtag_strength", label_el=LABELS["hashtag_strength"],
                               raw=0.0, weight=self.weights.get("hashtag_strength", 0),
                               weighted=0.0, note="Κανένα hashtag.")
        notes = []
        score = hset.score                       # 0-100 από τη μηχανή hashtags
        measured = sum(1 for c in hset.candidates if c.media_count is not None)
        coverage = measured / len(hset.candidates)
        if coverage < 0.4:
            score *= 0.90
            notes.append(f"Μόνο για {measured}/{len(hset.candidates)} hashtags "
                         f"μετρήθηκε πραγματικό μέγεθος από το API.")
        if len(hset.tags) < 8:
            score *= 0.92
            notes.append("Λίγα hashtags — χάνεται δυναμικό ανακάλυψης.")
        if len(hset.tags) > 20:
            score *= 0.88
            notes.append("Πάνω από 20 hashtags — το Instagram τα αγνοεί μετά τα 30 "
                         "και η υπερβολή διαβάζεται ως spam.")
        n_evidence = sum(c.evidence_count for c in hset.candidates)
        return PillarScore(name="hashtag_strength", label_el=LABELS["hashtag_strength"],
                           raw=_clamp(score), weight=self.weights.get("hashtag_strength", 0),
                           weighted=0.0, evidence_n=n_evidence,
                           origin=Origin.DERIVED, note=" ".join(notes))

    def _greek_fit(self, caption: CaptionCandidate, hset: HashtagSet,
                   analysis: VideoAnalysis) -> PillarScore:
        # Γλώσσα λεζάντας — το βαρύτερο συστατικό
        lang = 100.0 * min(1.0, caption.greek_ratio / 0.75) if caption.greek_ratio else 0.0
        tags = 100.0 * hset.greek_share
        cultural = float(analysis.signals.greek_cultural_fit)
        raw = 0.50 * lang + 0.25 * tags + 0.25 * cultural
        notes = []
        if caption.greek_ratio < 0.6:
            notes.append(f"Η λεζάντα είναι μόνο {caption.greek_ratio:.0%} ελληνική.")
        if hset.greek_share < 0.5:
            notes.append(f"Μόνο {hset.greek_share:.0%} των hashtags στοχεύουν ελληνικό κοινό.")
        return PillarScore(name="greek_audience_fit", label_el=LABELS["greek_audience_fit"],
                           raw=_clamp(raw), weight=self.weights.get("greek_audience_fit", 0),
                           weighted=0.0, origin=Origin.DERIVED, note=" ".join(notes))

    def _historical(self, memory_support: Optional[dict],
                    memory_hits: int) -> PillarScore:
        if not memory_support or not memory_support.get("patterns"):
            return PillarScore(
                name="historical_evidence", label_el=LABELS["historical_evidence"],
                raw=35.0, weight=self.weights.get("historical_evidence", 0),
                weighted=0.0, evidence_n=0, origin=Origin.INFERRED,
                note="Η μνήμη δεν έχει ακόμη αρκετά δεδομένα — ουδέτερη τιμή. "
                     "Βελτιώνεται με κάθε ανάλυση.")
        patterns = memory_support["patterns"]
        n = sum(int(p.get("n", 0)) for p in patterns)
        raw = 100.0 * memory_support["score"] * (0.55 + 0.45 * min(1.0, memory_support.get("coverage", 0)))
        # Μπόνους από πλήθος ανακτημένων ιστορικών παραδειγμάτων
        raw += min(10.0, memory_hits * 0.6)
        return PillarScore(name="historical_evidence", label_el=LABELS["historical_evidence"],
                           raw=_clamp(raw), weight=self.weights.get("historical_evidence", 0),
                           weighted=0.0, evidence_n=n, origin=Origin.DERIVED,
                           note=f"{len(patterns)} μοτίβα, {n} συνολικά δείγματα.")

    def _competitor(self, research: Optional[ResearchBundle],
                    hset: HashtagSet) -> PillarScore:
        if research is None or not research.posts:
            return PillarScore(
                name="competitor_evidence", label_el=LABELS["competitor_evidence"],
                raw=30.0, weight=self.weights.get("competitor_evidence", 0),
                weighted=0.0, evidence_n=0, origin=Origin.INFERRED,
                note="Δεν έγινε έρευνα HikerAPI — καμία τεκμηρίωση ανταγωνισμού.")
        outliers = research.outliers
        n_greek = research.greek_posts
        # Πόσο του σετ hashtags στηρίζεται σε πραγματικά επιτυχημένα posts
        outlier_tags = set()
        for p in outliers:
            outlier_tags.update(p.hashtags)
        overlap = (len(set(hset.tags) & outlier_tags) / len(hset.tags)) if hset.tags else 0.0
        volume = min(1.0, n_greek / 35.0)
        quality = min(1.0, len(outliers) / 10.0)
        raw = 100.0 * (0.42 * overlap + 0.33 * quality + 0.25 * volume)
        return PillarScore(name="competitor_evidence", label_el=LABELS["competitor_evidence"],
                           raw=_clamp(raw), weight=self.weights.get("competitor_evidence", 0),
                           weighted=0.0, evidence_n=len(outliers), origin=Origin.DERIVED,
                           note=f"{len(outliers)} ελληνικά outliers· {overlap:.0%} των "
                                f"hashtags εμφανίζονται σε αυτά.")

    def _trend(self, research: Optional[ResearchBundle],
               analysis: VideoAnalysis) -> PillarScore:
        if research is None or not research.posts:
            return PillarScore(name="trend_signal", label_el=LABELS["trend_signal"],
                               raw=40.0, weight=self.weights.get("trend_signal", 0),
                               weighted=0.0, origin=Origin.INFERRED,
                               note="Χωρίς φρέσκα δεδομένα δεν ανιχνεύεται τάση.")
        now = time.time()
        recent = [p for p in research.outliers
                  if p.taken_at and (now - p.taken_at) / 86400.0 <= 21]
        share = (len(recent) / len(research.outliers)) if research.outliers else 0.0
        raw = 30.0 + 60.0 * share
        note = (f"{len(recent)}/{len(research.outliers)} outliers είναι των "
                f"τελευταίων 21 ημερών.") if research.outliers else "Χωρίς outliers."
        if analysis.technical.duration_s and research.outliers:
            durations = [p.duration_s for p in research.outliers if p.duration_s]
            if len(durations) >= 4:
                med = statistics.median(durations)
                if abs(analysis.technical.duration_s - med) <= med * 0.4:
                    raw += 8.0
                    note += f" Η διάρκεια ({analysis.technical.duration_s:.0f}s) " \
                            f"είναι κοντά στο διάμεσο των επιτυχημένων ({med:.0f}s)."
        return PillarScore(name="trend_signal", label_el=LABELS["trend_signal"],
                           raw=_clamp(raw), weight=self.weights.get("trend_signal", 0),
                           weighted=0.0, evidence_n=len(recent), origin=Origin.DERIVED,
                           note=note)

    # ── συνολικό ─────────────────────────────────────────────────────
    def score(self, analysis: VideoAnalysis, angle: ViralAngle,
              caption: CaptionCandidate, hset: HashtagSet,
              research: Optional[ResearchBundle] = None,
              mined: Optional[MinedPatterns] = None,
              memory_support: Optional[dict] = None,
              memory_hits: int = 0) -> ViralScore:
        pillars = [
            self._content_angle_fit(analysis, angle),
            self._caption_strength(caption, angle, mined, memory_support),
            self._hashtag_strength(hset),
            self._greek_fit(caption, hset, analysis),
            self._historical(memory_support, memory_hits),
            self._competitor(research, hset),
            self._trend(research, analysis),
        ]
        for p in pillars:
            p.weighted = round(p.raw * p.weight, 2)
        raw_total = sum(p.weighted for p in pillars)

        greek_posts = research.greek_posts if research else 0
        outliers = len(research.outliers) if research else 0
        multiplier = self._evidence_multiplier(greek_posts, outliers)
        total = raw_total * multiplier
        confidence, half_width = self._confidence(greek_posts, outliers)

        notes = []
        if multiplier < 0.95:
            notes.append(
                f"Το σκορ περιορίστηκε σε ×{multiplier:.2f} λόγω περιορισμένων "
                f"τεκμηρίων ({greek_posts} ελληνικά posts, {outliers} outliers). "
                f"Χωρίς δεδομένα το σύστημα δεν δικαιούται υψηλή βεβαιότητα.")
        if research is not None and research.degraded:
            notes.append(f"Η έρευνα ήταν υποβαθμισμένη: {research.degraded_reason}")

        return ViralScore(
            total=round(total, 1), raw_total=round(raw_total, 1),
            evidence_multiplier=round(multiplier, 3), confidence=confidence,
            interval=[round(max(0.0, total - half_width), 1),
                      round(min(100.0, total + half_width), 1)],
            pillars=pillars, observed_posts_used=len(research.posts) if research else 0,
            greek_posts_used=greek_posts, notes=notes)

    def _evidence_multiplier(self, greek_posts: int, outliers: int) -> float:
        cfg = (self.raw.get("evidence_multiplier") or {})
        lo = float(cfg.get("min", 0.72))
        hi = float(cfg.get("max", 1.0))
        need_posts = float(cfg.get("full_evidence_greek_posts", 40))
        need_out = float(cfg.get("full_evidence_outliers", 12))
        # Λογαριθμικός κορεσμός: τα πρώτα δεδομένα μετράνε πολύ περισσότερο
        # από τα επόμενα — από 0 σε 10 posts κερδίζεις πολύ, από 60 σε 70 ελάχιστα.
        p = math.log1p(greek_posts) / math.log1p(need_posts)
        o = math.log1p(outliers) / math.log1p(need_out)
        coverage = min(1.0, 0.6 * p + 0.4 * o)
        return lo + (hi - lo) * coverage

    def _confidence(self, greek_posts: int, outliers: int) -> tuple:
        bands = self.raw.get("confidence_bands") or {}
        for name in ("υψηλή", "μεσαία"):
            band = bands.get(name)
            if band and greek_posts >= band.get("min_greek_posts", 10 ** 9) \
                    and outliers >= band.get("min_outliers", 10 ** 9):
                return name, float(band.get("interval", 8))
        low = bands.get("χαμηλή", {})
        return "χαμηλή", float(low.get("interval", 16))


def _clamp(x: float) -> float:
    return round(max(0.0, min(100.0, float(x))), 1)
