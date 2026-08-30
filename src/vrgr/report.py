"""
Παρουσίαση αποτελέσματος στα ελληνικά (απαίτηση #19).

ΚΑΝΟΝΑΣ ΠΑΡΟΥΣΙΑΣΗΣ: κάθε αριθμός εμφανίζεται με την προέλευσή του.
  ✓ = OBSERVED  (μετρήθηκε από HikerAPI — γεγονός)
  ≈ = DERIVED   (υπολογίστηκε από μετρημένα)
  ~ = INFERRED  (εκτίμηση μοντέλου — ΟΧΙ γεγονός)

Έτσι είναι αδύνατο να διαβαστεί μια εκτίμηση ως μέτρηση.
"""
from __future__ import annotations

from typing import Optional

from .generation.hashtags import format_for_instagram
from .schemas import AnalysisResult, Origin

MARK = {Origin.OBSERVED: "✓", Origin.DERIVED: "≈", Origin.INFERRED: "~"}
W = 78


def _rule(char: str = "─") -> str:
    return char * W


def _title(text: str) -> str:
    return f"\n{_rule('═')}\n  {text}\n{_rule('═')}"


def _section(text: str) -> str:
    return f"\n{text}\n{_rule()}"


def _num(n: Optional[float], suffix: str = "") -> str:
    if n is None:
        return "—"
    if isinstance(n, float) and not n.is_integer():
        return f"{n:,.1f}{suffix}".replace(",", ".")
    return f"{int(n):,}{suffix}".replace(",", ".")


def _bar(value: float, width: int = 22) -> str:
    filled = int(round(width * max(0.0, min(100.0, value)) / 100.0))
    return "█" * filled + "░" * (width - filled)


def render(result: AnalysisResult, verbose: bool = False) -> str:
    if result.winner is None:
        return "Η ανάλυση δεν παρήγαγε αποτέλεσμα."
    w = result.winner
    a = result.video
    out = []

    out.append(_title("ΑΠΟΤΕΛΕΣΜΑ ΑΝΑΛΥΣΗΣ REEL"))
    out.append(f"  Εκτέλεση: {result.run_id}   ·   Διάρκεια: {result.duration_s}s"
               f"   ·   Κλήσεις HikerAPI: {result.api_calls}")

    # ── 1. ΑΝΑΛΥΣΗ ΒΙΝΤΕΟ ────────────────────────────────────────────
    out.append(_section("1. ΑΝΑΛΥΣΗ ΒΙΝΤΕΟ"))
    t, c = a.technical, a.content
    out.append(f"  {c.summary}")
    out.append("")
    out.append(f"  Θέμα           ~ {c.main_subject}")
    out.append(f"  Niche          ~ {c.niche} → {c.sub_niche}")
    out.append(f"  Διάθεση        ~ {c.mood} (ενέργεια: {c.energy_level})")
    out.append(f"  Κοινό          ~ {c.target_audience}")
    if c.potential_audience:
        out.append(f"  Ευρύτερο κοινό ~ {c.potential_audience}")
    out.append(f"  Hook (0-3s)    ~ {c.hook_description}")
    out.append(f"  Τεχνικά        ✓ {t.duration_s:.1f}s · {t.width}x{t.height} "
               f"({t.aspect_ratio}) · {t.cut_count} κοψίματα "
               f"({t.hook_cut_count} στο hook) · ήχος: {'ναι' if t.has_audio else 'όχι'}")
    if c.on_screen_text:
        out.append(f"  Κείμενο οθόνης ~ {' | '.join(c.on_screen_text[:4])}")
    if c.spoken_transcript:
        out.append(f"  Ομιλία         ✓ «{c.spoken_transcript[:120]}»")
    if c.cultural_markers:
        out.append(f"  Ελληνικά στοιχεία ~ {', '.join(c.cultural_markers[:5])}")

    s = a.signals
    out.append("")
    out.append("  Δείκτες δυναμικού (~ εκτίμηση μοντέλου):")
    for label, value in (("Διατήρηση προσοχής", s.attention_hold),
                         ("Περιέργεια", s.curiosity_gap),
                         ("Ταύτιση", s.relatability),
                         ("Αποστολή σε φίλο", s.shareability),
                         ("Πρόκληση σχολίου", s.commentability),
                         ("Αποθήκευση", s.save_potential),
                         ("Επαναπροβολή", s.rewatch_potential),
                         ("Ελληνικό ταίριασμα", s.greek_cultural_fit)):
        out.append(f"    {label:22} {_bar(value)} {value:3d}")
    if s.notes:
        out.append(f"    ⚠ Ασθενέστερο σημείο: {s.notes}")

    # ── 2. VIRAL ΓΩΝΙΑ ──────────────────────────────────────────────
    out.append(_section("2. VIRAL ΓΩΝΙΑ"))
    ang = a.chosen_angle
    out.append(f"  «{ang.name}»  ({ang.strategy}, ισχύς {ang.strength}/100)")
    out.append("")
    out.append(f"  Γιατί σταματά το scroll : {ang.why_greek_stops}")
    out.append(f"  Γιατί σχολιάζει         : {ang.why_comment}")
    out.append(f"  Γιατί το στέλνει        : {ang.why_share}")
    out.append(f"  Τι προσθέτει η λεζάντα  : {ang.caption_should_add}")
    if ang.risk:
        out.append(f"  ⚠ Κίνδυνος              : {ang.risk}")
    if len(a.angles) > 1 and verbose:
        others = [x for x in a.angles if x.name != ang.name]
        out.append(f"\n  Γωνίες που απορρίφθηκαν: "
                   + ", ".join(f"«{x.name}» ({x.strength})" for x in others))

    # ── 3. ΝΙΚΗΤΡΙΑ ΛΕΖΑΝΤΑ ─────────────────────────────────────────
    out.append(_section("3. ΛΕΖΑΝΤΑ ΠΟΥ ΕΠΙΛΕΧΘΗΚΕ"))
    out.append("")
    for line in w.caption.text.split("\n"):
        out.append(f"    {line}")
    out.append("")
    out.append(f"  Στρατηγική: {w.caption.strategy}  ·  {w.caption.length_chars} χαρακτ."
               f"  ·  {w.caption.emoji_count} emoji"
               f"  ·  {'με ερώτηση' if w.caption.has_question else 'χωρίς ερώτηση'}")
    if w.caption.rationale:
        out.append(f"  Σκεπτικό: {w.caption.rationale}")

    # ── 4. HASHTAGS ─────────────────────────────────────────────────
    out.append(_section("4. HASHTAGS ΠΟΥ ΕΠΙΛΕΧΘΗΚΑΝ"))
    hs = w.hashtag_set
    out.append("")
    out.append(f"    {format_for_instagram(hs.tags, on_new_line=False)}")
    out.append("")
    dist = " · ".join(f"{k}: {v}" for k, v in sorted(hs.tier_distribution.items()))
    out.append(f"  Χαρτοφυλάκιο «{hs.strategy}» — {dist}")
    out.append(f"  Ελληνικά: {hs.greek_share:.0%}  ·  "
               f"Με τεκμήριο σε πραγματικό viral: {hs.evidence_share:.0%}")
    out.append(f"  {hs.rationale}")
    if verbose:
        out.append("")
        out.append(f"  {'hashtag':24}{'επίπεδο':10}{'μέγεθος':>13}{'δυσκ.':>7}{'τεκμ.':>7}")
        for cand in hs.candidates[:18]:
            mc = _num(cand.media_count) if cand.media_count is not None else "—"
            mark = "✓" if cand.media_count is not None else "~"
            out.append(f"  #{cand.tag:23}{cand.tier:10}{mark + ' ' + mc:>13}"
                       f"{_num(cand.difficulty):>7}{cand.evidence_count:>7}")

    # ── 5. VIRAL SCORE ──────────────────────────────────────────────
    out.append(_section("5. VIRAL SCORE"))
    sc = w.score
    out.append("")
    out.append(f"      {sc.total:.0f} / 100      {_bar(sc.total, 34)}")
    out.append(f"      εύρος {sc.interval[0]:.0f}–{sc.interval[1]:.0f}"
               f"   ·   βεβαιότητα: {sc.confidence}")
    out.append("")
    out.append(f"  {'πυλώνας':36}{'σκορ':>6}{'βάρος':>8}{'συνεισφ.':>10}  πηγή")
    for p in sorted(sc.pillars, key=lambda x: -x.weighted):
        out.append(f"  {p.label_el:36}{p.raw:>6.0f}{p.weight:>8.2f}"
                   f"{p.weighted:>10.2f}  {MARK.get(p.origin, '~')}")
    out.append(f"  {'ΣΥΝΟΛΟ (πριν τα τεκμήρια)':36}{'':>6}{'':>8}{sc.raw_total:>10.1f}")
    out.append(f"  {'× πολλαπλασιαστής τεκμηρίων':36}{'':>6}{'':>8}"
               f"{sc.evidence_multiplier:>10.2f}")
    out.append(f"  {'= ΤΕΛΙΚΟ':36}{'':>6}{'':>8}{sc.total:>10.1f}")
    if w.synergy:
        out.append(f"  (περιλαμβάνεται συνέργεια λεζάντας–hashtags: {w.synergy:+.1f})")
    for note in sc.notes:
        out.append(f"\n  ⚠ {note}")

    # ── 6. ΓΙΑΤΙ ΚΕΡΔΙΣΕ ────────────────────────────────────────────
    out.append(_section("6. ΓΙΑΤΙ ΚΕΡΔΙΣΕ ΑΥΤΟΣ Ο ΣΥΝΔΥΑΣΜΟΣ"))
    out.append(f"  {result.why_won}")
    if result.backups:
        runner = result.backups[0]
        out.append(f"\n  Έναντι της δεύτερης ({runner.score.total:.0f}/100, "
                   f"«{runner.caption.strategy}»): διαφορά "
                   f"{w.score.total - runner.score.total:+.1f} πόντοι.")

    # ── 7. ΤΕΚΜΗΡΙΑ ─────────────────────────────────────────────────
    out.append(_section("7. ΤΕΚΜΗΡΙΑ ΑΠΟ ΠΡΑΓΜΑΤΙΚΑ ΔΕΔΟΜΕΝΑ"))
    if result.research:
        r = result.research
        out.append(f"  ✓ {len(r.posts)} posts εξετάστηκαν · {r.greek_posts} ελληνικά · "
                   f"{len(r.outliers)} με δυσανάλογη απόδοση")
        out.append(f"  ✓ {r.api_calls} κλήσεις HikerAPI ({r.cache_hits} από cache)")
    if result.memory_hits:
        out.append(f"  ✓ {result.memory_hits} ιστορικά ανάλογα από τη μνήμη")
    if result.patterns and result.patterns.outlier_sample_size >= 3:
        p = result.patterns
        out.append(f"\n  Μοτίβα από {p.outlier_sample_size} επιτυχημένα ελληνικά Reels:")
        if p.caption_len_median:
            out.append(f"    ≈ μήκος λεζάντας: διάμεσο {p.caption_len_median:.0f} χαρ. "
                       f"(εύρος {p.caption_len_p25:.0f}–{p.caption_len_p75:.0f})")
        if p.question_share is not None:
            out.append(f"    ≈ με ερώτηση {p.question_share:.0%} · "
                       f"με CTA {p.cta_share:.0%} · "
                       f"σε β' πρόσωπο {p.second_person_share:.0%}")
        if p.greek_expressions:
            out.append(f"    ≈ εκφράσεις με ανύψωση: "
                       + ", ".join(f"«{e['phrase']}» ({e['lift']}×)"
                                   for e in p.greek_expressions[:4]))
        if p.top_hashtags:
            out.append(f"    ≈ hashtags με ανύψωση: "
                       + ", ".join(f"#{t} ({l}×)" for t, l, _ in p.top_hashtags[:6]))

    if result.evidence:
        out.append("\n  Συγκεκριμένα posts που στηρίζουν την απόφαση:")
        for e in result.evidence[:5]:
            vf = f"V/F {e.vf_ratio}×" if e.vf_ratio else "V/F —"
            out.append(f"    ✓ @{e.username} — {_num(e.views)} προβολές / "
                       f"{_num(e.followers)} followers ({vf})"
                       + (f", {e.age_days:.0f} ημ." if e.age_days else ""))
            if e.caption_excerpt:
                out.append(f"        «{e.caption_excerpt[:100]}»")
            out.append(f"        → {e.why_relevant}")
    elif not result.research:
        out.append("  ⚠ Δεν συλλέχθηκαν φρέσκα δεδομένα από το Instagram για "
                   "αυτή την ανάλυση.")

    # ── 8. ΕΦΕΔΡΙΚΕΣ ΛΕΖΑΝΤΕΣ ───────────────────────────────────────
    if result.backups:
        out.append(_section("8. ΕΦΕΔΡΙΚΕΣ ΛΕΖΑΝΤΕΣ"))
        for i, b in enumerate(result.backups[:5], 2):
            out.append(f"\n  {i}. [{b.score.total:.0f}/100 · {b.caption.strategy}]")
            for line in b.caption.text.split("\n"):
                out.append(f"     {line}")

    # ── 9. ΕΦΕΔΡΙΚΑ ΣΕΤ HASHTAGS ────────────────────────────────────
    if result.backup_hashtag_sets:
        out.append(_section("9. ΕΝΑΛΛΑΚΤΙΚΑ ΣΕΤ HASHTAGS (για τη νικήτρια λεζάντα)"))
        for b in result.backup_hashtag_sets:
            out.append(f"\n  [{b.score.total:.0f}/100 · «{b.hashtag_set.strategy}»] "
                       f"— {b.hashtag_set.rationale[:70]}")
            out.append(f"  {format_for_instagram(b.hashtag_set.tags, on_new_line=False)}")

    # ── 10. ΠΕΡΙΟΡΙΣΜΟΙ ─────────────────────────────────────────────
    if result.data_gaps or result.warnings:
        out.append(_section("10. ΠΕΡΙΟΡΙΣΜΟΙ ΚΑΙ ΚΕΝΑ ΔΕΔΟΜΕΝΩΝ"))
        for warning in result.warnings:
            out.append(f"  ⚠ {warning}")
        for gap in result.data_gaps:
            out.append(f"  · {gap}")

    # ── ΕΤΟΙΜΟ ΓΙΑ ΑΝΤΙΓΡΑΦΗ ────────────────────────────────────────
    out.append(_title("ΕΤΟΙΜΟ ΓΙΑ ΕΠΙΚΟΛΛΗΣΗ ΣΤΟ INSTAGRAM"))
    out.append("")
    out.append(w.caption.text + format_for_instagram(hs.tags))
    out.append("")
    out.append(_rule("═"))
    out.append(f"  Υπόμνημα:  ✓ μετρημένο από HikerAPI   "
               f"≈ υπολογισμένο από μετρημένα   ~ εκτίμηση μοντέλου")
    out.append(f"  Μετά τη δημοσίευση:  ./run.sh feedback {result.run_id} <URL>")
    out.append(_rule("═"))
    return "\n".join(out)


def render_compact(result: AnalysisResult) -> str:
    """Σύντομη έξοδος — μόνο η απόφαση."""
    if result.winner is None:
        return "—"
    w = result.winner
    return (f"ΣΚΟΡ {w.score.total:.0f}/100 ({w.score.confidence})\n\n"
            f"{w.caption.text}\n{format_for_instagram(w.hashtag_set.tags).strip()}")
