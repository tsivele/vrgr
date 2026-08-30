"""
Μηχανή παραγωγής λεζαντών.

Ροή:
  1. Χτίσιμο prompt ΤΕΚΜΗΡΙΩΜΕΝΟΥ σε πραγματικά δεδομένα (όχι γενικές οδηγίες)
  2. Παραγωγή 6-10 υποψηφίων με ΔΙΑΦΟΡΕΤΙΚΗ στρατηγική η καθεμία
  3. Έλεγχος λογοκλοπής έναντι του corpus
  4. Έλεγχος ποιότητας ελληνικών (μεταφρασμένα μοτίβα, λάθος ερωτηματικό)

Το βήμα 3 δεν είναι διακοσμητικό: δίνουμε στο μοντέλο πραγματικές
επιτυχημένες λεζάντες ως παραδείγματα δομής, οπότε ο κίνδυνος να
αναπαράγει μία από αυτές είναι υπαρκτός. Ό,τι ξεπερνά το κατώφλι
ομοιότητας απορρίπτεται αυτόματα.
"""
from __future__ import annotations

import re
from typing import Optional

from .. import greek as G
from ..clients.llm.base import AnthropicClient, json_block, text_block
from ..clients.llm.embeddings import jaccard_ngram
from ..logging_setup import get_logger
from ..schemas import (CaptionCandidate, MinedPatterns, VideoAnalysis, ViralAngle)
from .prompts import CAPTION_SCHEMA, CAPTION_SYSTEM

log = get_logger("generation.captions")

PLAGIARISM_THRESHOLD = 0.62      # πάνω από αυτό θεωρείται αντιγραφή

# Μοτίβα που προδίδουν μετάφραση από αγγλικά — ο Έλληνας τα αναγνωρίζει αμέσως.
TRANSLATIONESE = [
    (re.compile(r"\bχτυπ[άα]ει\s+διαφορετικ", re.I), "«χτυπάει διαφορετικά» (hits different)"),
    (re.compile(r"\bκυριολεκτικ[άα]\s+εγ[ώω]\b", re.I), "«κυριολεκτικά εγώ» (literally me)"),
    (re.compile(r"\bζω\s+γι'?\s*αυτ[όο]\b", re.I), "«ζω γι' αυτό» (living for this)"),
    (re.compile(r"\bκανε[ίι]ς[:：]\s*κανε[ίι]ς[:：]", re.I), "«κανείς: κανείς:» (nobody: nobody:)"),
    (re.compile(r"\bπες\s+μου\s+[όο]τι\s+.*\s+χωρ[ίι]ς\s+να\s+μου\s+πεις", re.I),
     "«πες μου ότι… χωρίς να μου πεις»"),
    (re.compile(r"\bείναι\s+μια\s+διάθεση\b", re.I), "«είναι μια διάθεση» (it's a mood)"),
    (re.compile(r"\bκύρι[εο]\s+μαθήματα\b", re.I), "κακή μετάφραση"),
]


def _evidence_block(mined: Optional[MinedPatterns], memory_support: Optional[dict],
                    examples: list) -> list:
    """Τα ΜΕΤΡΗΜΕΝΑ στοιχεία που δεσμεύουν τον generator."""
    blocks = []
    if mined and mined.outlier_sample_size >= 3:
        blocks.append(json_block("μετρημένα_μοτίβα_επιτυχημένων_ελληνικών_reels", {
            "_πηγή": f"{mined.outlier_sample_size} ελληνικά Reels με δυσανάλογη "
                     f"απόδοση, από δείγμα {mined.sample_size} (HikerAPI, τώρα)",
            "μήκος_λεζάντας_χαρακτήρες": {
                "διάμεσο": mined.caption_len_median,
                "συνηθισμένο_εύρος": [mined.caption_len_p25, mined.caption_len_p75]},
            "ποσοστό_με_ερώτηση": mined.question_share,
            "ποσοστό_με_κάλεσμα_δράσης": mined.cta_share,
            "ποσοστό_σε_β_πρόσωπο": mined.second_person_share,
            "ποσοστό_σε_α_πρόσωπο": mined.first_person_share,
            "διάμεσο_emoji": mined.emoji_median,
            "λέξεις_με_ανύψωση": [
                {"λέξη": w, "ανύψωση": l, "εμφανίσεις": n}
                for w, l, n in (mined.top_words or [])[:12]],
            "εκφράσεις_με_ανύψωση": (mined.greek_expressions or [])[:10],
        }))
    if memory_support and memory_support.get("patterns"):
        blocks.append(json_block("μοτίβα_από_τη_μνήμη", {
            "_σημείωση": "Μαθεύτηκαν σε προηγούμενες αναλύσεις. Το «lower» είναι "
                         "συντηρητικό κάτω φράγμα — εμπιστεύσου το, όχι τον μέσο όρο.",
            "μοτίβα": memory_support["patterns"][:10],
        }))
    if examples:
        blocks.append(json_block("παραδείγματα_δομής_ΜΗΝ_ΑΝΤΙΓΡΑΨΕΙΣ", {
            "_προσοχή": "Πραγματικές λεζάντες επιτυχημένων ελληνικών Reels. "
                        "Μελέτησε ΠΩΣ είναι φτιαγμένες (ρυθμός, στροφή, κλείσιμο). "
                        "Η αντιγραφή φράσεων ΑΠΟΡΡΙΠΤΕΤΑΙ αυτόματα.",
            "παραδείγματα": examples[:12],
        }))
    return blocks


def generate(analysis: VideoAnalysis, angle: ViralAngle, llm: AnthropicClient,
             mined: Optional[MinedPatterns] = None,
             memory_support: Optional[dict] = None,
             corpus_captions: Optional[list] = None,
             model: str = "", n_target: int = 8) -> list:
    corpus_captions = corpus_captions or []
    examples = [c for c in corpus_captions if 15 <= len(c) <= 320][:12]

    content = [
        text_block(
            f"Γράψε {n_target} διαφορετικές υποψήφιες λεζάντες για αυτό το Reel.\n\n"
            f"Κάθε μία με ΔΙΑΦΟΡΕΤΙΚΗ στρατηγική — όχι παραλλαγές της ίδιας ιδέας. "
            f"Θα βαθμολογηθούν και θα επιλεγεί ΜΙΑ."),
        json_block("ανάλυση_βίντεο", {
            "περίληψη": analysis.content.summary,
            "θέμα": analysis.content.main_subject,
            "hook_πρώτα_3_δευτ": analysis.content.hook_description,
            "περιβάλλον": analysis.content.environment,
            "διάθεση": analysis.content.mood,
            "ενέργεια": analysis.content.energy_level,
            "συναισθήματα": analysis.content.emotions,
            "χιούμορ": analysis.content.humor,
            "κείμενο_στην_οθόνη": analysis.content.on_screen_text,
            "μεταγραφή_ήχου": analysis.content.spoken_transcript or "(μη διαθέσιμη)",
            "niche": analysis.content.niche,
            "sub_niche": analysis.content.sub_niche,
            "κοινό": analysis.content.target_audience,
            "ελληνικά_στοιχεία": analysis.content.cultural_markers,
            "διάρκεια_δευτ": round(analysis.technical.duration_s, 1),
        }),
        json_block("ΕΠΙΛΕΓΜΕΝΗ_ΓΩΝΙΑ", {
            "όνομα": angle.name,
            "στρατηγική": angle.strategy,
            "γιατί_σταματά_ο_Έλληνας": angle.why_greek_stops,
            "γιατί_σχολιάζει": angle.why_comment,
            "γιατί_στέλνει": angle.why_share,
            "ΤΙ_ΠΡΕΠΕΙ_ΝΑ_ΠΡΟΣΘΕΣΕΙ_Η_ΛΕΖΑΝΤΑ": angle.caption_should_add,
            "τμήμα_κοινού": angle.target_segment,
            "κίνδυνος": angle.risk,
        }),
    ]
    content += _evidence_block(mined, memory_support, examples)
    content.append(text_block(
        "Θυμήσου: η λεζάντα ΔΕΝ περιγράφει το βίντεο. Προσθέτει το στρώμα που "
        "λείπει. Αν μια υποψήφια μπορεί να διαβαστεί ως λεζάντα φωτογραφίας, "
        "είναι λάθος — πρέπει να συνδέεται με ΑΥΤΟ το συγκεκριμένο Reel."))

    log.info("Παραγωγή %d υποψηφίων λεζαντών (γωνία: %s)…", n_target, angle.name)
    raw = llm.structured(system=CAPTION_SYSTEM, content=content,
                         schema=CAPTION_SCHEMA, tool_name="submit_captions",
                         model=model or None, max_tokens=6000, thinking_budget=3000)

    candidates = []
    for item in raw.get("captions", []):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        text = _clean(text)
        c = CaptionCandidate(
            text=text, strategy=item.get("strategy", ""), angle_name=angle.name,
            rationale=item.get("rationale", ""),
        ).fill()
        c.has_cta = bool(item.get("expected_reaction") in
                         ("σχόλιο", "αποστολή σε φίλο", "αποθήκευση"))
        c.similarity_to_corpus = _max_similarity(text, corpus_captions)
        candidates.append(c)

    kept, rejected = [], []
    for c in candidates:
        problem = quality_problem(c)
        if problem:
            rejected.append((c.text[:60], problem))
            continue
        kept.append(c)

    for text, reason in rejected:
        log.info("Απορρίφθηκε «%s…» — %s", text, reason)
    if not kept and candidates:
        log.warning("Όλες οι υποψήφιες απορρίφθηκαν — κρατάμε την καλύτερη διαθέσιμη")
        kept = sorted(candidates, key=lambda c: c.similarity_to_corpus)[:3]
    log.info("Κρατήθηκαν %d/%d υποψήφιες", len(kept), len(candidates))
    return kept


def _clean(text: str) -> str:
    """Αφαίρεση hashtags που ξέφυγαν + κανονικοποίηση ερωτηματικού."""
    text = G.HASHTAG_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Το αγγλικό «?» σε ελληνικό κείμενο είναι λάθος στίξη.
    if G.greek_ratio(text) > 0.5:
        text = text.replace("?", ";")
    return text.strip()


def _max_similarity(text: str, corpus: list) -> float:
    if not corpus:
        return 0.0
    return round(max(jaccard_ngram(text, c) for c in corpus), 3)


def quality_problem(c: CaptionCandidate) -> Optional[str]:
    """Επιστρέφει τον λόγο απόρριψης, ή None αν η λεζάντα είναι αποδεκτή."""
    if c.similarity_to_corpus >= PLAGIARISM_THRESHOLD:
        return f"πολύ κοντά σε υπάρχουσα λεζάντα (ομοιότητα {c.similarity_to_corpus})"
    if len(c.text) < 8:
        return "πολύ σύντομη"
    if len(c.text) > 2100:
        return "ξεπερνά το όριο του Instagram (2.200 χαρακτήρες με τα hashtags)"
    for pattern, label in TRANSLATIONESE:
        if pattern.search(c.text):
            return f"μεταφρασμένο μοτίβο: {label}"
    if c.greek_ratio < 0.45 and len(c.text) > 25:
        return f"δεν είναι κυρίως ελληνικά (ελληνικοί χαρακτήρες {c.greek_ratio:.0%})"
    if c.emoji_count > 8:
        return f"υπερβολικά emoji ({c.emoji_count})"
    return None


def diversify(candidates: list, max_n: int = 8) -> list:
    """
    Απομάκρυνση σχεδόν-διπλών μεταξύ των υποψηφίων.

    Χωρίς αυτό, το μοντέλο συχνά δίνει 8 παραλλαγές της ίδιας ιδέας — και
    τότε η «επιλογή ανάμεσα σε 8» δεν είναι πραγματική επιλογή.
    """
    kept: list = []
    for c in candidates:
        if any(jaccard_ngram(c.text, k.text) > 0.55 for k in kept):
            continue
        kept.append(c)
        if len(kept) >= max_n:
            break
    return kept
