"""
Σχεδιασμός έρευνας: από την ανάλυση βίντεο → συγκεκριμένα ερωτήματα HikerAPI.

Δύο πηγές ερωτημάτων, σκόπιμα:

  1. ΣΤΑΘΕΡΗ ΒΑΣΗ (config/niches.json) — δουλεύει πάντα, ακόμη και χωρίς LLM,
     και δίνει επαναληψιμότητα.
  2. ΔΥΝΑΜΙΚΗ ΕΠΕΚΤΑΣΗ (LLM) — όροι ειδικά για ΑΥΤΟ το βίντεο, στα ελληνικά
     όπως τα γράφει πραγματικά ο κόσμος (με τόνους, χωρίς τόνους, greeklish).

Χωρίς το (1) το σύστημα γίνεται απρόβλεπτο. Χωρίς το (2) ψάχνει πάντα τα ίδια.

Το budget κατανέμεται ΠΡΙΝ γίνει οποιαδήποτε κλήση: κάθε κλήση κοστίζει
credits, οπότε η ιεράρχηση πρέπει να είναι ρητή απόφαση, όχι παρενέργεια
της σειράς εκτέλεσης.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..clients.llm.base import AnthropicClient, json_block, text_block
from ..logging_setup import get_logger
from ..schemas import ResearchQuery, ViralAngle, VideoContent

log = get_logger("research.planner")

SYSTEM = """Είσαι ερευνητής ελληνικής αγοράς Instagram. Παράγεις ΟΡΟΥΣ ΑΝΑΖΗΤΗΣΗΣ, \
όχι περιεχόμενο.

Στόχος: να βρεθούν ΠΡΑΓΜΑΤΙΚΑ ελληνικά Reels που πέτυχαν με παρόμοιο θέμα/γωνία, \
ώστε να μάθουμε από αυτά.

ΚΑΝΟΝΕΣ:
1. Γράψε όπως γράφει ο ΕΛΛΗΝΑΣ χρήστης στο Instagram — όχι όπως γράφει ένα λεξικό.
   Ο κόσμος ψάχνει «αστεια ελληνικα», όχι «ελληνικό χιουμοριστικό περιεχόμενο».
2. Στα hashtags: ΠΟΤΕ τόνους (το Instagram τα κανονικοποιεί) και ΠΟΤΕ κενά.
   Σωστό: `ελληνικοχιουμορ`. Λάθος: `ελληνικό χιούμορ`, `#ελληνικό_χιούμορ`.
3. Δώσε ΜΕΙΓΜΑ μεγεθών: 2-3 πλατιά (εκατομμύρια posts), 4-6 μεσαία, 4-6 στενά
   του συγκεκριμένου θέματος. Τα στενά είναι που δίνουν πραγματική ανακάλυψη.
4. Πρόσθεσε 2-3 greeklish παραλλαγές — μεγάλο μέρος του κοινού γράφει έτσι.
5. Οι λέξεις-κλειδιά για αναζήτηση Reels είναι ΦΡΑΣΕΙΣ (2-5 λέξεις), όχι
   μονολεκτικά και όχι ολόκληρες προτάσεις.
6. Μη βάλεις όρους άσχετους με το βίντεο για να «πιάσεις κόσμο». Άσχετη
   έρευνα δίνει άσχετα τεκμήρια και χαλάει την τελική απόφαση."""

SCHEMA = {
    "type": "object",
    "properties": {
        "niche": {"type": "string", "description": "κύριο niche στα ελληνικά"},
        "sub_niche": {"type": "string"},
        "search_keywords": {
            "type": "array", "minItems": 4, "maxItems": 10,
            "items": {"type": "string"},
            "description": "ελληνικές φράσεις αναζήτησης Reels (2-5 λέξεις)"},
        "hashtags_broad": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        "hashtags_mid": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "hashtags_niche": {"type": "array", "maxItems": 10, "items": {"type": "string"}},
        "greeklish_variants": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        "locations": {"type": "array", "maxItems": 3, "items": {"type": "string"},
                      "description": "μόνο αν το βίντεο έχει σαφή τοπική διάσταση"},
        "reasoning": {"type": "string", "description": "γιατί αυτοί οι όροι, 2-3 προτάσεις"},
    },
    "required": ["niche", "sub_niche", "search_keywords", "hashtags_broad",
                 "hashtags_mid", "hashtags_niche", "greeklish_variants", "reasoning"],
}


def load_niche_config(config_dir: Path) -> dict:
    path = config_dir / "niches.json"
    if not path.is_file():
        return {"niches": {}, "location_seeds": [], "broad_greek_hashtags": []}
    return json.loads(path.read_text(encoding="utf-8"))


def load_seed_creators(config_dir: Path) -> list:
    path = config_dir / "seed_creators.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [c["username"] for c in data.get("creators", []) if c.get("username")]


def _match_niche(cfg: dict, niche: str, sub_niche: str) -> dict:
    """Ταίριασμα με χαλαρή σύγκριση — το LLM δεν επιστρέφει πάντα ακριβώς το κλειδί."""
    from .. import greek as G
    niches = cfg.get("niches", {})
    target = G.normalize(f"{niche} {sub_niche}")
    best, best_hits = None, 0
    for key, val in niches.items():
        k = G.normalize(key)
        hits = 2 if k in target else 0
        hits += sum(1 for w in G.tokenize(val.get("audience", "")) if w in target)
        for kw in val.get("keywords", []):
            if any(t in target for t in G.tokenize(kw)):
                hits += 1
        if hits > best_hits:
            best, best_hits = val, hits
    return best or {}


def plan(content: VideoContent, angle: Optional[ViralAngle],
         llm: Optional[AnthropicClient], config_dir: Path,
         budget: int = 120, model: str = "") -> dict:
    """Επιστρέφει `{niche, sub_niche, queries, terms, reasoning}`."""
    cfg = load_niche_config(config_dir)
    terms = {"search_keywords": [], "hashtags_broad": [], "hashtags_mid": [],
             "hashtags_niche": [], "greeklish_variants": [], "locations": []}
    niche = content.niche or ""
    sub_niche = content.sub_niche or ""
    reasoning = ""

    # 1) Δυναμική επέκταση από το μοντέλο
    if llm is not None:
        try:
            blocks = [
                text_block("Δώσε όρους αναζήτησης για να βρούμε ελληνικά Reels "
                           "παρόμοια με το παρακάτω."),
                json_block("ανάλυση_βίντεο", {
                    "περίληψη": content.summary, "θέμα": content.main_subject,
                    "περιβάλλον": content.environment, "διάθεση": content.mood,
                    "niche": content.niche, "sub_niche": content.sub_niche,
                    "κοινό": content.target_audience,
                    "ευρύτερο_κοινό": content.potential_audience,
                    "χιούμορ": content.humor, "αισθητική": content.aesthetic,
                    "ενέργειες": content.actions,
                    "κείμενο_στην_οθόνη": content.on_screen_text,
                    "πολιτισμικά_στοιχεία": content.cultural_markers,
                }),
            ]
            if angle is not None:
                blocks.append(json_block("επιλεγμένη_γωνία", {
                    "όνομα": angle.name, "στρατηγική": angle.strategy,
                    "γιατί_σταματά": angle.why_greek_stops,
                    "τι_προσθέτει_η_λεζάντα": angle.caption_should_add,
                    "τμήμα_κοινού": angle.target_segment}))
            raw = llm.structured(system=SYSTEM, content=blocks, schema=SCHEMA,
                                 tool_name="submit_queries", model=model or None,
                                 max_tokens=2000, temperature=0.6)
            niche = raw.get("niche") or niche
            sub_niche = raw.get("sub_niche") or sub_niche
            reasoning = raw.get("reasoning", "")
            for key in terms:
                terms[key] = [str(v).strip() for v in (raw.get(key) or []) if str(v).strip()]
        except Exception as exc:                    # noqa: BLE001
            log.warning("Ο δυναμικός σχεδιασμός απέτυχε (%s) — μόνο σταθερή βάση",
                        type(exc).__name__)

    # 2) Σταθερή βάση από config
    base = _match_niche(cfg, niche, sub_niche)
    terms["search_keywords"] += [k for k in base.get("keywords", [])
                                 if k not in terms["search_keywords"]]
    terms["hashtags_mid"] += [h for h in base.get("hashtags", [])
                              if h not in terms["hashtags_mid"]]
    terms["hashtags_broad"] += [h for h in cfg.get("broad_greek_hashtags", [])[:3]
                                if h not in terms["hashtags_broad"]]

    # Καθαρισμός hashtags: χωρίς δίεση, χωρίς κενά, χωρίς τόνους
    from .. import greek as G
    for key in ("hashtags_broad", "hashtags_mid", "hashtags_niche"):
        cleaned, seen = [], set()
        for tag in terms[key]:
            t = G.strip_accents(tag.lstrip("#").replace(" ", "").replace("_", "")).lower()
            if t and t not in seen and 2 <= len(t) <= 40:
                seen.add(t)
                cleaned.append(t)
        terms[key] = cleaned

    queries = _build_queries(terms, budget)
    log.info("Σχέδιο έρευνας: %d ερωτήματα, niche=«%s»", len(queries), niche)
    return {"niche": niche, "sub_niche": sub_niche, "queries": queries,
            "terms": terms, "reasoning": reasoning}


def _build_queries(terms: dict, budget: int) -> list:
    """
    Ιεράρχηση ερωτημάτων μέσα στο budget.

    Η σειρά προτεραιότητας δεν είναι αυθαίρετη:
      1. Στενά hashtags → εκεί βρίσκονται τα μικρά viral, το πιο διδακτικό υλικό.
      2. Λέξεις-κλειδιά  → «τι δουλεύει ΤΩΡΑ σε αυτό το θέμα».
      3. Μεσαία hashtags → όγκος τεκμηρίων.
      4. Πλατιά          → μόνο για μέτρο σύγκρισης· έχουν τον χειρότερο λόγο
                            σήματος/κόστους γιατί κυριαρχούνται από μεγάλα προφίλ.
    """
    queries: list = []
    spend = 0

    def add(kind: str, value: str, reason: str, priority: int, cost: int = 1) -> None:
        nonlocal spend
        if spend + cost > budget:
            return
        queries.append(ResearchQuery(kind=kind, value=value, reason=reason,
                                     priority=priority, estimated_calls=cost))
        spend += cost

    for tag in terms["hashtags_niche"][:8]:
        add("hashtag", tag, "στενό hashtag — εδώ εμφανίζονται μικρά viral", 1, 2)
    for kw in terms["search_keywords"][:6]:
        add("keyword", kw, "τι δουλεύει τώρα σε αυτό το θέμα", 2, 1)
    for tag in terms["hashtags_mid"][:8]:
        add("hashtag", tag, "μεσαίο hashtag — όγκος τεκμηρίων", 3, 2)
    for gl in terms["greeklish_variants"][:3]:
        add("keyword", gl, "greeklish παραλλαγή — άλλο τμήμα κοινού", 4, 1)
    for tag in terms["hashtags_broad"][:3]:
        add("hashtag", tag, "πλατύ hashtag — μέτρο σύγκρισης", 5, 2)
    for loc in terms["locations"][:2]:
        add("location", loc, "τοπική στόχευση", 6, 2)

    queries.sort(key=lambda q: q.priority)
    return queries
