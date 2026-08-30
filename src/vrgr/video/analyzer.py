"""
Οπτική ανάλυση Reel + εξαγωγή viral γωνιών.

Η διαφορά από έναν «περιγραφέα βίντεο»:

Ένας περιγραφέας λέει «μια κοπέλα χορεύει σε πάρτι».
Αυτό δεν βοηθά να γραφτεί λεζάντα — η λεζάντα δεν πρέπει να ΕΠΑΝΑΛΑΒΕΙ
αυτό που ήδη βλέπει ο θεατής.

Εδώ ζητάμε δύο ξεχωριστά πράγματα:
  (α) τι δείχνει το βίντεο            → `VideoContent`
  (β) τι ΔΕΝ δείχνει και πρέπει να    → `ViralAngle.caption_should_add`
      προσθέσει η λεζάντα

Το (β) είναι η στρατηγική απόφαση. Χωρίς αυτό, κάθε λεζάντα καταλήγει
περιγραφή — που είναι ο πιο σίγουρος τρόπος να μην πάρει καμία αντίδραση.
"""
from __future__ import annotations

from typing import Optional

from ..clients.llm.base import AnthropicClient, json_block, text_block
from ..config import Settings
from ..logging_setup import get_logger
from ..schemas import VideoAnalysis, VideoContent, VideoTechnical, ViralAngle, ViralSignals
from . import frames as F

log = get_logger("video.analyzer")

SYSTEM = """Είσαι ανώτερος αναλυτής viral περιεχομένου Instagram Reels με εξειδίκευση \
ΑΠΟΚΛΕΙΣΤΙΚΑ στην ελληνική αγορά. Δεν είσαι κειμενογράφος — είσαι αναλυτής \
συμπεριφοράς κοινού.

Το κοινό σου: Έλληνες χρήστες Instagram, 16-45, που κάνουν scroll στο κινητό, \
συνήθως βράδυ, με τον αντίχειρα έτοιμο να προσπεράσει.

ΤΙ ΞΕΡΕΙΣ ΓΙΑ ΤΟ ΕΛΛΗΝΙΚΟ ΚΟΙΝΟ:
- Αντιδρά έντονα στο «αυτό είμαι εγώ / αυτή είναι η παρέα μου» (αναγνώριση).
- Στέλνει σε φίλο όταν το Reel λειτουργεί ως ΣΧΟΛΙΟ για τη σχέση τους
  («ρε συ αυτό είσαι εσύ»), όχι όταν είναι απλώς αστείο.
- Σχολιάζει όταν διαφωνεί ή όταν του ζητείται να πάρει θέση — όχι σε γενικές ερωτήσεις.
- Αναγνωρίζει αμέσως το «ξένο» και το προσπερνά: μεταφρασμένα αγγλικά, στημένο ύφος,
  αμερικανικές αναφορές χωρίς αντίστοιχο εδώ.
- Οι τοπικές αναφορές (γειτονιές, συνήθειες, ελληνική οικογένεια, στρατός, φοιτητική
  ζωή, καφές, καλοκαίρι/νησί) δουλεύουν ΜΟΝΟ όταν είναι ακριβείς. Η γενικόλογη
  «ελληνικότητα» ενοχλεί.

ΚΑΝΟΝΕΣ ΑΝΑΛΥΣΗΣ:
1. Περιέγραψε ΜΟΝΟ ό,τι πραγματικά βλέπεις στα καρέ. Μη συμπληρώνεις κενά με εικασίες.
2. Αν κάτι δεν φαίνεται (π.χ. τι λέγεται χωρίς μεταγραφή), πες το ρητά — μη μαντεύεις.
3. Τα καρέ είναι σε χρονολογική σειρά· η χρονική στιγμή κάθε καρέ σου δίνεται.
   Τα πρώτα καρέ είναι το HOOK — εκεί κρίνεται αν ο χρήστης θα μείνει.
4. Οι βαθμολογίες 0-100 πρέπει να ξεχωρίζουν. Αν όλα παίρνουν 70-80, η ανάλυση
   είναι άχρηστη. Χρησιμοποίησε όλο το εύρος· το μέτριο περιεχόμενο παίρνει 30-45.
5. Στις γωνίες: κάθε γωνία πρέπει να προτείνει κάτι που η λεζάντα ΠΡΟΣΘΕΤΕΙ,
   όχι που επαναλαμβάνει. Αν το `caption_should_add` περιγράφει το βίντεο, η γωνία
   είναι λάθος.

Γράφε ΟΛΑ τα κείμενα στα ελληνικά."""

SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "2-3 προτάσεις: τι συμβαίνει"},
                "main_subject": {"type": "string"},
                "people_count": {"type": "integer", "description": "ευδιάκριτα άτομα στο προσκήνιο"},
                "people_description": {"type": "string"},
                "environment": {"type": "string"},
                "actions": {"type": "array", "items": {"type": "string"}},
                "facial_expressions": {"type": "string"},
                "emotions": {"type": "array", "items": {"type": "string"}},
                "mood": {"type": "string"},
                "energy_level": {"type": "string", "enum": ["χαμηλή", "μεσαία", "υψηλή"]},
                "visual_style": {"type": "string"},
                "editing_style": {"type": "string"},
                "pace": {"type": "string"},
                "on_screen_text": {"type": "array", "items": {"type": "string"},
                                   "description": "κείμενο ΜΕΣΑ στο βίντεο, αυτούσιο"},
                "audio_type": {"type": "string",
                               "enum": ["ομιλία", "μουσική", "και τα δύο", "σιωπή", "άγνωστο"]},
                "hook_description": {"type": "string",
                                     "description": "τι ακριβώς συμβαίνει στα πρώτα 3 δευτ."},
                "story_arc": {"type": "string"},
                "humor": {"type": "string", "description": "τύπος χιούμορ ή «κανένα»"},
                "aesthetic": {"type": "string"},
                "niche": {"type": "string"},
                "sub_niche": {"type": "string"},
                "target_audience": {"type": "string"},
                "potential_audience": {"type": "string",
                                       "description": "ποιο ΕΥΡΥΤΕΡΟ κοινό θα μπορούσε να το δει"},
                "cultural_markers": {"type": "array", "items": {"type": "string"},
                                     "description": "ορατά ελληνικά/πολιτισμικά στοιχεία"},
            },
            "required": ["summary", "main_subject", "people_count", "environment",
                         "mood", "energy_level", "hook_description", "niche",
                         "sub_niche", "target_audience", "audio_type"],
        },
        "signals": {
            "type": "object",
            "properties": {
                "curiosity_gap": {"type": "integer"},
                "emotional_trigger": {"type": "integer"},
                "relatability": {"type": "integer"},
                "shareability": {"type": "integer"},
                "commentability": {"type": "integer"},
                "save_potential": {"type": "integer"},
                "rewatch_potential": {"type": "integer"},
                "shock_factor": {"type": "integer"},
                "attention_hold": {"type": "integer",
                                   "description": "πιθανότητα να μη γίνει scroll στα πρώτα 3 δευτ."},
                "greek_cultural_fit": {"type": "integer"},
                "notes": {"type": "string", "description": "το ασθενέστερο σημείο του Reel"},
            },
            "required": ["curiosity_gap", "emotional_trigger", "relatability",
                         "shareability", "commentability", "save_potential",
                         "rewatch_potential", "shock_factor", "attention_hold",
                         "greek_cultural_fit", "notes"],
        },
        "angles": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "σύντομο όνομα γωνίας"},
                    "strategy": {
                        "type": "string",
                        "enum": ["περιέργεια", "ταύτιση", "χιούμορ", "συναίσθημα",
                                 "αντιπαράθεση", "αφήγηση", "ανοιχτός βρόχος",
                                 "ερώτηση", "κοινωνική παρατήρηση", "απρόσμενη οπτική"],
                    },
                    "rationale": {"type": "string"},
                    "why_greek_stops": {"type": "string",
                                        "description": "γιατί ΕΛΛΗΝΑΣ χρήστης σταματά το scroll"},
                    "why_comment": {"type": "string"},
                    "why_share": {"type": "string",
                                  "description": "σε ΠΟΙΟΝ το στέλνει και τι λέει στέλνοντάς το"},
                    "caption_should_add": {
                        "type": "string",
                        "description": "ΤΙ ΔΕΝ ΔΕΙΧΝΕΙ το βίντεο και πρέπει να δώσει η λεζάντα. "
                                       "ΑΠΑΓΟΡΕΥΕΤΑΙ να περιγράφει το βίντεο."},
                    "target_segment": {"type": "string"},
                    "strength": {"type": "integer", "description": "0-100"},
                    "risk": {"type": "string", "description": "τι μπορεί να πάει στραβά"},
                },
                "required": ["name", "strategy", "rationale", "why_greek_stops",
                             "why_comment", "why_share", "caption_should_add",
                             "target_segment", "strength", "risk"],
            },
        },
        "data_gaps": {
            "type": "array", "items": {"type": "string"},
            "description": "τι ΔΕΝ μπόρεσες να προσδιορίσεις από τα καρέ",
        },
    },
    "required": ["content", "signals", "angles", "data_gaps"],
}


def _technical_context(tech: VideoTechnical, transcript: dict) -> dict:
    return {
        "διάρκεια_δευτ": round(tech.duration_s, 2),
        "ανάλυση": f"{tech.width}x{tech.height}",
        "λόγος_πλευρών": tech.aspect_ratio,
        "κάθετο_για_reels": tech.aspect_ratio in ("9:16", "9:20", "10:16"),
        "fps": tech.fps,
        "έχει_ήχο": tech.has_audio,
        "αλλαγές_σκηνής": tech.cut_count,
        "κοψίματα_ανά_δευτ": tech.cuts_per_second,
        "μέσο_πλάνο_δευτ": tech.avg_shot_len_s,
        "κοψίματα_στο_hook": tech.hook_cut_count,
        "μεταγραφή_ήχου": transcript.get("text") or "(μη διαθέσιμη)",
        "μεταγραφή_διαθέσιμη": transcript.get("available", False),
        "σημείωση_ήχου": transcript.get("note", ""),
    }


def analyze(tech: VideoTechnical, llm: AnthropicClient, settings: Settings,
            transcript: Optional[dict] = None,
            user_context: str = "") -> VideoAnalysis:
    """Καρέ + τεχνικά → πλήρης ανάλυση με υποψήφιες γωνίες."""
    transcript = transcript or {"text": "", "available": False}
    if not tech.frame_paths:
        raise ValueError("Δεν υπάρχουν καρέ για ανάλυση.")

    content_blocks = [text_block(
        "Ανάλυσε αυτό το Instagram Reel για την ΕΛΛΗΝΙΚΗ αγορά.\n\n"
        "Ακολουθούν τα καρέ σε χρονολογική σειρά. Κάθε καρέ συνοδεύεται από "
        "τη χρονική του στιγμή. Τα πρώτα είναι το HOOK.")]

    for path, t in zip(tech.frame_paths, tech.frame_times):
        block = F.to_base64(path)
        if block is None:
            continue
        marker = " ← HOOK" if t <= settings.video.hook_window_s else ""
        content_blocks.append(text_block(f"[καρέ στο {t:.2f}s{marker}]"))
        content_blocks.append(block)

    content_blocks.append(json_block("τεχνικά_στοιχεία",
                                     _technical_context(tech, transcript)))
    if user_context:
        content_blocks.append(text_block(
            f"<συμφραζόμενα_από_τον_χρήστη>\n{user_context}\n</συμφραζόμενα_από_τον_χρήστη>"))
    content_blocks.append(text_block(
        "Δώσε: (1) τι δείχνει το βίντεο, (2) βαθμολογίες viral δυναμικού με "
        "πραγματική διαφοροποίηση, (3) 3-5 ΓΩΝΙΕΣ. Για κάθε γωνία, το "
        "`caption_should_add` πρέπει να λέει τι ΘΑ ΠΡΟΣΘΕΣΕΙ η λεζάντα — "
        "όχι τι δείχνει το βίντεο."))

    log.info("Ανάλυση %d καρέ με %s…", len(tech.frame_paths),
             settings.models.vision_model)
    raw = llm.structured(
        system=SYSTEM, content=content_blocks, schema=SCHEMA,
        tool_name="submit_analysis",
        tool_description="Υπέβαλε την πλήρη ανάλυση του Reel.",
        model=settings.models.vision_model,
        max_tokens=8000, thinking_budget=4000)

    return _build(raw, tech, transcript)


def _build(raw: dict, tech: VideoTechnical, transcript: dict) -> VideoAnalysis:
    c = raw.get("content", {}) or {}
    content = VideoContent(
        summary=c.get("summary", ""), main_subject=c.get("main_subject", ""),
        people_count=int(c.get("people_count") or 0),
        people_description=c.get("people_description", ""),
        environment=c.get("environment", ""), actions=c.get("actions") or [],
        facial_expressions=c.get("facial_expressions", ""),
        emotions=c.get("emotions") or [], mood=c.get("mood", ""),
        energy_level=c.get("energy_level", ""),
        visual_style=c.get("visual_style", ""),
        editing_style=c.get("editing_style", ""), pace=c.get("pace", ""),
        on_screen_text=c.get("on_screen_text") or [],
        spoken_transcript=transcript.get("text", ""),
        audio_type=c.get("audio_type", ""),
        hook_description=c.get("hook_description", ""),
        story_arc=c.get("story_arc", ""), humor=c.get("humor", ""),
        aesthetic=c.get("aesthetic", ""), niche=c.get("niche", ""),
        sub_niche=c.get("sub_niche", ""),
        target_audience=c.get("target_audience", ""),
        potential_audience=c.get("potential_audience", ""),
        cultural_markers=c.get("cultural_markers") or [],
    )
    s = raw.get("signals", {}) or {}
    signals = ViralSignals(**{k: int(s.get(k) or 0) for k in (
        "curiosity_gap", "emotional_trigger", "relatability", "shareability",
        "commentability", "save_potential", "rewatch_potential", "shock_factor",
        "attention_hold", "greek_cultural_fit")}, notes=s.get("notes", ""))

    angles = [ViralAngle(
        name=a.get("name", ""), strategy=a.get("strategy", ""),
        rationale=a.get("rationale", ""), why_greek_stops=a.get("why_greek_stops", ""),
        why_comment=a.get("why_comment", ""), why_share=a.get("why_share", ""),
        caption_should_add=a.get("caption_should_add", ""),
        target_segment=a.get("target_segment", ""),
        strength=int(a.get("strength") or 0), risk=a.get("risk", ""),
    ) for a in (raw.get("angles") or [])]

    notes = list(raw.get("data_gaps") or [])
    if not transcript.get("available"):
        notes.append(transcript.get("note")
                     or "Δεν υπάρχει μεταγραφή ήχου — η ανάλυση βασίζεται μόνο σε καρέ.")

    return VideoAnalysis(technical=tech, content=content, signals=signals,
                         angles=angles, analysis_notes=notes)
