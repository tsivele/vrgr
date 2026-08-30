"""
Ολοκληρωμένος έλεγχος αλυσίδας — από αρχείο βίντεο ως τελική απόφαση.

Οι εξωτερικές υπηρεσίες (Anthropic, HikerAPI) αντικαθίστανται με ψεύτικες
που επιστρέφουν ΡΕΑΛΙΣΤΙΚΑ σχήματα. Έτσι ελέγχεται η πραγματική λογική —
σύνδεση βημάτων, κανονικοποίηση, μνήμη, σκοράρισμα, απόφαση — χωρίς
κόστος API και χωρίς κλειδιά.

Ο έλεγχος αποδεικνύει ότι το σύστημα:
  • διαβάζει αληθινό βίντεο και βγάζει καρέ
  • κανονικοποιεί δεδομένα Instagram και βρίσκει τα outliers
  • γράφει στη μνήμη και τα ξαναβρίσκει
  • παράγει, σκοράρει και ΕΠΙΛΕΓΕΙ έναν συνδυασμό
  • συνεχίζει να λειτουργεί όταν πέφτει το HikerAPI
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from vrgr.clients.hiker import endpoints as E
from vrgr.config import (EmbeddingConfig, HikerConfig, ModelConfig, Settings,
                         VideoConfig)


# ── ψεύτικες υπηρεσίες ────────────────────────────────────────────────

class FakeLLM:
    """Απαντά με έγκυρες δομές ανάλογα με το εργαλείο που ζητήθηκε."""

    def __init__(self):
        self.calls = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def structured(self, *, system, content, schema, tool_name="submit", **kw):
        self.calls.append(tool_name)
        if tool_name == "submit_analysis":
            return self._analysis()
        if tool_name == "submit_queries":
            return self._queries()
        if tool_name == "submit_captions":
            return self._captions()
        raise AssertionError(f"απροσδόκητο εργαλείο: {tool_name}")

    def text(self, **kw):
        return ""

    def usage_summary(self):
        return dict(self.usage)

    @staticmethod
    def _analysis():
        return {
            "content": {
                "summary": "Κοπέλα ετοιμάζεται και κοιτάζει επανειλημμένα το κινητό "
                           "περιμένοντας μήνυμα που δεν έρχεται.",
                "main_subject": "κοπέλα που περιμένει", "people_count": 1,
                "people_description": "μία γυναίκα, 20-30",
                "environment": "υπνοδωμάτιο με φυσικό φως",
                "actions": ["ετοιμάζεται", "κοιτάζει κινητό", "αναστενάζει"],
                "facial_expressions": "από προσμονή σε εκνευρισμό",
                "emotions": ["προσμονή", "εκνευρισμός", "αυτοσαρκασμός"],
                "mood": "τραγικωμική", "energy_level": "μεσαία",
                "visual_style": "φυσικό φως, χειροκίνητη κάμερα",
                "editing_style": "γρήγορα κοψίματα στον ρυθμό",
                "pace": "γρήγορος", "on_screen_text": ["ώρα 21:04", "ώρα 22:47"],
                "audio_type": "μουσική",
                "hook_description": "κοντινό στο κινητό με άδεια οθόνη ειδοποιήσεων",
                "story_arc": "προσμονή → αναμονή → παραίτηση",
                "humor": "αυτοσαρκασμός", "aesthetic": "καθημερινό ρεαλιστικό",
                "niche": "χιούμορ", "sub_niche": "σχέσεις",
                "target_audience": "γυναίκες 18-30 σε αβέβαιη σχέση",
                "potential_audience": "όποιος έχει περιμένει μήνυμα",
                "cultural_markers": ["ελληνικό σπίτι", "ελληνικά στο κινητό"],
            },
            "signals": {"curiosity_gap": 62, "emotional_trigger": 74,
                        "relatability": 91, "shareability": 84,
                        "commentability": 79, "save_potential": 22,
                        "rewatch_potential": 48, "shock_factor": 18,
                        "attention_hold": 71, "greek_cultural_fit": 83,
                        "notes": "Το τέλος είναι αδύναμο — σβήνει αντί να κλείνει."},
            "angles": [
                {"name": "Η σιωπή που ακούγεται", "strategy": "ταύτιση",
                 "rationale": "Η αναμονή μηνύματος είναι καθολική εμπειρία.",
                 "why_greek_stops": "Αναγνωρίζει αμέσως τη σκηνή — το έχει ζήσει.",
                 "why_comment": "Θέλει να πει «σε ποιον δεν έχει συμβεί».",
                 "why_share": "Στέλνει στη φίλη που της το είχε πει.",
                 "caption_should_add": "Το εσωτερικό μονόλογο που δεν ακούγεται.",
                 "target_segment": "γυναίκες 20-30", "strength": 86,
                 "risk": "Μπορεί να διαβαστεί ως γκρίνια."},
                {"name": "Το χρονόμετρο", "strategy": "χιούμορ",
                 "rationale": "Η ώρα στην οθόνη κάνει το αστείο.",
                 "why_greek_stops": "Η υπερβολή του χρόνου είναι κωμική.",
                 "why_comment": "Λέει το δικό του ρεκόρ αναμονής.",
                 "why_share": "Στέλνει σε φίλο ως πείραγμα.",
                 "caption_should_add": "Το νούμερο που δεν φαίνεται.",
                 "target_segment": "νέοι 18-28", "strength": 71, "risk": "Πιο ρηχό."},
                {"name": "Η ερώτηση χωρίς απάντηση", "strategy": "ερώτηση",
                 "rationale": "Προκαλεί τοποθέτηση.",
                 "why_greek_stops": "Ρωτιέται κάτι που τον αφορά.",
                 "why_comment": "Παίρνει θέση.", "why_share": "Λιγότερο πιθανό.",
                 "caption_should_add": "Το δίλημμα.", "target_segment": "όλοι",
                 "strength": 64, "risk": "Γενικόλογο."},
            ],
            "data_gaps": ["Δεν υπάρχει μεταγραφή — άγνωστο αν υπάρχει ομιλία."],
        }

    @staticmethod
    def _queries():
        return {
            "niche": "χιούμορ", "sub_niche": "σχέσεις",
            "search_keywords": ["οταν περιμενεις μηνυμα", "ελληνικες σχεσεις αστειο",
                                "περιμενω απαντηση", "σχεσεις χιουμορ"],
            "hashtags_broad": ["ελλαδα", "greece"],
            "hashtags_mid": ["σχεσεις", "ελληνικοχιουμορ", "greekmemes"],
            "hashtags_niche": ["ελληνικεςσχεσεις", "περιμενωμηνυμα", "ζευγαρακια"],
            "greeklish_variants": ["sxeseis ellinika", "perimeno minima"],
            "locations": [], "reasoning": "Όροι γύρω από την αναμονή μηνύματος.",
        }

    @staticmethod
    def _captions():
        texts = [
            ("Τρεις ώρες. Δεν μετράω, απλά ξέρω.", "ταύτιση", "σχόλιο",
             "τον εσωτερικό μονόλογο"),
            ("Το «σε λίγο σου γράφω» έχει γίνει μονάδα μέτρησης χρόνου στην Ελλάδα.",
             "κοινωνική παρατήρηση", "αποστολή σε φίλο", "κοινωνικό σχόλιο"),
            ("Ετοιμάστηκα για κάτι που δεν επιβεβαιώθηκε ποτέ. Κλασικά.",
             "χιούμορ", "σχόλιο", "την παραδοχή"),
            ("Πες μου ότι δεν είμαι η μόνη που ανοίγει τη συνομιλία για να δει "
             "αν χάλασε το ίντερνετ.", "ερώτηση", "σχόλιο", "την ενοχή"),
            ("Δεν περιμένω. Απλά τυχαίνει να κοιτάω.", "σύντομη κοφτή",
             "αποστολή σε φίλο", "την άρνηση"),
            ("Η πιο μεγάλη απόσταση δεν είναι Αθήνα–Θεσσαλονίκη. Είναι το "
             "«είδε» χωρίς απάντηση.", "συναίσθημα", "αποστολή σε φίλο",
             "τη μεταφορά"),
            ("Ώρα 21:04 ελπίδα. Ώρα 22:47 αξιοπρέπεια.", "αφήγηση",
             "επαναπροβολή", "τη χρονογραμμή"),
        ]
        return {"captions": [{"text": t, "strategy": s, "rationale": "Δοκιμή.",
                              "expected_reaction": r, "adds_what": a}
                             for t, s, r, a in texts],
                "notes": "Δοκιμαστική παραγωγή."}


class FakeHiker:
    """Επιστρέφει ρεαλιστικά σχήματα HikerAPI χωρίς δίκτυο."""

    class _Stats:
        calls = 24
        cache_hits = 3
        stale_hits = 0
        per_endpoint = {"/v2/hashtag/medias/top": {"calls": 8, "cache": 1,
                                                   "errors": 0, "ms": 900.0}}

    def __init__(self, fail: bool = False):
        self.stats = self._Stats()
        self.fail = fail

    def reset_budget(self, budget=None):
        pass

    def try_call(self, endpoint, **params):
        if self.fail:
            return None
        path = endpoint.path
        if path == E.HASHTAG_BY_NAME.path:
            return {"name": params["name"], "media_count": 85_000}
        if path in (E.HASHTAG_CLIPS.path, E.HASHTAG_TOP.path,
                    E.SEARCH_REELS.path, E.HASHTAG_RECENT.path):
            return {"response": {"items": _fake_items(params.get("name")
                                                      or params.get("query", "x"))}}
        if path == E.USER_BY_USERNAME.path:
            return {"user": _fake_user(params["username"], 6_400)}
        if path == E.USER_BY_ID.path:
            return {"user": _fake_user(f"u{params['id']}", 12_000)}
        if path in (E.USER_CLIPS_GQL.path, E.USER_CLIPS.path):
            return {"response": {"items": _fake_items("creator", n=6)}}
        return None

    def call(self, endpoint, **params):
        return self.try_call(endpoint, **params)

    def close(self):
        pass


def _fake_user(username, followers):
    return {"pk": f"pk_{username}", "username": username,
            "full_name": f"Χρήστης {username}", "follower_count": followers,
            "biography": "Αθήνα 🇬🇷", "is_private": False, "media_count": 210}


def _fake_items(seed, n=10):
    """Δείγμα με ΣΚΟΠΙΜΗ ανομοιογένεια: μικρά viral, μεγάλα μέτρια, μη ελληνικά."""
    profiles = [
        # (followers, views, likes, comments, ελληνικά, λεζάντα)
        (3_200, 980_000, 41_000, 2_100, True, "Ποιος άλλος περιμένει απάντηση; "
                                              "Πες μου ότι δεν είμαι μόνη"),
        (8_900, 1_450_000, 62_000, 3_400, True, "Τρεις ώρες και ακόμα «είδε». "
                                                "Κάνε tag αυτόν που το κάνει"),
        (2_100, 610_000, 28_000, 1_900, True, "Το κινητό μου έχει γίνει ρολόι "
                                              "αναμονής. Συμφωνείτε;"),
        (45_000, 88_000, 3_100, 95, True, "Καλή σας μέρα σε όλους"),
        (1_900_000, 2_400_000, 51_000, 800, True, "Νέο βίντεο στο κανάλι"),
        (5_400, 720_000, 33_000, 2_800, True, "Γράψε «εγώ» αν σου έχει τύχει "
                                              "να ετοιμαστείς για το τίποτα"),
        (12_000, 34_000, 900, 20, True, "Απλά μια μέρα ακόμα"),
        (7_800, 95_000, 2_400, 60, False, "Just another day in the city"),
        (900, 240_000, 12_000, 700, True, "Δεν περιμένω, απλά κοιτάω. Ποιος άλλος;"),
        (33_000, 51_000, 1_800, 40, True, "Ευχαριστώ για την υποστήριξη"),
    ]
    tag_sets = [["ελληνικοχιουμορ", "σχεσεις", "greekmemes"],
                ["ελληνικοχιουμορ", "περιμενωμηνυμα", "σχεσεις"],
                ["σχεσεις", "ζευγαρακια"], ["ελλαδα"], ["greece", "travel"]]
    items = []
    now = int(time.time())
    for i in range(min(n, len(profiles))):
        fol, views, likes, comments, is_greek, caption = profiles[i]
        tags = tag_sets[i % len(tag_sets)]
        # Ρεαλιστικά αριθμητικά media pk — το Instagram δεν επιστρέφει ποτέ
        # αλφαριθμητικά id, και τα ψεύτικα δεδομένα δεν πρέπει να είναι
        # πιο «εύκολα» από τα αληθινά.
        pk = str(3500000000000000000 + abs(hash(seed)) % 10 ** 9 * 100 + i)
        items.append({"media": {
            "pk": f"{pk}_{7000000 + i}", "code": f"C{seed}{i}", "media_type": 2,
            "product_type": "clips", "taken_at": now - (i + 1) * 3 * 86400,
            "caption": {"text": caption + " " + " ".join(f"#{t}" for t in tags)},
            "like_count": likes, "comment_count": comments, "play_count": views,
            "video_duration": 12.0 + i,
            "user": _fake_user(f"acc_{seed}_{i}", fol) if is_greek
            else {"pk": f"x{i}", "username": f"intl_{i}",
                  "follower_count": fol, "biography": "NYC"},
        }})
    return items


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    """Πραγματικό αρχείο βίντεο 8 δευτ. με 3 σκηνές και ήχο."""
    import imageio_ffmpeg
    out = tmp_path_factory.mktemp("video") / "reel.mp4"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:s=540x960:d=2.5",
        "-f", "lavfi", "-i", "color=c=blue:s=540x960:d=3",
        "-f", "lavfi", "-i", "color=c=green:s=540x960:d=2.5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", "30", "-c:a", "aac", "-shortest", "-y", str(out)], check=True)
    return out


def _settings(tmp_path) -> Settings:
    root = Path(__file__).resolve().parents[1]
    return Settings(
        hiker=HikerConfig(api_key="fake"),
        models=ModelConfig(anthropic_key="fake"),
        embeddings=EmbeddingConfig(provider="local"),
        video=VideoConfig(max_frames=6),
        data_dir=tmp_path / "data", config_dir=root / "config")


def _pipeline(tmp_path, hiker_fail: bool = False):
    from vrgr.pipeline.orchestrator import Pipeline
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    pipe = Pipeline(settings)
    pipe._llm = FakeLLM()
    pipe._hiker = FakeHiker(fail=hiker_fail)
    return pipe


# ── έλεγχοι ───────────────────────────────────────────────────────────

def test_full_pipeline_produces_a_decision(sample_video, tmp_path):
    pipe = _pipeline(tmp_path)
    try:
        result = pipe.analyze(sample_video, n_captions=7)
    finally:
        pipe.close()

    # 1. Βίντεο: πραγματικά καρέ, ανιχνευμένα κοψίματα
    assert len(result.video.technical.frame_paths) >= 4
    assert result.video.technical.cut_count >= 2
    assert result.video.technical.aspect_ratio == "9:16"

    # 2. Γωνία: επιλέχθηκε η ισχυρότερη ευθυγραμμισμένη με τα σήματα
    assert result.video.chosen_angle is not None
    assert result.video.chosen_angle.name == "Η σιωπή που ακούγεται"

    # 3. Έρευνα: τα μικρά viral ξεχώρισαν, τα μη ελληνικά αποκλείστηκαν
    assert result.research is not None
    assert result.research.greek_posts > 0
    assert len(result.research.outliers) >= 3
    assert all(p.greek_confidence >= 0.4 for p in result.research.outliers)

    # 4. Απόφαση: ΕΝΑΣ νικητής, στα ελληνικά, με hashtags
    winner = result.winner
    assert winner is not None
    from vrgr import greek as G
    assert G.greek_ratio(winner.caption.text) > 0.7
    assert 8 <= len(winner.hashtag_set.tags) <= 20
    assert 0 < winner.score.total <= 100

    # 5. Εναλλακτικές: διαφορετικές μεταξύ τους
    assert len(result.backups) >= 2
    ids = [b.caption.id for b in result.backups] + [winner.caption.id]
    assert len(ids) == len(set(ids))

    # 6. Τεκμήρια από πραγματικά posts
    assert result.evidence
    assert any(e.vf_ratio and e.vf_ratio > 5 for e in result.evidence)

    # 7. Καμία λεζάντα δεν περιέχει hashtags (επιλέγονται χωριστά)
    assert "#" not in winner.caption.text


def test_memory_persists_and_is_reused(sample_video, tmp_path):
    """Δεύτερη εκτέλεση πρέπει να ΒΡΙΣΚΕΙ ό,τι έμαθε η πρώτη."""
    pipe = _pipeline(tmp_path)
    try:
        pipe.analyze(sample_video, n_captions=6)
        stats_after_first = pipe.retriever.corpus_stats()
        patterns_after_first = pipe.patterns.stats()

        second = pipe.analyze(sample_video, n_captions=6)
    finally:
        pipe.close()

    assert stats_after_first["posts"] > 0
    assert stats_after_first["greek_posts"] > 0
    assert patterns_after_first["patterns"] > 0
    # Η δεύτερη εκτέλεση αξιοποίησε την ιστορική μνήμη
    assert second.memory_hits > 0


def test_degrades_honestly_when_hiker_is_down(sample_video, tmp_path):
    """
    Όταν πέφτει το HikerAPI το σύστημα ΔΕΝ σταματά — αλλά ΔΗΛΩΝΕΙ την
    έλλειψη και το σκορ πέφτει. Σιωπηλή υποβάθμιση θα ήταν το χειρότερο.
    """
    pipe = _pipeline(tmp_path, hiker_fail=True)
    try:
        result = pipe.analyze(sample_video, n_captions=6)
    finally:
        pipe.close()

    assert result.winner is not None                     # παρήγαγε αποτέλεσμα
    assert result.winner.score.evidence_multiplier < 1.0  # τιμωρήθηκε
    assert result.winner.score.confidence != "υψηλή"
    assert result.data_gaps                              # και το είπε


def test_no_research_flag_is_reported(sample_video, tmp_path):
    pipe = _pipeline(tmp_path)
    try:
        result = pipe.analyze(sample_video, skip_research=True, n_captions=6)
    finally:
        pipe.close()
    assert result.research is None
    assert any("παραλείφθηκε" in w.lower() or "ΠΑΡΑΛΕΙΦΘΗΚΕ" in w
               for w in result.warnings)
    assert result.winner.score.evidence_multiplier < 1.0


def test_report_renders_without_error(sample_video, tmp_path):
    from vrgr.report import render, render_compact
    pipe = _pipeline(tmp_path)
    try:
        result = pipe.analyze(sample_video, n_captions=6)
    finally:
        pipe.close()
    full = render(result, verbose=True)
    assert "VIRAL SCORE" in full and "ΤΕΚΜΗΡΙΑ" in full
    assert "ΕΤΟΙΜΟ ΓΙΑ ΕΠΙΚΟΛΛΗΣΗ" in full
    assert result.winner.caption.text in full
    assert len(render_compact(result)) > 20
