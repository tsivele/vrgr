"""
Ο ενορχηστρωτής: τα 12 βήματα από βίντεο σε απόφαση.

ΑΡΧΗ ΥΠΟΒΑΘΜΙΣΗΣ: κάθε βήμα μετά την ανάλυση βίντεο μπορεί να αποτύχει
χωρίς να καταρρεύσει η εκτέλεση. Αν πέσει το HikerAPI, το σύστημα συνεχίζει
με μνήμη μόνο — και το ΔΗΛΩΝΕΙ, και το σκορ πέφτει ανάλογα. Μια ανάλυση που
«πέτυχε» κρύβοντας ότι δεν είχε δεδομένα θα ήταν χειρότερη από αποτυχία.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from .. import greek as G
from ..analysis import metrics as M
from ..cache.store import HttpCache
from ..clients.hiker.client import HikerClient
from ..clients.llm.base import AnthropicClient
from ..clients.llm.embeddings import build_embedder
from ..config import Settings
from ..errors import ConfigError, VideoError
from ..generation import captions as CAP
from ..generation import hashtags as HT
from ..logging_setup import get_logger
from ..memory.db import Database
from ..memory.patterns import PatternStore, hashtag_key, structure_key
from ..memory.repository import Repository
from ..memory.retrieval import MemoryRetriever
from ..research import mining as MINE
from ..research import planner as PLAN
from ..research.collector import Collector
from .. import niches as NICHE
from ..schemas import (AnalysisResult, EvidenceItem, MinedPatterns, Origin,
                       ResearchBundle, VideoAnalysis, ViralAngle)
from ..scoring import ranking as RANK
from ..scoring.viral_score import ViralScorer
from ..video import analyzer as VA
from ..video import audio as AUD
from ..video import frames as FR
from ..video import probe as PR

log = get_logger("pipeline")


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(settings.db_path)
        self.embedder = build_embedder(settings.embeddings)
        self.repo = Repository(self.db, self.embedder)
        self.retriever = MemoryRetriever(self.db, self.embedder)
        self.patterns = PatternStore(self.db)
        self.scorer = ViralScorer(settings.config_dir)
        self._llm: Optional[AnthropicClient] = None
        self._hiker: Optional[HikerClient] = None
        self._cache: Optional[HttpCache] = None

    # ── τεμπέλικη αρχικοποίηση εξωτερικών υπηρεσιών ──────────────────
    @property
    def llm(self) -> AnthropicClient:
        if self._llm is None:
            self._llm = AnthropicClient(self.settings.require_models())
        return self._llm

    @property
    def hiker(self) -> Optional[HikerClient]:
        if self._hiker is None and self.settings.hiker.enabled:
            self._cache = HttpCache(self.settings.cache_path)
            self._hiker = HikerClient(self.settings.hiker, self._cache)
        return self._hiker

    # ── κύρια ροή ────────────────────────────────────────────────────
    def analyze(self, video_path: Path, user_context: str = "",
                skip_research: bool = False, n_captions: int = 8,
                benchmark_creators: Optional[list] = None,
                progress: Optional[Callable] = None) -> AnalysisResult:
        """
        `progress(step, total, label, detail)` καλείται σε κάθε βήμα.

        Υπάρχει επειδή η ανάλυση διαρκεί λεπτά: χωρίς ορατή πρόοδο, μια
        διεπαφή που περιμένει 5 λεπτά είναι δυσδιάκριτη από μια κολλημένη.
        """
        started = time.time()
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        result = AnalysisResult(run_id=run_id)

        def step(n: int, label: str, detail: str = "") -> None:
            log.info("[%d/12] %s%s", n, label, f" — {detail}" if detail else "")
            if progress is not None:
                try:
                    progress(n, 12, label, detail)
                except Exception:                  # noqa: BLE001
                    pass                           # η πρόοδος δεν ρίχνει ποτέ την ανάλυση

        log.info("═══ Εκτέλεση %s ═══", run_id)

        # ΒΗΜΑ 1 — Ανάλυση βίντεο (ντετερμινιστικά)
        step(1, "Τεχνική ανάλυση βίντεο")
        tech = PR.full_probe(video_path, self.settings.video.scene_threshold,
                             self.settings.video.hook_window_s)
        work_dir = self.settings.media_dir / run_id
        tech = FR.sample(video_path, tech, work_dir / "frames",
                         self.settings.video.max_frames,
                         self.settings.video.hook_window_s,
                         self.settings.video.frame_width)
        if not tech.frame_paths:
            raise VideoError("Δεν εξήχθη κανένα καρέ — το αρχείο ίσως είναι κατεστραμμένο.")

        transcript = {"text": "", "available": False}
        if tech.has_audio:
            audio = AUD.extract_audio(video_path, work_dir)
            if audio:
                transcript = AUD.transcribe(audio, self.settings.video)

        # ΒΗΜΑ 2+3 — Οπτική κατανόηση + γωνίες
        step(2, "Οπτική ανάλυση καρέ με AI")
        analysis: VideoAnalysis = VA.analyze(tech, self.llm, self.settings,
                                             transcript, user_context)
        step(3, "Επιλογή viral γωνίας", f"{len(analysis.angles)} υποψήφιες")
        analysis.chosen_angle = self._choose_angle(analysis)
        result.video = analysis
        if analysis.chosen_angle is None:
            raise ConfigError("Το μοντέλο δεν επέστρεψε καμία viral γωνία.")
        angle = analysis.chosen_angle
        log.info("     → «%s» (%s, ισχύς %d)", angle.name, angle.strategy, angle.strength)

        # ΒΗΜΑ 4 — Σχεδιασμός έρευνας
        step(4, "Σχεδιασμός ελληνικής έρευνας")
        budget = self.settings.hiker.budget_per_run
        plan = PLAN.plan(analysis.content, angle, self.llm,
                         self.settings.config_dir, budget,
                         self.settings.models.fast_model)
        # ΔΥΟ ΞΕΧΩΡΙΣΤΑ ΠΡΑΓΜΑΤΑ:
        #   `niche_label` — η ελεύθερη περιγραφή του μοντέλου, για το report
        #   `niche`       — κανονικό κλειδί, για μνήμη και μοτίβα
        # Χωρίς τον διαχωρισμό, το ίδιο βίντεο δίνει κάθε φορά άλλο κλειδί
        # («Lifestyle & Personality Creators» vs «Lifestyle / Προσωπικό Brand»)
        # και η μάθηση δεν συσσωρεύεται ΠΟΤΕ.
        niche_label = plan["niche"] or analysis.content.niche
        niche = NICHE.canonical(niche_label, analysis.content.sub_niche)
        log.info("     → niche «%s» → κλειδί «%s»", niche_label, niche)

        # ΒΗΜΑ 5 — Έρευνα HikerAPI
        research: Optional[ResearchBundle] = None
        if skip_research:
            result.warnings.append(
                "Η έρευνα HikerAPI παραλείφθηκε κατόπιν εντολής (--no-research). "
                "Το σκορ είναι περιορισμένο λόγω έλλειψης τεκμηρίων.")
            step(5, "Έρευνα HikerAPI", "παραλείφθηκε")
        elif self.hiker is None:
            result.warnings.append(
                "Λείπει το HIKER_API_KEY — δεν έγινε έρευνα πραγματικών δεδομένων. "
                "Το σύστημα λειτουργεί μόνο με μνήμη και ανάλυση βίντεο.")
            result.data_gaps.append("Καμία φρέσκια απόδειξη από το Instagram.")
            step(5, "Έρευνα HikerAPI", "αδύνατη — λείπει κλειδί")
        else:
            step(5, "Έρευνα HikerAPI", f'{len(plan["queries"])} ερωτήματα')
            self.hiker.reset_budget(budget)
            seeds = benchmark_creators if benchmark_creators is not None \
                else PLAN.load_seed_creators(self.settings.config_dir)
            collector = Collector(self.hiker, self.repo)
            research = collector.collect(plan["queries"], seeds, run_id)
            result.research = research
            result.api_calls = research.api_calls
            if research.degraded:
                result.warnings.append(f"Υποβαθμισμένη έρευνα: {research.degraded_reason}")
            for err in research.errors:
                result.warnings.append(err)

        # ΒΗΜΑ 6 — Αποθήκευση στη μνήμη
        if research and research.posts:
            step(6, "Αποθήκευση στη μνήμη", f"{len(research.posts)} posts")
            for creator in research.creators.values():
                self.repo.upsert_creator(creator, niche)
            stats = self.repo.save_posts(research.posts, niche,
                                         analysis.content.sub_niche)
            for tag_stat in research.hashtag_stats.values():
                self.repo.save_hashtag_stat(tag_stat)
            log.info("     → %d νέα posts, %d νέα στιγμιότυπα",
                     stats["new_posts"], stats["new_snapshots"])
        else:
            step(6, "Αποθήκευση στη μνήμη", "κανένα νέο δεδομένο")

        # ΒΗΜΑ 7 — Ανάκτηση από ιστορική μνήμη
        step(7, "Αναζήτηση ιστορικών αναλόγων")
        memory_posts = self.retriever.similar_to_video(
            analysis.content, angle, limit=25, niche=niche)
        result.memory_hits = len(memory_posts)
        log.info("     → %d ιστορικά ανάλογα", len(memory_posts))

        # ΒΗΜΑ 8 — Εξόρυξη μοτίβων (φρέσκα + ιστορικά)
        step(8, "Εξόρυξη μοτίβων")
        mined = self._mine(research, memory_posts)
        result.patterns = mined
        memory_support = self._memory_support(mined, niche)

        # ΒΗΜΑ 9 — Παραγωγή λεζαντών
        step(9, "Παραγωγή λεζαντών με AI")
        corpus = self._corpus_captions(research, memory_posts)
        candidates = CAP.generate(analysis, angle, self.llm, mined, memory_support,
                                  corpus, self.settings.models.writer_model,
                                  n_captions)
        candidates = CAP.diversify(candidates, max_n=n_captions)
        if not candidates:
            raise ConfigError("Δεν παρήχθη καμία αποδεκτή λεζάντα.")
        log.info("     → %d υποψήφιες μετά τους ελέγχους", len(candidates))

        # ΒΗΜΑ 10 — Χαρτοφυλάκια hashtags
        step(10, "Κατασκευή χαρτοφυλακίων hashtags")
        memory_tags = self.repo.top_hashtags_for_niche(niche, limit=40)
        hashtag_stats = research.hashtag_stats if research else {}
        # Τα usernames όσων creators είδαμε: χρειάζονται για να απορριφθούν τα
        # προσωπικά τους brand hashtags, που δεν ωφελούν κανέναν άλλον.
        seen_creators = {p.username for p in (research.posts if research else [])
                         if p.username}
        seen_creators |= {c.username for c in (research.creators.values()
                                               if research else [])}
        hcands = HT.build_candidates(analysis.content, angle, mined, hashtag_stats,
                                     memory_tags, plan["terms"], self.repo, niche,
                                     creator_usernames=seen_creators)
        hsets = HT.build_sets(hcands, mined=mined, target_size=14)
        if not hsets:
            raise ConfigError("Δεν κατασκευάστηκε κανένα σετ hashtags.")
        log.info("     → %d χαρτοφυλάκια από %d υποψήφια hashtags",
                 len(hsets), len(hcands))

        # ΒΗΜΑ 11 — Κοινό σκοράρισμα και κατάταξη
        step(11, "Σκοράρισμα συνδυασμών", f"{len(candidates) * len(hsets)} ζεύγη")
        ranked = RANK.rank(analysis, angle, candidates, hsets, self.scorer,
                           research, mined, memory_support, result.memory_hits,
                           top_n=6)
        if not ranked:
            raise ConfigError("Το σκοράρισμα δεν επέστρεψε αποτέλεσμα.")
        winner = ranked[0]
        result.winner = winner
        result.backups = ranked[1:6]
        result.backup_hashtag_sets = RANK.alternative_sets(
            winner, hsets, analysis, angle, self.scorer, research, mined,
            memory_support, result.memory_hits, limit=3)
        result.why_won = RANK.explain(winner)

        # ΒΗΜΑ 12 — Τεκμήρια, μάθηση, αποθήκευση
        step(12, "Τεκμήρια και μάθηση")
        result.evidence = self._evidence(research, memory_posts, winner)
        result.data_gaps += self._data_gaps(research, analysis, mined)
        self._learn(mined, winner, niche)
        result.duration_s = round(time.time() - started, 1)
        self._persist(result, niche, analysis)
        log.info("═══ Ολοκληρώθηκε σε %.1fs — σκορ %.1f (%s) ═══",
                 result.duration_s, winner.score.total, winner.score.confidence)
        return result

    # ── βοηθητικά βήματα ─────────────────────────────────────────────
    def _choose_angle(self, analysis: VideoAnalysis) -> Optional[ViralAngle]:
        """
        Επιλογή γωνίας: ισχύς από το μοντέλο + ευθυγράμμιση με τα σήματα.

        Δεν εμπιστευόμαστε τυφλά το `strength`. Αν το βίντεο έχει χαμηλή
        commentability, μια γωνία «ερώτησης» δεν αξίζει την πρώτη θέση όσο
        υψηλά κι αν βαθμολογήθηκε.
        """
        if not analysis.angles:
            return None
        s = analysis.signals
        alignment = {
            "περιέργεια": s.curiosity_gap, "ανοιχτός βρόχος": s.curiosity_gap,
            "ταύτιση": s.relatability, "κοινωνική παρατήρηση": s.relatability,
            "χιούμορ": max(s.shareability, s.relatability),
            "συναίσθημα": s.emotional_trigger,
            "αντιπαράθεση": s.commentability, "ερώτηση": s.commentability,
            "αφήγηση": s.rewatch_potential, "απρόσμενη οπτική": s.shock_factor,
        }
        best, best_score = None, -1.0
        for angle in analysis.angles:
            aligned = alignment.get(angle.strategy, s.mean())
            combined = 0.62 * angle.strength + 0.38 * aligned
            if combined > best_score:
                best, best_score = angle, combined
        return best

    def _mine(self, research: Optional[ResearchBundle],
              memory_posts: list) -> MinedPatterns:
        """
        Συνδυασμός φρέσκων και ιστορικών (απαίτηση #17).

        Τα ιστορικά posts της μνήμης προστίθενται στο corpus εξόρυξης, ώστε
        τα μοτίβα να βασίζονται σε μεγαλύτερο δείγμα από ό,τι επιτρέπει το
        budget μιας εκτέλεσης.
        """
        posts = list(research.posts) if research else []
        outliers = list(research.outliers) if research else []

        historical = _rows_to_posts(memory_posts)
        if historical:
            M.enrich(historical)
            posts += historical
            outliers += [p for p in historical
                         if (p.normalized.outlier_score or 0) >= 45]

        if not posts:
            return MinedPatterns()
        return MINE.mine(posts, MINE_dedupe(outliers))

    def _memory_support(self, mined: MinedPatterns, niche: str) -> dict:
        keys = [structure_key("ερώτηση"), structure_key("cta"),
                structure_key("β_πρόσωπο")]
        keys += [hashtag_key(t) for t, _, _ in (mined.top_hashtags or [])[:12]]
        return self.patterns.support(keys, niche)

    def _corpus_captions(self, research: Optional[ResearchBundle],
                         memory_posts: list) -> list:
        """Λεζάντες επιτυχημένων posts — παραδείγματα ΔΟΜΗΣ και έλεγχος λογοκλοπής."""
        out = []
        if research:
            for p in research.outliers[:30]:
                if p.caption_body and p.greek_confidence >= 0.5:
                    out.append(p.caption_body)
        for row in memory_posts[:30]:
            body = row.get("caption_body") or ""
            if body and (row.get("greek_confidence") or 0) >= 0.5:
                out.append(body)
        seen, unique = set(), []
        for c in out:
            key = G.normalize(c)[:80]
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def _evidence(self, research: Optional[ResearchBundle], memory_posts: list,
                  winner) -> list:
        """
        Συγκεκριμένες παραπομπές σε ΠΡΑΓΜΑΤΙΚΑ posts.

        Προτεραιότητα σε όσα μοιράζονται hashtags με το νικητήριο σετ — αυτά
        είναι τα πραγματικά συγκρίσιμα, όχι απλώς τα πιο viral.
        """
        winner_tags = set(winner.hashtag_set.tags)
        items = []

        def relevance(post_tags: set) -> int:
            return len(post_tags & winner_tags)

        if research:
            ranked = sorted(
                research.outliers,
                key=lambda p: (-relevance(set(p.hashtags)),
                               -(p.normalized.outlier_score or 0)))
            # Ποικιλία λογαριασμών: πέντε posts του ίδιου creator δεν είναι
            # πέντε τεκμήρια — είναι ένα. Κρατάμε έως δύο ανά λογαριασμό ώστε
            # τα τεκμήρια να αντιπροσωπεύουν πράγματι διαφορετικές επιτυχίες.
            per_creator: dict = {}
            diverse = []
            for p in ranked:
                key = p.creator_pk or p.username
                if per_creator.get(key, 0) >= 2:
                    continue
                per_creator[key] = per_creator.get(key, 0) + 1
                diverse.append(p)
            ranked = diverse
            for p in ranked[:6]:
                shared = sorted(set(p.hashtags) & winner_tags)
                items.append(EvidenceItem(
                    username=p.username, url=p.url,
                    followers=p.followers_at_observation,
                    views=p.metrics.views,
                    vf_ratio=round(p.normalized.vf_ratio, 1)
                    if p.normalized.vf_ratio else None,
                    caption_excerpt=(p.caption_body or "")[:150],
                    hashtags=p.hashtags[:12],
                    age_days=round(p.age_days, 1) if p.age_days else None,
                    why_relevant=(f"Κοινά hashtags με την πρότασή μας: "
                                  f"{', '.join('#' + t for t in shared[:4])}."
                                  if shared else
                                  f"Δυσανάλογη απόδοση ({M.tier(p.normalized)}) "
                                  f"στο ίδιο niche."),
                    origin=Origin.OBSERVED))
        for row in memory_posts[:3]:
            items.append(EvidenceItem(
                username=row.get("username", ""),
                url=f"https://www.instagram.com/reel/{row.get('code','')}/"
                    if row.get("code") else "",
                followers=row.get("followers"), views=row.get("views"),
                vf_ratio=round(row["vf_ratio"], 1) if row.get("vf_ratio") else None,
                caption_excerpt=(row.get("caption_body") or "")[:150],
                hashtags=row.get("hashtags", [])[:12],
                why_relevant=f"Από τη μνήμη — σημασιολογική ομοιότητα "
                             f"{row.get('similarity', 0):.2f}.",
                origin=Origin.OBSERVED))
        return items

    def _data_gaps(self, research: Optional[ResearchBundle],
                   analysis: VideoAnalysis, mined: MinedPatterns) -> list:
        from ..clients.hiker.endpoints import UNAVAILABLE
        gaps = list(analysis.analysis_notes)
        gaps.append(UNAVAILABLE["shares_saves"])
        if research is None:
            gaps.append(UNAVAILABLE["historical_timeseries"])
            return gaps
        if not research.hashtag_stats:
            gaps.append("Δεν μετρήθηκε μέγεθος για κανένα hashtag — τα επίπεδα "
                        "(broad/mid/niche) είναι εκτίμηση, όχι μέτρηση.")
        elif any(s.difficulty is not None for s in research.hashtag_stats.values()):
            gaps.append(UNAVAILABLE["hashtag_difficulty"])
        if mined.outlier_sample_size < 5:
            gaps.append(f"Μόνο {mined.outlier_sample_size} ελληνικά outliers "
                        f"βρέθηκαν — τα μοτίβα είναι ενδεικτικά, όχι στατιστικά "
                        f"ασφαλή. Επανάλαβε αργότερα ή διεύρυνε τα seed accounts.")
        missing_followers = sum(1 for p in research.posts
                                if not p.followers_at_observation)
        if missing_followers > len(research.posts) * 0.4 and research.posts:
            gaps.append(f"Για {missing_followers}/{len(research.posts)} posts δεν "
                        f"ήταν διαθέσιμος ο αριθμός followers — ο δείκτης V/F "
                        f"υπολογίστηκε σε μικρότερο δείγμα.")
        return gaps

    def _learn(self, mined: MinedPatterns, winner, niche: str) -> None:
        """Ενημέρωση μακροπρόθεσμης μνήμης από τα ευρήματα αυτής της εκτέλεσης."""
        observations = MINE.patterns_to_observations(mined, niche)
        if observations:
            self.patterns.observe_batch(observations)
            log.info("     → %d μοτίβα ενημερώθηκαν στη μνήμη", len(observations))

    def _persist(self, result: AnalysisResult, niche: str,
                 analysis: VideoAnalysis) -> None:
        w = result.winner
        pattern_keys = [structure_key("ερώτηση")] if w.caption.has_question else []
        pattern_keys += [hashtag_key(t) for t in w.hashtag_set.tags[:10]]
        self.repo.save_run(result.run_id, {
            "created_at": result.created_at,
            "video_path": analysis.technical.path,
            "niche": niche, "sub_niche": analysis.content.sub_niche,
            "angle_name": analysis.chosen_angle.name if analysis.chosen_angle else "",
            "angle_strategy": analysis.chosen_angle.strategy if analysis.chosen_angle else "",
            "caption": w.caption.text, "hashtags": w.hashtag_set.tags,
            "predicted_score": w.score.total, "confidence": w.score.confidence,
            "api_calls": result.api_calls, "duration_s": result.duration_s,
            "result": result.model_dump(mode="json"),
            "pattern_keys": pattern_keys,
        })
        if self.hiker is not None:
            self.repo.log_api(result.run_id, self.hiker.stats.per_endpoint)

    def close(self) -> None:
        if self._hiker is not None:
            self._hiker.close()
        if self._cache is not None:
            self._cache.close()
        self.db.close()


def MINE_dedupe(posts: list) -> list:
    seen, out = set(), []
    for p in posts:
        if p.media_id not in seen:
            seen.add(p.media_id)
            out.append(p)
    return out


def _rows_to_posts(rows: list) -> list:
    """Γραμμές μνήμης → `ObservedPost` για ενιαία εξόρυξη μοτίβων."""
    from ..schemas import ObservedPost, PostMetrics
    out = []
    for r in rows:
        out.append(ObservedPost(
            media_id=r.get("media_id", ""), code=r.get("code", ""),
            username=r.get("username", ""), creator_pk=r.get("creator_pk", ""),
            caption=r.get("caption", ""), caption_body=r.get("caption_body", ""),
            hashtags=r.get("hashtags", []),
            followers_at_observation=r.get("followers"),
            metrics=PostMetrics(views=r.get("views"), likes=r.get("likes"),
                                comments=r.get("comments")),
            taken_at=r.get("taken_at"), duration_s=r.get("duration_s"),
            greek_confidence=r.get("greek_confidence") or 0.0,
            source_endpoint="memory"))
    return out
