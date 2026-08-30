"""
Εκτέλεση αναλύσεων στο παρασκήνιο, με ορατή πρόοδο.

Γιατί χρειάζεται: μια ανάλυση διαρκεί 3-5 λεπτά. Ένα HTTP αίτημα που
περιμένει τόσο θα χτυπήσει timeout στον browser ή σε οποιοδήποτε proxy.
Το αίτημα λοιπόν επιστρέφει αμέσως ένα `job_id`, και η διεπαφή ρωτά για
την πρόοδο.

Ένα Pipeline ανά διεργασία, με κλείδωμα: το SQLite σε WAL αντέχει
παράλληλες αναγνώσεις, αλλά δύο ταυτόχρονες αναλύσεις θα μοιράζονταν το
budget του HikerAPI χωρίς συντονισμό — και θα ξόδευαν διπλά credits για
επικαλυπτόμενη έρευνα. Η ουρά είναι φθηνότερη από την παραλληλία εδώ.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..logging_setup import get_logger

log = get_logger("web.jobs")

MAX_JOBS_KEPT = 40


class Job:
    def __init__(self, job_id: str, kind: str, label: str):
        self.id = job_id
        self.kind = kind
        self.label = label
        self.status = "queued"           # queued | running | done | error
        self.step = 0
        self.total = 12
        self.step_label = "Σε αναμονή…"
        self.step_detail = ""
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.result: Any = None
        self.error: str = ""

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "label": self.label,
            "status": self.status, "step": self.step, "total": self.total,
            "step_label": self.step_label, "step_detail": self.step_detail,
            "elapsed": round(self.elapsed, 1), "error": self.error,
            "has_result": self.result is not None,
        }


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._jobs: dict = {}
        self._order: list = []
        self._lock = threading.Lock()
        self._pipeline_lock = threading.Lock()
        self._pipeline = None

    # ── pipeline (τεμπέλικο, επαναχρησιμοποιούμενο) ──────────────────
    def pipeline(self):
        if self._pipeline is None:
            from ..pipeline.orchestrator import Pipeline
            self._pipeline = Pipeline(self.settings)
        return self._pipeline

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None

    # ── εργασίες ─────────────────────────────────────────────────────
    def _register(self, kind: str, label: str) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, label)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > MAX_JOBS_KEPT:
                self._jobs.pop(self._order.pop(0), None)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 12) -> list:
        with self._lock:
            ids = list(reversed(self._order))[:limit]
            return [self._jobs[i].to_dict() for i in ids if i in self._jobs]

    def start_analysis(self, video_path: Path, options: dict) -> Job:
        job = self._register("analyze", video_path.name)

        def run():
            # Η κατάσταση μένει «queued» όσο περιμένει το κλείδωμα: μια
            # εργασία σε ουρά που δηλώνει «τρέχει» με βήμα 0 μοιάζει κολλημένη.
            job.step_label = "Σε αναμονή — τρέχει άλλη ανάλυση…"
            try:
                with self._pipeline_lock:
                    job.status = "running"
                    job.started_at = time.time()
                    job.step_label = "Ξεκινά…"
                    def on_progress(step, total, label, detail):
                        job.step, job.total = step, total
                        job.step_label, job.step_detail = label, detail

                    result = self.pipeline().analyze(
                        video_path,
                        user_context=options.get("context", ""),
                        skip_research=bool(options.get("no_research")),
                        n_captions=int(options.get("captions") or 8),
                        benchmark_creators=options.get("creators") or None,
                        progress=on_progress)
                job.result = result
                job.status = "done"
                job.step, job.step_label = job.total, "Ολοκληρώθηκε"
                job.step_detail = f"σκορ {result.winner.score.total:.0f}/100" \
                    if result.winner else ""
            except Exception as exc:                  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.step_label = "Σφάλμα"
                log.error("Η εργασία %s απέτυχε: %s", job.id, job.error)
                log.debug("%s", traceback.format_exc())
            finally:
                job.finished_at = time.time()
                try:
                    video_path.unlink(missing_ok=True)   # καθαρισμός ανεβασμένου
                except Exception:                        # noqa: BLE001
                    pass

        threading.Thread(target=run, daemon=True, name=f"job-{job.id}").start()
        return job

    def start_research(self, target: str, budget: int = 25) -> Job:
        job = self._register("research", target)

        def run():
            job.status = "running"
            job.started_at = time.time()
            job.total = 1
            job.step_label = f"Έρευνα {target}"
            try:
                job.result = self._research(target, budget)
                job.status = "done"
                job.step = 1
                job.step_label = "Ολοκληρώθηκε"
            except Exception as exc:                  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished_at = time.time()

        threading.Thread(target=run, daemon=True, name=f"job-{job.id}").start()
        return job

    def _research(self, target: str, budget: int) -> dict:
        from ..analysis import metrics as M
        from ..research.collector import Collector, fill_hashtag_stat
        pipe = self.pipeline()
        if pipe.hiker is None:
            raise RuntimeError("Λείπει το HIKER_API_KEY.")
        pipe.hiker.reset_budget(budget)
        collector = Collector(pipe.hiker, pipe.repo)
        target = target.strip()

        if target.startswith("#"):
            tag = target.lstrip("#")
            posts, stat = collector.hashtag(tag)
            trend = collector.hashtag_trend(tag)
            collector.enrich_followers(posts, max_profiles=20)
            M.enrich(posts)
            # Δυσκολία και «χωράει μικρός;» προκύπτουν ΜΟΝΟ αφού γεμίσουν οι
            # followers — αλλιώς τα πεδία μένουν κενά και η καρτέλα δείχνει
            # μισή εικόνα.
            if stat is not None:
                fill_hashtag_stat(stat, [p for p in posts if p.metrics.views])
            ranked = M.rank_outliers(posts, min_score=0, greek_only=False, limit=25)
            return {"kind": "hashtag", "target": tag,
                    "stat": stat.model_dump(mode="json") if stat else None,
                    "trend": trend, "posts": _posts_json(ranked)}

        username = target.lstrip("@")
        creator, posts = collector.creator_reels(username)
        if not creator:
            raise RuntimeError(f"Δεν βρέθηκε ο @{username}")
        M.enrich(posts)
        ranked = M.rank_outliers(posts, min_score=0, greek_only=False, limit=25)
        return {"kind": "creator", "target": username,
                "creator": creator.model_dump(mode="json"),
                "summary": M.corpus_summary(posts), "posts": _posts_json(ranked)}


def _posts_json(posts: list) -> list:
    out = []
    for p in posts:
        out.append({
            "username": p.username, "url": p.url,
            "caption": (p.caption_body or "")[:220],
            "hashtags": p.hashtags[:12],
            "views": p.metrics.views, "likes": p.metrics.likes,
            "comments": p.metrics.comments,
            "followers": p.followers_at_observation,
            "vf_ratio": round(p.normalized.vf_ratio, 1) if p.normalized.vf_ratio else None,
            "outlier_score": p.normalized.outlier_score,
            "age_days": round(p.age_days, 1) if p.age_days is not None else None,
            "thumbnail": p.thumbnail_url,
        })
    return out
