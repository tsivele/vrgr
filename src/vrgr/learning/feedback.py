"""
Βρόχος ανατροφοδότησης — ο ισχυρότερος μηχανισμός μάθησης (απαίτηση #6, Βρόχος Γ).

Ό,τι μαθαίνει το σύστημα από ξένους λογαριασμούς είναι έμμεσο. Ό,τι μαθαίνει
από ΤΑ ΔΙΚΑ ΣΟΥ posts είναι άμεσο: ίδιο κοινό, ίδιος λογαριασμός, ίδιος
αλγόριθμος. Δώδεκα μετρημένες δικές σου δημοσιεύσεις αξίζουν περισσότερο από
χίλια ξένα posts.

Ροή:
    δημοσιεύεις  →  ./run.sh feedback <run_id> <url>
                 →  /v2/media/info/by/url  (ΠΡΑΓΜΑΤΙΚΗ απόδοση)
                 →  σύγκριση με την πρόβλεψη
                 →  ενίσχυση/αποδυνάμωση των μοτίβων που χρησιμοποιήθηκαν
                 →  βαθμονόμηση του σκορ
"""
from __future__ import annotations

import json
import time
from typing import Optional

from ..analysis import metrics as M
from ..clients.hiker import endpoints as E
from ..clients.hiker import normalize as N
from ..clients.hiker.client import HikerClient
from ..errors import VRGRError
from ..logging_setup import get_logger
from ..memory.patterns import PatternStore
from ..memory.repository import Repository

log = get_logger("learning.feedback")

# Κατώφλι επιτυχίας: V/F ≥ 3 σημαίνει ότι ο αλγόριθμος έσπρωξε το Reel
# αισθητά πέρα από το υπάρχον κοινό. Κάτω από αυτό, το post απλώς
# σερβιρίστηκε στους followers.
SUCCESS_VF = 3.0


class FeedbackLoop:
    def __init__(self, repo: Repository, patterns: PatternStore,
                 client: Optional[HikerClient] = None):
        self.repo = repo
        self.patterns = patterns
        self.client = client

    def record(self, run_id: str, posted_url: str,
               manual: Optional[dict] = None) -> dict:
        """
        Καταγράφει το ΠΡΑΓΜΑΤΙΚΟ αποτέλεσμα μιας πρότασης.

        `manual` επιτρέπει χειροκίνητη εισαγωγή μετρήσεων (π.χ. από το
        Insights σου, που δίνει shares/saves — δεδομένα που το HikerAPI
        δεν βλέπει για κανέναν λογαριασμό).
        """
        run = self.repo.get_run(run_id)
        if not run:
            raise VRGRError(f"Δεν βρέθηκε εκτέλεση με id «{run_id}». "
                            f"Δες τις διαθέσιμες: ./run.sh runs")

        measured = dict(manual or {})
        if not measured.get("views") and self.client is not None and posted_url:
            log.info("Ανάκτηση πραγματικής απόδοσης από HikerAPI…")
            payload = self.client.try_call(E.MEDIA_BY_URL, url=posted_url)
            if payload:
                posts = N.posts_from(payload, source="feedback")
                if posts:
                    post = posts[0]
                    if not post.followers_at_observation and post.creator_pk:
                        prof = self.client.try_call(E.USER_BY_ID, id=post.creator_pk)
                        creator = N.creator_from(prof) if prof else None
                        if creator:
                            post.followers_at_observation = creator.followers
                    post.normalized = M.compute(post)
                    measured = {
                        "media_id": post.media_id,
                        "followers": post.followers_at_observation,
                        "views": post.metrics.views,
                        "likes": post.metrics.likes,
                        "comments": post.metrics.comments,
                        "vf_ratio": post.normalized.vf_ratio,
                        "outlier_score": post.normalized.outlier_score,
                    }
                    # Το ίδιο το post μπαίνει στη μνήμη ως τεκμήριο.
                    self.repo.save_posts([post], run.get("niche", ""))

        if not measured.get("views"):
            raise VRGRError(
                "Δεν ήταν δυνατή η μέτρηση της απόδοσης. Δώσε τα νούμερα "
                "χειροκίνητα: --views 120000 --likes 4500 --comments 210 "
                "--followers 8300")

        # Υπολογισμός V/F αν δόθηκε χειροκίνητα
        if measured.get("vf_ratio") is None and measured.get("followers"):
            measured["vf_ratio"] = measured["views"] / max(1, measured["followers"])
        if measured.get("outlier_score") is None:
            from ..schemas import NormalizedMetrics
            nm = NormalizedMetrics(
                vf_ratio=measured.get("vf_ratio"),
                comment_rate=(measured["comments"] / measured["views"])
                if measured.get("comments") and measured.get("views") else None)
            measured["outlier_score"] = M.outlier_score(nm, measured.get("followers"))

        predicted = run.get("predicted_score")
        actual = measured.get("outlier_score")
        measured["predicted_score"] = predicted
        measured["posted_url"] = posted_url
        measured["measured_at"] = time.time()
        if predicted is not None and actual is not None:
            measured["error"] = round(actual - predicted, 2)

        self.repo.save_outcome(run_id, measured)
        succeeded = (measured.get("vf_ratio") or 0) >= SUCCESS_VF
        self._reinforce(run, succeeded, measured)

        log.info("Καταγράφηκε: V/F=%.1f, πραγματικό σκορ=%.1f, πρόβλεψη=%.1f",
                 measured.get("vf_ratio") or 0, actual or 0, predicted or 0)
        return {
            "run_id": run_id, "succeeded": succeeded,
            "predicted": predicted, "actual": actual,
            "error": measured.get("error"),
            "vf_ratio": measured.get("vf_ratio"),
            "views": measured.get("views"),
            "followers": measured.get("followers"),
            "patterns_updated": len(_pattern_keys(run)),
        }

    def _reinforce(self, run: dict, succeeded: bool, measured: dict) -> None:
        """
        Ενημέρωση των μοτίβων που ΠΡΑΓΜΑΤΙΚΑ χρησιμοποιήθηκαν.

        Το βάρος κλιμακώνεται με το μέγεθος του αποτελέσματος: ένα Reel με
        V/F 40 είναι ισχυρότερη απόδειξη από ένα με V/F 3,2. Ένα οριακό
        αποτέλεσμα δεν πρέπει να μετράει σαν θρίαμβος.
        """
        keys = _pattern_keys(run)
        if not keys:
            return
        vf = measured.get("vf_ratio") or 0.0
        if succeeded:
            weight = min(2.0, 0.6 + vf / 25.0)
        else:
            weight = min(1.2, 0.5 + (SUCCESS_VF - vf) / 8.0)
        niche = run.get("niche", "")
        for key in keys:
            kind = key.split(":", 1)[0]
            self.patterns.observe(
                key=key, kind={"hashtag": "hashtag",
                               "structure": "caption_structure",
                               "caption": "caption_structure"}.get(kind, kind),
                success=succeeded, niche=niche, weight=weight,
                sample_id=run.get("run_id", ""))
        log.info("Ενισχύθηκαν %d μοτίβα (%s, βάρος %.2f)", len(keys),
                 "επιτυχία" if succeeded else "αποτυχία", weight)

    def summary(self) -> dict:
        """Πόσο καλά προβλέπει το σύστημα — η μόνη τίμια αυτοαξιολόγηση."""
        outcomes = self.repo.outcomes()
        pairs = [(o["predicted_score"], o["outlier_score"]) for o in outcomes
                 if o.get("predicted_score") is not None
                 and o.get("outlier_score") is not None]
        result = {"n_outcomes": len(outcomes), "n_comparable": len(pairs)}
        if len(pairs) >= 3:
            errors = [abs(a - p) for p, a in pairs]
            result["mean_abs_error"] = round(sum(errors) / len(errors), 1)
            result["correlation"] = _spearman([p for p, _ in pairs],
                                              [a for _, a in pairs])
        successes = [o for o in outcomes if (o.get("vf_ratio") or 0) >= SUCCESS_VF]
        result["success_rate"] = (round(len(successes) / len(outcomes), 3)
                                  if outcomes else None)
        result["median_vf"] = _median([o["vf_ratio"] for o in outcomes
                                       if o.get("vf_ratio")])
        return result


def _pattern_keys(run: dict) -> list:
    try:
        return json.loads(run.get("pattern_keys_json") or "[]")
    except json.JSONDecodeError:
        return []


def _median(values: list):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return round(values[mid] if len(values) % 2 else
                 (values[mid - 1] + values[mid]) / 2, 2)


def _spearman(xs: list, ys: list) -> Optional[float]:
    """
    Συσχέτιση κατάταξης — το σωστό μέτρο εδώ.

    Το σκορ δεν φιλοδοξεί να προβλέψει ΤΙΜΗ, μόνο ΣΕΙΡΑ («αυτό είναι
    καλύτερο από εκείνο»). Το Spearman μετρά ακριβώς αυτό, ενώ το Pearson
    θα τιμωρούσε το σύστημα για κάτι που ποτέ δεν υποσχέθηκε.
    """
    n = len(xs)
    if n < 3:
        return None

    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None
