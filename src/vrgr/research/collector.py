"""
Συλλογή πραγματικών δεδομένων Instagram μέσω HikerAPI.

ΤΟ ΚΡΙΣΙΜΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΛΥΝΕΙ ΑΥΤΟ ΤΟ MODULE:

Τα endpoints hashtag και αναζήτησης επιστρέφουν posts των οποίων το
ενσωματωμένο `user` object συχνά ΔΕΝ έχει `follower_count`. Χωρίς followers
δεν υπάρχει V/F — δηλαδή χάνεται ολόκληρη η κανονικοποίηση ως προς μέγεθος,
που είναι ο πυρήνας του συστήματος (απαίτηση #14).

Λύση: στάδιο ΕΜΠΛΟΥΤΙΣΜΟΥ. Μαζεύουμε τα υποψήφια posts, βρίσκουμε ποιοι
creators λείπουν, και τραβάμε τα προφίλ τους — αλλά ΜΟΝΟ για όσους αξίζει,
κατά σειρά προβολών, μέσα στο υπόλοιπο budget. Ένα post με 3 προβολές δεν
αξίζει μία κλήση API· ένα με 800.000 αξίζει.
"""
from __future__ import annotations

import concurrent.futures as cf
from typing import Optional

from .. import greek as G
from ..analysis import metrics as M
from ..clients.hiker import endpoints as E
from ..clients.hiker import normalize as N
from ..clients.hiker.client import HikerClient
from ..errors import BudgetExceeded, HikerAuthError
from ..logging_setup import get_logger
from ..schemas import Creator, HashtagStat, ResearchBundle, ResearchQuery

log = get_logger("research.collector")

MAX_WORKERS = 4
ENRICH_MIN_VIEWS = 5_000        # κάτω από αυτό δεν αξίζει κλήση προφίλ
ENRICH_MAX_PROFILES = 45


class Collector:
    def __init__(self, client: HikerClient, repo=None, max_workers: int = MAX_WORKERS):
        self.client = client
        self.repo = repo
        self.max_workers = max_workers

    # ── επιμέρους έρευνες ────────────────────────────────────────────
    def hashtag(self, tag: str, want_stats: bool = True) -> tuple:
        """
        Reels + στατιστικά ενός hashtag. Επιστρέφει `(posts, HashtagStat|None)`.

        Σειρά κλήσεων με βάση ΜΕΤΡΗΜΕΝΗ πληρότητα δεδομένων (έλεγχος σε
        ζωντανό API):

            /v2/hashtag/medias/top          λεζάντα ✓ likes ✓ σχόλια ✓ ΗΜΕΡΟΜΗΝΙΑ ✓
            /v1/hashtag/medias/clips/chunk  λεζάντα ✓ likes ✓ σχόλια ✓ ημερομηνία ✗

        Το `taken_at` είναι απαραίτητο για το σήμα τάσης και για την ηλικία των
        τεκμηρίων, γι' αυτό προηγείται το `top` παρότι το `clips` επιστρέφει
        αποκλειστικά Reels. Το δεύτερο μπαίνει μόνο ως συμπλήρωμα.
        """
        posts: list = []
        stat: Optional[HashtagStat] = None

        if want_stats:
            info = self.client.try_call(E.HASHTAG_BY_NAME, name=tag)
            if info:
                pairs = N.hashtags_from(info)
                count = next((c for n, c in pairs if n == tag.lower()), None)
                if count is None and pairs:
                    count = pairs[0][1]
                stat = HashtagStat(tag=tag, media_count=count,
                                   is_greek=G.is_greek_hashtag(tag))

        top = self.client.try_call(E.HASHTAG_TOP, name=tag)
        if top:
            posts += N.posts_from(top, source=f"hashtag_top:{tag}")
        dated = sum(1 for p in posts if p.taken_at and p.metrics.views)
        if dated < 12:
            clips = self.client.try_call(E.HASHTAG_CLIPS, name=tag)
            if clips:
                posts += N.posts_from(clips, source=f"hashtag_clips:{tag}")

        if stat is not None:
            stat.tier = classify_tier(stat.media_count)
        return _dedupe(posts), stat

    def hashtag_trend(self, tag: str) -> Optional[dict]:
        """
        Ανίχνευση τάσης — ΔΕΝ υπάρχει endpoint γι' αυτό.

        Το παράγουμε συγκρίνοντας τα «πρόσφατα» με τα «κορυφαία» posts του
        hashtag: αν τα πρόσφατα τραβάνε ήδη προβολές κοντά στα κορυφαία,
        το hashtag ανεβαίνει· αν υστερούν πολύ, είναι κορεσμένο.
        """
        recent = self.client.try_call(E.HASHTAG_RECENT, name=tag)
        top = self.client.try_call(E.HASHTAG_TOP, name=tag)
        if not recent or not top:
            return None
        rp = [p for p in N.posts_from(recent, source=f"recent:{tag}") if p.metrics.views]
        tp = [p for p in N.posts_from(top, source=f"top:{tag}") if p.metrics.views]
        if len(rp) < 4 or len(tp) < 4:
            return None
        import statistics
        med_r = statistics.median(p.metrics.views for p in rp)
        med_t = statistics.median(p.metrics.views for p in tp)
        ratio = med_r / med_t if med_t else 0.0
        if ratio >= 0.45:
            label = "ανοδικό"
        elif ratio >= 0.18:
            label = "σταθερό"
        else:
            label = "κορεσμένο"
        return {"tag": tag, "recent_median_views": int(med_r),
                "top_median_views": int(med_t), "ratio": round(ratio, 3),
                "label": label, "origin": "DERIVED"}

    def keyword(self, query: str) -> list:
        res = self.client.try_call(E.SEARCH_REELS, query=query)
        return _dedupe(N.posts_from(res, source=f"search:{query}")) if res else []

    def creator_reels(self, username: str, top_by_views: bool = True,
                      creator: Optional[Creator] = None,
                      max_backfill: int = 4) -> tuple:
        """
        Reels ενός creator, με τα ΙΣΤΟΡΙΚΑ viral του.

        ΜΕΤΡΗΜΕΝΟ ΣΕ ΖΩΝΤΑΝΟ API — γιατί η σειρά των κλήσεων είναι έτσι:

        Το `/gql/user/clips?sort_by_views=true` είναι ο μόνος τρόπος να δεις τα
        Reels ενός λογαριασμού ταξινομημένα κατά προβολές. ΟΜΩΣ επιστρέφει
        ΠΕΡΙΚΟΜΜΕΝΟ αντικείμενο: μόνο `pk`, `play_count`, `video_duration` —
        χωρίς λεζάντα, likes, σχόλια ή ημερομηνία. Είναι δηλαδή άχρηστο ως
        πηγή τεκμηρίων, αλλά πολύτιμο ως ΕΥΡΕΤΗΡΙΟ.

        Γι' αυτό:
          1. `/v2/user/clips` → πλήρη πρόσφατα Reels (η βάση των τεκμηρίων)
          2. `/gql/user/clips?sort_by_views` → ποια είναι τα κορυφαία όλων των
             εποχών· όσα λείπουν από το (1) είναι παλιά viral που δεν θα τα
             βρίσκαμε ποτέ χρονολογικά
          3. `/v2/media/info/by/id` για τα κορυφαία που λείπουν → πλήρη στοιχεία

        Ο περιορισμός `max_backfill` κρατά το κόστος προβλέψιμο: τα παλιά viral
        είναι το πιο διδακτικό υλικό, αλλά δεν αξίζουν απεριόριστες κλήσεις.
        """
        if creator is None:
            prof = self.client.try_call(E.USER_BY_USERNAME, username=username)
            creator = N.creator_from(prof) if prof else None
        if creator is None or not creator.pk:
            return None, []
        if creator.is_private:
            log.info("@%s είναι ιδιωτικός — παράλειψη", username)
            return creator, []

        # (1) Πλήρη πρόσφατα Reels
        posts: list = []
        res = self.client.try_call(E.USER_CLIPS, user_id=creator.pk)
        if res:
            posts += N.posts_from(res, creator=creator,
                                  source=f"user_clips:{username}")
        if not posts:
            res = self.client.try_call(E.USER_CLIPS_CHUNK, user_id=creator.pk)
            if res:
                posts += N.posts_from(res, creator=creator,
                                      source=f"user_clips_chunk:{username}")
        posts = _dedupe(posts)

        # (2)+(3) Ιστορικά viral που λείπουν από το πρόσφατο παράθυρο
        if top_by_views and max_backfill > 0:
            posts += self._backfill_top_reels(creator, posts, max_backfill)

        for p in posts:            # κανένα endpoint δεν δίνει followers — τα βάζουμε εμείς
            if not p.followers_at_observation and creator.followers:
                p.followers_at_observation = creator.followers
            if not p.creator_pk:
                p.creator_pk = creator.pk
                p.username = creator.username
        return creator, _dedupe(posts)

    def _backfill_top_reels(self, creator: Creator, have: list,
                            limit: int) -> list:
        """Συμπλήρωση των κορυφαίων-όλων-των-εποχών που λείπουν."""
        index = self.client.try_call(E.USER_CLIPS_GQL, user_id=creator.pk,
                                     sort_by_views="true")
        if not index:
            return []
        ranked = N.posts_from(index, creator=creator, source="clips_index")
        known = {p.media_id for p in have}
        best_known = max((p.metrics.views or 0) for p in have) if have else 0

        missing = [p for p in ranked
                   if p.media_id not in known and (p.metrics.views or 0) > 0]
        missing.sort(key=lambda p: -(p.metrics.views or 0))
        # Αξίζει κλήση μόνο για ό,τι ξεπερνά τα ήδη γνωστά — αλλιώς πληρώνουμε
        # για δεδομένα που δεν αλλάζουν την εικόνα.
        worth = [p for p in missing if (p.metrics.views or 0) >= best_known * 0.6]
        out = []
        for p in worth[:limit]:
            full = self.client.try_call(E.MEDIA_BY_ID, id=p.media_id)
            if not full:
                continue
            detailed = N.posts_from(full, creator=creator,
                                    source=f"top_backfill:{creator.username}")
            if detailed:
                out.append(detailed[0])
        if out:
            log.info("@%s: +%d ιστορικά viral μέσω ευρετηρίου προβολών",
                     creator.username, len(out))
        return out

    def similar_creators(self, creator_pk: str, limit: int = 8) -> list:
        res = self.client.try_call(E.USER_SUGGESTED, user_id=creator_pk)
        if not res:
            return []
        found = N.creators_from(res)
        greek = [c for c in found if c.greek_confidence >= 0.35 or
                 G.greek_ratio(c.full_name + c.biography) > 0.2]
        return (greek or found)[:limit]

    # ── εμπλουτισμός ─────────────────────────────────────────────────
    def enrich_followers(self, posts: list, max_profiles: int = ENRICH_MAX_PROFILES,
                         known: Optional[dict] = None) -> dict:
        """
        Συμπλήρωση follower counts που λείπουν.

        Χωρίς αυτό το βήμα, τα posts από hashtag/αναζήτηση είναι ΑΧΡΗΣΤΑ για
        κανονικοποίηση: ξέρουμε ότι έκαναν 400.000 προβολές αλλά όχι αν αυτό
        είναι θαύμα ή ρουτίνα.
        """
        known = dict(known or {})
        need: dict = {}
        for p in posts:
            if p.followers_at_observation or not p.creator_pk:
                continue
            if p.creator_pk in known:
                p.followers_at_observation = known[p.creator_pk].followers
                continue
            views = p.metrics.views or 0
            if views < ENRICH_MIN_VIEWS:
                continue
            # Κρατάμε ανά creator το post με τις περισσότερες προβολές ως κριτήριο.
            if p.creator_pk not in need or views > need[p.creator_pk][0]:
                need[p.creator_pk] = (views, p.username)

        if not need:
            return known
        ranked = sorted(need.items(), key=lambda kv: -kv[1][0])[:max_profiles]
        log.info("Εμπλουτισμός followers για %d creators (από %d υποψήφιους)",
                 len(ranked), len(need))

        def fetch(item):
            pk, (_, username) = item
            try:
                res = self.client.try_call(E.USER_BY_ID, id=pk)
                return pk, (N.creator_from(res) if res else None)
            except BudgetExceeded:
                return pk, None
            except Exception:                          # noqa: BLE001
                return pk, None

        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for pk, creator in pool.map(fetch, ranked):
                if creator and creator.followers:
                    known[pk] = creator

        for p in posts:
            if not p.followers_at_observation and p.creator_pk in known:
                p.followers_at_observation = known[p.creator_pk].followers
        return known

    # ── ενορχήστρωση ─────────────────────────────────────────────────
    def collect(self, queries: list, seed_creators: Optional[list] = None,
                run_id: str = "", greek_min: float = 0.4,
                max_creators: int = 6) -> ResearchBundle:
        bundle = ResearchBundle(run_id=run_id, queries=queries)
        all_posts: list = []
        creators: dict = {}

        def run_query(q: ResearchQuery):
            try:
                if q.kind == "hashtag":
                    posts, stat = self.hashtag(q.value)
                    return q, posts, stat
                if q.kind == "keyword":
                    return q, self.keyword(q.value), None
                return q, [], None
            except BudgetExceeded:
                raise
            except Exception as exc:                   # noqa: BLE001
                log.warning("Ερώτημα %s «%s» απέτυχε: %s", q.kind, q.value,
                            type(exc).__name__)
                return q, [], None

        try:
            with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for q, posts, stat in pool.map(run_query, queries):
                    all_posts.extend(posts)
                    if stat is not None:
                        bundle.hashtag_stats[stat.tag] = stat
        except BudgetExceeded as exc:
            bundle.degraded = True
            bundle.degraded_reason = str(exc)
            log.warning("%s", exc)
        except HikerAuthError as exc:
            bundle.degraded = True
            bundle.degraded_reason = str(exc)
            bundle.errors.append(str(exc))
            log.error("%s", exc)

        # Benchmark creators — παράλληλα.
        #
        # Σειριακά κοστίζει ~15 δευτ. ανά λογαριασμό (προφίλ + clips + backfill
        # ιστορικών viral) — δηλαδή 1,5 λεπτό για έξι. Ο ρυθμιστής ρυθμού του
        # client προστατεύει ήδη το API, οπότε η σειριακότητα δεν αγόραζε
        # τίποτα· απλώς έκανε την ανάλυση να μοιάζει κολλημένη.
        if seed_creators and not bundle.degraded:
            def fetch_creator(username: str):
                try:
                    return username, self.creator_reels(username), None
                except BudgetExceeded as exc:
                    return username, (None, []), exc
                except Exception as exc:               # noqa: BLE001
                    return username, (None, []), exc

            with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for username, (creator, posts), err in pool.map(
                        fetch_creator, seed_creators[:max_creators]):
                    if isinstance(err, BudgetExceeded):
                        bundle.degraded = True
                        bundle.degraded_reason = str(err)
                        continue
                    if err is not None:
                        log.info("Παράλειψη @%s: %s", username, type(err).__name__)
                        continue
                    if creator:
                        creators[creator.pk] = creator
                    all_posts.extend(posts)

        all_posts = _dedupe(all_posts)
        log.info("Συλλέχθηκαν %d μοναδικά posts", len(all_posts))

        # Εμπλουτισμός + κανονικοποίηση
        try:
            creators = self.enrich_followers(all_posts, known=creators)
        except BudgetExceeded as exc:
            bundle.degraded = True
            bundle.degraded_reason = str(exc)

        M.enrich(all_posts)

        greek_posts = [p for p in all_posts if p.greek_confidence >= greek_min]
        bundle.posts = all_posts
        bundle.creators = creators
        bundle.greek_posts = len(greek_posts)
        bundle.outliers = M.rank_outliers(all_posts, min_score=45.0,
                                          greek_only=True, min_greek=greek_min)

        # Στατιστικά hashtag από το πραγματικό δείγμα
        for tag, stat in bundle.hashtag_stats.items():
            sample = [p for p in all_posts
                      if tag in p.hashtags and p.metrics.views]
            fill_hashtag_stat(stat, sample)

        s = self.client.stats
        bundle.api_calls = s.calls
        bundle.cache_hits = s.cache_hits
        if s.stale_hits:
            bundle.errors.append(
                f"{s.stale_hits} αποκρίσεις σερβιρίστηκαν από ΛΗΓΜΕΝΟ cache "
                f"(το HikerAPI δεν απάντησε).")
        log.info("Έρευνα: %d posts, %d ελληνικά, %d outliers, %d κλήσεις, %d cache",
                 len(all_posts), bundle.greek_posts, len(bundle.outliers),
                 s.calls, s.cache_hits)
        return bundle


# ── βοηθητικά ─────────────────────────────────────────────────────────

def _dedupe(posts: list) -> list:
    seen, out = set(), []
    for p in posts:
        if p.media_id and p.media_id not in seen:
            seen.add(p.media_id)
            out.append(p)
    return out


def classify_tier(media_count: Optional[int]) -> str:
    """Κατηγοριοποίηση hashtag κατά μέγεθος — OBSERVED από media_count."""
    if media_count is None:
        return "unknown"
    if media_count >= 1_000_000:
        return "broad"
    if media_count >= 100_000:
        return "mid"
    if media_count >= 10_000:
        return "niche"
    return "micro"


def fill_hashtag_stat(stat: HashtagStat, sample: list) -> None:
    """Παράγωγα στατιστικά hashtag από πραγματικό δείγμα (DERIVED)."""
    if not sample:
        return
    import statistics
    views = [p.metrics.views for p in sample if p.metrics.views]
    fol = [p.followers_at_observation for p in sample if p.followers_at_observation]
    ages = [p.age_days for p in sample if p.age_days is not None]
    stat.sample_size = len(sample)
    if views:
        stat.median_views_top = float(statistics.median(views))
    if fol:
        stat.median_followers_top = float(statistics.median(fol))
        small = sum(1 for f in fol if f < 50_000)
        stat.small_account_share = round(small / len(fol), 3)
    if ages:
        stat.recency_days_median = round(float(statistics.median(ages)), 1)
    stat.difficulty = _difficulty(stat)


def _difficulty(stat: HashtagStat) -> Optional[float]:
    """
    Δυσκολία 0-100 — ΔΕΝ υπάρχει τέτοιο endpoint, το υπολογίζουμε εμείς.

    Τρεις συνιστώσες:
      • μέγεθος hashtag (περισσότερα posts = πιο δύσκολη κατάταξη)
      • διάμεσο followers των κορυφαίων (κυριαρχία μεγάλων = κλειστή πόρτα)
      • ποσοστό μικρών λογαριασμών στα κορυφαία (ΑΝΤΙΣΤΡΟΦΑ: αν χωράνε
        μικροί, η πόρτα είναι ανοιχτή)
    """
    import math
    parts, weights = [], []
    if stat.media_count:
        parts.append(min(100.0, 14.0 * math.log10(max(10, stat.media_count))))
        weights.append(0.40)
    if stat.median_followers_top:
        parts.append(min(100.0, 16.0 * math.log10(max(10, stat.median_followers_top))))
        weights.append(0.35)
    if stat.small_account_share is not None:
        parts.append(100.0 * (1.0 - stat.small_account_share))
        weights.append(0.25)
    if not parts:
        return None
    total = sum(w for w in weights)
    return round(sum(p * w for p, w in zip(parts, weights)) / total, 1)
