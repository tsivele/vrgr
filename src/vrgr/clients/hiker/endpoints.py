"""
Μητρώο endpoints HikerAPI — ΕΠΑΛΗΘΕΥΜΕΝΟ.

Κάθε καταχώρηση εδώ έχει διασταυρωθεί με το επίσημο
`https://api.hikerapi.com/openapi.json` (154 paths, ελέγχθηκε 2026-08-29).
Ο client ΑΡΝΕΙΤΑΙ να καλέσει path που δεν υπάρχει σε αυτό το μητρώο —
έτσι είναι αδύνατο να «εφευρεθεί» endpoint.

Για επανέλεγχο μετά από αλλαγές του παρόχου:
    ./run.sh verify-endpoints
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Endpoint:
    path: str
    required: tuple = ()
    optional: tuple = ()
    ttl_family: str = "search"        # profile|media|search|hashtag|system
    cost_units: int = 1               # σχετικό κόστος για το budget
    note: str = ""
    deprecated: bool = False
    replacement: str = ""


# ── Προφίλ ────────────────────────────────────────────────────────────
USER_BY_USERNAME = Endpoint(
    "/v2/user/by/username", ("username",), ("safe_int",), "profile", 1,
    "Προφίλ + follower_count. Το θεμέλιο κάθε κανονικοποίησης ως προς μέγεθος.")
USER_BY_ID = Endpoint(
    "/v2/user/by/id", ("id",), ("safe_int",), "profile", 1)

# ── Reels creator ─────────────────────────────────────────────────────
USER_CLIPS = Endpoint(
    "/v2/user/clips", (), ("user_id", "page_id", "safe_int"), "media", 1,
    "Reels με σελιδοποίηση page_id. Χρονολογική σειρά.")
USER_CLIPS_CHUNK = Endpoint(
    "/v1/user/clips/chunk", ("user_id",), ("end_cursor",), "media", 1)
USER_CLIPS_GQL = Endpoint(
    "/gql/user/clips", ("user_id",), ("max_id", "sort_by_views", "flat"), "media", 1,
    "ΚΡΙΣΙΜΟ: sort_by_views=true φέρνει πρώτα τα viral. Χωρίς αυτό θα "
    "ξοδεύαμε δεκάδες κλήσεις για να βρούμε τα outliers ενός creator.")

# ── Ανακάλυψη ανταγωνιστών ────────────────────────────────────────────
USER_SUGGESTED = Endpoint(
    "/v2/user/suggested/profiles", ("user_id",), ("expand_suggestion", "safe_int"),
    "profile", 1, "Παρόμοιοι λογαριασμοί κατά Instagram — seed ανταγωνιστών.")
USER_EXPLORE_BUSINESSES = Endpoint(
    "/v2/user/explore/businesses/by/id", ("user_id",), ("safe_int",), "profile", 1)

# ── Hashtags ──────────────────────────────────────────────────────────
HASHTAG_BY_NAME = Endpoint(
    "/v2/hashtag/by/name", ("name",), ("safe_int",), "hashtag", 1,
    "Δίνει media_count — το ΜΟΝΟ πραγματικό μέγεθος hashtag που υπάρχει. "
    "Δεν υπάρχει endpoint για δυσκολία/τάση: τα υπολογίζουμε εμείς (DERIVED).")
HASHTAG_TOP = Endpoint(
    "/v2/hashtag/medias/top", ("name",), ("page_id", "safe_int"), "hashtag", 1)
HASHTAG_RECENT = Endpoint(
    "/v2/hashtag/medias/recent", ("name",), ("page_id", "safe_int"), "hashtag", 1,
    "Recent vs Top: η σύγκρισή τους είναι ο ΜΟΝΟΣ τρόπος ανίχνευσης τάσης.")
HASHTAG_CLIPS = Endpoint(
    "/v1/hashtag/medias/clips/chunk", ("name",), ("max_id",), "hashtag", 1,
    "Μόνο Reels για το hashtag — ό,τι ακριβώς χρειαζόμαστε.")

# ── Αναζήτηση ─────────────────────────────────────────────────────────
SEARCH_REELS = Endpoint(
    "/v2/fbsearch/reels", ("query",), ("reels_max_id", "rank_token", "safe_int"),
    "search", 2, "Αναζήτηση Reels με ελληνικές λέξεις-κλειδιά — ο πιο άμεσος "
    "τρόπος να βρεθεί «τι δουλεύει τώρα σε αυτό το θέμα».")
SEARCH_ACCOUNTS = Endpoint(
    "/v2/fbsearch/accounts", ("query",), ("page_token", "safe_int"), "search", 1)
SEARCH_HASHTAGS = Endpoint(
    "/v2/search/hashtags", ("query",), ("page_token", "safe_int"), "search", 1)
SEARCH_HASHTAGS_TOP = Endpoint(
    "/v1/fbsearch/topsearch/hashtags", ("query",), (), "search", 1)
TOPSEARCH = Endpoint(
    "/gql/topsearch", ("query",), ("end_cursor", "flat"), "search", 1,
    "Λογαριασμοί + hashtags + τοποθεσίες σε μία κλήση.")

# ── Posts ─────────────────────────────────────────────────────────────
MEDIA_BY_CODE = Endpoint(
    "/v2/media/info/by/code", ("code",), ("safe_int",), "media", 1)
MEDIA_BY_ID = Endpoint(
    "/v2/media/info/by/id", ("id",), ("safe_int",), "media", 1)
MEDIA_BY_URL = Endpoint(
    "/v2/media/info/by/url", ("url",), ("safe_int",), "media", 1,
    "Χρησιμοποιείται στον βρόχο ανατροφοδότησης: πραγματική απόδοση "
    "δημοσιευμένου Reel vs πρόβλεψη του συστήματος.")
MEDIA_COMMENTS = Endpoint(
    "/v2/media/comments", ("id",), ("can_support_threading", "safe_int", "page_id"),
    "media", 1, "Πώς αντιδρά πραγματικά το ελληνικό κοινό — γλώσσα, τόνος, ερωτήσεις.")
MEDIA_COMMENTS_INFOS = Endpoint(
    "/v2/media/comments/infos", (), ("media_ids",), "media", 1,
    "Έως 10 media σε μία κλήση — φθηνό batch για comment_count.")
CLIPS_METADATA = Endpoint(
    "/gql/media/clips_metadata", (), ("media_ids",), "media", 1,
    "Audio/music metadata για έως 10 media μαζί.")

# ── Ήχος ──────────────────────────────────────────────────────────────
TRACK_BY_ID = Endpoint(
    "/v2/track/by/id", ("track_id",), ("page_id", "safe_int"), "media", 1)
SEARCH_MUSIC = Endpoint(
    "/v2/search/music", ("query",), ("next_max_id", "safe_int"), "search", 1)

# ── Τοποθεσίες (ελληνική γεωγραφική στόχευση) ─────────────────────────
PLACES_SEARCH = Endpoint(
    "/v3/fbsearch/places", ("query",), (), "search", 1)
LOCATION_TOP = Endpoint(
    "/v1/location/medias/top/chunk", ("location_pk",), ("max_id",), "media", 1)

# ── Σύστημα ───────────────────────────────────────────────────────────
BALANCE = Endpoint(
    "/sys/balance", (), (), "system", 0,
    "Υπόλοιπο credits — παρακολούθηση κόστους, δεν χρεώνεται.")


REGISTRY = {e.path: e for e in [
    USER_BY_USERNAME, USER_BY_ID,
    USER_CLIPS, USER_CLIPS_CHUNK, USER_CLIPS_GQL,
    USER_SUGGESTED, USER_EXPLORE_BUSINESSES,
    HASHTAG_BY_NAME, HASHTAG_TOP, HASHTAG_RECENT, HASHTAG_CLIPS,
    SEARCH_REELS, SEARCH_ACCOUNTS, SEARCH_HASHTAGS, SEARCH_HASHTAGS_TOP, TOPSEARCH,
    MEDIA_BY_CODE, MEDIA_BY_ID, MEDIA_BY_URL, MEDIA_COMMENTS,
    MEDIA_COMMENTS_INFOS, CLIPS_METADATA,
    TRACK_BY_ID, SEARCH_MUSIC,
    PLACES_SEARCH, LOCATION_TOP,
    BALANCE,
]}

TTL_FAMILIES = ("profile", "media", "search", "hashtag", "system")


# ── Ρητά ΜΗ διαθέσιμα ─────────────────────────────────────────────────
# Τεκμηριώνονται εδώ ώστε κανείς (άνθρωπος ή μοντέλο) να μην τα «θυμηθεί»
# ως υπαρκτά. Το report τα αναφέρει ως data gaps όταν επηρεάζουν απόφαση.
UNAVAILABLE = {
    "hashtag_difficulty":
        "Δεν υπάρχει endpoint δυσκολίας/ανταγωνιστικότητας hashtag. "
        "Υπολογίζεται από εμάς σε δείγμα top posts → σημαίνεται DERIVED.",
    "trending_hashtags_gr":
        "Δεν υπάρχει «trending hashtags στην Ελλάδα». Η τάση ανιχνεύεται "
        "μόνο συγκρίνοντας /v2/hashtag/medias/recent με /v2/hashtag/medias/top.",
    "historical_timeseries":
        "Το API δίνει στιγμιότυπο, όχι ιστορικό. Το ιστορικό το χτίζουμε "
        "εμείς αποθηκεύοντας διαδοχικά snapshots του ίδιου media_id.",
    "shares_saves":
        "Shares/saves/reach δεν είναι δημόσια για ξένα posts (μόνο ο "
        "ιδιοκτήτης τα βλέπει στο Insights). Δουλεύουμε με views/likes/comments.",
    "audience_demographics":
        "Δεν υπάρχουν δημογραφικά κοινού για ξένους λογαριασμούς.",
    "reach_impressions":
        "Reach/impressions δεν εκτίθενται· το `play_count` είναι ό,τι πλησιέστερο.",
}


def get(path: str) -> Endpoint:
    ep = REGISTRY.get(path)
    if ep is None:
        raise KeyError(
            f"Το endpoint {path!r} ΔΕΝ υπάρχει στο επαληθευμένο μητρώο. "
            f"Πρόσθεσέ το στο endpoints.py μόνο αφού το επιβεβαιώσεις στο "
            f"openapi.json του HikerAPI."
        )
    return ep
