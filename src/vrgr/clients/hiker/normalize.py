"""
Κανονικοποίηση αποκρίσεων HikerAPI → μοντέλα `schemas`.

Το πρόβλημα: 154 endpoints σε 5 «οικογένειες» (v1, v2, gql, g1/g2, a2)
επιστρέφουν το ίδιο πράγμα με ΔΙΑΦΟΡΕΤΙΚΗ δομή — άλλοτε
`{"media": {...}}`, άλλοτε `{"response": {"items": [...]}}`, άλλοτε
`{"response": {"sections": [{"layout_content": {"medias": [...]}}]}}`.

Η λύση δεν είναι να μαντέψουμε κάθε σχήμα. Είναι ένας αναδρομικός walker
που αναγνωρίζει ένα media/user dict από την ΥΠΟΓΡΑΦΗ των πεδίων του.
Έτσι νέα ή αλλαγμένα σχήματα δουλεύουν χωρίς αλλαγή κώδικα.
"""
from __future__ import annotations

from typing import Any, Iterator, Optional

from ... import greek as G
from ...schemas import Creator, ObservedPost, PostMetrics

MAX_DEPTH = 8

# Πεδία που προδίδουν media αντικείμενο του Instagram.
#
# Οι δύο ομάδες είναι απαραίτητες: τα endpoints /v1 και /v2 χρησιμοποιούν τα
# ονόματα του private API, ενώ τα /gql, /g1, /g2 επιστρέφουν σχήμα GraphQL με
# εντελώς άλλα ονόματα για ΤΑ ΙΔΙΑ δεδομένα. Χωρίς τη δεύτερη ομάδα, τα
# αποτελέσματα του /gql/user/clips — δηλαδή ακριβώς τα viral posts που
# ψάχνουμε — απορρίπτονταν σιωπηλά ως «μη media».
_MEDIA_HINTS = (
    # private API (v1/v2)
    "like_count", "play_count", "ig_play_count", "view_count",
    "comment_count", "taken_at", "media_type", "product_type",
    "video_duration", "caption_text",
    # GraphQL (gql/g1/g2)
    "taken_at_timestamp", "edge_liked_by", "edge_media_to_comment",
    "edge_media_preview_like", "edge_media_to_parent_comment",
    "edge_media_to_caption", "video_view_count", "video_play_count",
    "shortcode", "display_url", "is_video",
)
_USER_HINTS = ("follower_count", "edge_followed_by", "followed_by_count",
               "media_count", "biography", "is_private")


def _num(value: Any) -> Optional[int]:
    """HikerAPI επιστρέφει ints, strings ή None ανάλογα με το `safe_int`."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace(" ", "").strip()
        if cleaned.lstrip("-").isdigit():
            return int(cleaned)
    if isinstance(value, dict):                    # {"count": N} του GraphQL
        for k in ("count", "value"):
            if k in value:
                return _num(value[k])
    return None


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return None


def _looks_like_media(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    if not (("pk" in d) or ("id" in d) or ("code" in d)):
        return False
    return sum(1 for h in _MEDIA_HINTS if h in d) >= 2


def _looks_like_user(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    if "username" not in d:
        return False
    return any(h in d for h in _USER_HINTS)


def _walk(node: Any, predicate, depth: int = 0) -> Iterator[dict]:
    """Αναδρομική εύρεση όλων των dicts που ικανοποιούν το predicate."""
    if depth > MAX_DEPTH:
        return
    if isinstance(node, dict):
        if predicate(node):
            yield node
            # Ένα media μπορεί να περιέχει carousel children — δεν κατεβαίνουμε.
            return
        for value in node.values():
            yield from _walk(value, predicate, depth + 1)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, predicate, depth + 1)


# ── Χρήστες ───────────────────────────────────────────────────────────

def creator_from(payload: Any) -> Optional[Creator]:
    """Βγάζει το ΠΡΩΤΟ προφίλ από οποιαδήποτε απόκριση."""
    if not payload:
        return None
    root = payload.get("user") if isinstance(payload, dict) else None
    node = root if _looks_like_user(root) else next(_walk(payload, _looks_like_user), None)
    if node is None:
        return None
    return _creator(node)


def _creator(d: dict) -> Creator:
    followers = _num(_first(d, "follower_count", "followers", "edge_followed_by",
                            "followed_by_count")) or 0
    bio = str(_first(d, "biography", "bio") or "")
    name = str(_first(d, "full_name", "fullname") or "")
    username = str(_first(d, "username", "user_name") or "")
    return Creator(
        pk=str(_first(d, "pk", "pk_id", "id", "user_id") or ""),
        username=username,
        full_name=name,
        followers=followers,
        following=_num(_first(d, "following_count", "edge_follow")) or 0,
        media_count=_num(_first(d, "media_count", "edge_owner_to_timeline_media")) or 0,
        biography=bio,
        is_private=bool(_first(d, "is_private") or False),
        is_verified=bool(_first(d, "is_verified") or False),
        category=str(_first(d, "category", "category_name", "business_category_name") or ""),
        external_url=str(_first(d, "external_url") or ""),
        greek_confidence=G.greek_confidence(
            f"{name} {bio}",
            extra_signals=[_is_greek_profile_signal(d)],
        ),
    )


def _is_greek_profile_signal(d: dict) -> bool:
    blob = " ".join(str(d.get(k, "")) for k in
                    ("biography", "full_name", "external_url", "city_name",
                     "address_street", "category")).lower()
    return any(m in blob for m in
               ("greece", "ελλάδα", "ελλαδα", "athens", "αθήνα", "αθηνα",
                "thessaloniki", "θεσσαλονίκη", "θεσσαλονικη", "🇬🇷", ".gr",
                "cyprus", "κύπρος", "κυπρος", "patra", "πάτρα", "heraklion"))


def creators_from(payload: Any) -> list:
    seen, out = set(), []
    for node in _walk(payload, _looks_like_user):
        c = _creator(node)
        if c.pk and c.pk not in seen:
            seen.add(c.pk)
            out.append(c)
    return out


# ── Media ─────────────────────────────────────────────────────────────

def _caption_text(d: dict) -> str:
    cap = _first(d, "caption_text", "caption")
    if isinstance(cap, dict):
        return str(cap.get("text") or "")
    if isinstance(cap, str):
        return cap
    edges = ((d.get("edge_media_to_caption") or {}).get("edges") or [])
    if edges:
        return str((edges[0].get("node") or {}).get("text") or "")
    return ""


def _views(d: dict) -> Optional[int]:
    """
    Σειρά προτίμησης: play_count → ig_play_count → view_count.

    Το `play_count` είναι ο μετρητής που δείχνει το Instagram κάτω από ένα
    Reel. Το `view_count` σε ορισμένα σχήματα αφορά μόνο βίντεο feed.
    """
    for key in ("play_count", "ig_play_count", "view_count", "video_view_count",
                "video_play_count"):
        v = _num(d.get(key))
        if v is not None and v > 0:
            return v
    return None


def _music(d: dict) -> tuple:
    ci = d.get("clips_metadata") or d.get("clips") or {}
    if not isinstance(ci, dict):
        return "", "", None
    music = (ci.get("music_info") or {}).get("music_asset_info") or {}
    original = ci.get("original_sound_info") or {}
    if music:
        return (str(music.get("title") or ""),
                str(music.get("display_artist") or ""), False)
    if original:
        return (str(original.get("original_audio_title") or "Πρωτότυπος ήχος"),
                str(original.get("ig_artist", {}).get("username") or ""), True)
    return "", "", None


def post_from(d: dict, creator: Optional[Creator] = None,
              source: str = "") -> Optional[ObservedPost]:
    """Ένα media dict → κανονικό `ObservedPost`."""
    media_id = str(_first(d, "pk", "id", "media_id") or "")
    if not media_id:
        return None
    # Το Instagram επιστρέφει άλλοτε «12345» και άλλοτε «12345_67890»
    # (media_pk + user_pk). Κρατάμε μόνο το media_pk — ΑΛΛΑ μόνο όταν και τα
    # δύο μέρη είναι αριθμητικά. Τυφλό κόψιμο στο «_» θα κατέστρεφε κάθε άλλη
    # μορφή id, συγχωνεύοντας άσχετα posts σε ένα και εξαφανίζοντας δεδομένα
    # στο deduplication χωρίς κανένα σφάλμα.
    if "_" in media_id:
        head, _, tail = media_id.partition("_")
        if head.isdigit() and tail.isdigit():
            media_id = head

    user_node = d.get("user") or d.get("owner") or {}
    user = _creator(user_node) if _looks_like_user(user_node) else None
    if creator is not None and (user is None or not user.followers):
        user = creator

    caption = _caption_text(d)
    tags = G.extract_hashtags(caption)
    title, artist, is_original = _music(d)
    location = d.get("location") or {}
    loc_name = str(location.get("name") or "") if isinstance(location, dict) else ""

    metrics = PostMetrics(
        views=_views(d),
        likes=_num(_first(d, "like_count", "edge_liked_by", "edge_media_preview_like")),
        comments=_num(_first(d, "comment_count", "edge_media_to_comment",
                             "edge_media_to_parent_comment")),
        plays=_num(d.get("play_count")),
    )

    thumb = ""
    cands = ((d.get("image_versions2") or {}).get("candidates") or [])
    if cands:
        thumb = str(cands[0].get("url") or "")
    elif d.get("thumbnail_url"):
        thumb = str(d["thumbnail_url"])
    elif d.get("display_url"):
        thumb = str(d["display_url"])

    post = ObservedPost(
        media_id=media_id,
        code=str(_first(d, "code", "shortcode") or ""),
        creator_pk=user.pk if user else "",
        username=user.username if user else "",
        followers_at_observation=user.followers if user and user.followers else None,
        caption=caption,
        caption_body=G.caption_body(caption),
        hashtags=tags,
        mentions=[m.group(1) for m in G.MENTION_RE.finditer(caption)],
        metrics=metrics,
        taken_at=_num(_first(d, "taken_at", "taken_at_timestamp", "device_timestamp")),
        duration_s=(lambda v: float(v) if v else None)(
            _first(d, "video_duration", "video_duration_seconds")),
        product_type=str(_first(d, "product_type") or ""),
        media_type=_num(d.get("media_type")),
        music_title=title,
        music_artist=artist,
        is_original_audio=is_original,
        location_name=loc_name,
        thumbnail_url=thumb,
        source_endpoint=source,
    )
    post.greek_confidence = G.greek_confidence(
        caption,
        extra_signals=[
            any(G.is_greek_hashtag(t) for t in tags),
            _greek_location(loc_name),
            (user.greek_confidence > 0.6) if user else False,
        ],
    )
    post.language = "el" if post.greek_confidence >= 0.5 else "other"
    return post


def _greek_location(name: str) -> bool:
    if not name:
        return False
    n = G.normalize(name)
    return G.GREEK_RANGE.search(name) is not None or any(
        k in n for k in ("greece", "athens", "thessaloniki", "cyprus", "crete",
                         "mykonos", "santorini", "rhodes", "patras", "corfu"))


def posts_from(payload: Any, creator: Optional[Creator] = None,
               source: str = "") -> list:
    """Όλα τα media οποιασδήποτε απόκρισης, χωρίς διπλά."""
    seen, out = set(), []
    for node in _walk(payload, _looks_like_media):
        p = post_from(node, creator=creator, source=source)
        if p and p.media_id not in seen:
            seen.add(p.media_id)
            out.append(p)
    return out


def next_cursor(payload: Any) -> Optional[str]:
    """Το επόμενο page token — τα ονόματα διαφέρουν ανά endpoint."""
    if not isinstance(payload, dict):
        return None
    for key in ("next_page_id", "next_max_id", "end_cursor", "max_id",
                "page_token", "reels_max_id", "next_cursor"):
        v = payload.get(key)
        if isinstance(v, (str, int)) and str(v).strip() not in ("", "None", "null"):
            return str(v)
    resp = payload.get("response")
    if isinstance(resp, dict):
        return next_cursor(resp)
    return None


def hashtags_from(payload: Any) -> list:
    """[(name, media_count)] από αποκρίσεις αναζήτησης hashtag."""
    out, seen = [], set()

    def visit(node: Any, depth: int = 0) -> None:
        if depth > MAX_DEPTH:
            return
        if isinstance(node, dict):
            name = node.get("name") or node.get("hashtag_name")
            if isinstance(name, str) and name and "media_count" in node:
                tag = name.lstrip("#").lower()
                if tag not in seen:
                    seen.add(tag)
                    out.append((tag, _num(node.get("media_count"))))
            for v in node.values():
                visit(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                visit(v, depth + 1)

    visit(payload)
    return out


def comments_from(payload: Any) -> list:
    """[(text, like_count)] — για ανάλυση αντίδρασης κοινού."""
    out = []

    def visit(node: Any, depth: int = 0) -> None:
        if depth > MAX_DEPTH:
            return
        if isinstance(node, dict):
            if "text" in node and ("created_at" in node or "comment_like_count" in node
                                   or "like_count" in node):
                txt = str(node.get("text") or "").strip()
                if txt:
                    out.append((txt, _num(_first(node, "comment_like_count",
                                                 "like_count")) or 0))
                return
            for v in node.values():
                visit(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                visit(v, depth + 1)

    visit(payload)
    return out
