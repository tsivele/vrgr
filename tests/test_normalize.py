"""Κανονικοποίηση αποκρίσεων HikerAPI από ετερογενή σχήματα."""
from vrgr.clients.hiker import normalize as N

V2_CLIPS = {"response": {"items": [{"media": {
    "pk": "3500000000000000001", "id": "3500000000000000001_123", "code": "CxAbC",
    "taken_at": 1787000000, "media_type": 2, "product_type": "clips",
    "caption": {"text": "Ωραίο βίντεο #ελλαδα #test"},
    "like_count": 48210, "comment_count": 1902, "play_count": 1520000,
    "video_duration": 14.2,
    "user": {"pk": "123", "username": "test_gr", "full_name": "Τεστ",
             "follower_count": 5200, "biography": "Αθήνα 🇬🇷"}}}]},
    "next_page_id": "PAGE2"}

HASHTAG_SECTIONS = {"response": {"sections": [{"layout_content": {"medias": [
    {"media": {"pk": "999", "code": "ZZ", "like_count": 12, "comment_count": 3,
               "play_count": 900, "taken_at": 1786000000, "media_type": 2,
               "caption_text": "τεστ #a",
               "user": {"pk": "9", "username": "u9", "follower_count": 100}}}]}}]}}

GRAPHQL = {"data": {"user": {"edge_owner_to_timeline_media": {"edges": [{"node": {
    "id": "777", "shortcode": "SC7", "taken_at_timestamp": 1786500000,
    "edge_media_to_caption": {"edges": [{"node": {"text": "GQL λεζάντα #x"}}]},
    "edge_liked_by": {"count": 500}, "edge_media_to_comment": {"count": 20},
    "video_view_count": 30000, "media_type": 2,
    "owner": {"id": "77", "username": "gqluser", "edge_followed_by": {"count": 3000},
              "biography": "bio"}}}]}}}}


def test_extracts_from_three_different_shapes():
    """Ένας walker καλύπτει v2/sections/GraphQL χωρίς ειδικό κώδικα ανά σχήμα."""
    for payload, expected_user in ((V2_CLIPS, "test_gr"),
                                   (HASHTAG_SECTIONS, "u9"),
                                   (GRAPHQL, "gqluser")):
        posts = N.posts_from(payload)
        assert len(posts) == 1, payload
        assert posts[0].username == expected_user


def test_composite_media_id_is_split():
    """Το «12345_67890» είναι media_id + user_id — κρατάμε μόνο το πρώτο."""
    post = N.posts_from(V2_CLIPS)[0]
    assert post.media_id == "3500000000000000001"


def test_graphql_count_dicts_are_unwrapped():
    post = N.posts_from(GRAPHQL)[0]
    assert post.metrics.likes == 500
    assert post.metrics.comments == 20
    assert post.followers_at_observation == 3000


def test_greek_detection_from_caption_and_profile():
    post = N.posts_from(V2_CLIPS)[0]
    assert post.language == "el" and post.greek_confidence >= 0.9


def test_cursor_extraction_across_naming():
    assert N.next_cursor(V2_CLIPS) == "PAGE2"
    assert N.next_cursor({"response": {"next_max_id": "abc"}}) == "abc"
    assert N.next_cursor({"end_cursor": None}) is None


def test_hashtag_search_extraction():
    payload = {"response": {"results": [{"name": "greekfood", "media_count": 123}]}}
    assert N.hashtags_from(payload) == [("greekfood", 123)]


def test_string_numbers_are_parsed():
    """Με safe_int=true το HikerAPI επιστρέφει τους μεγάλους αριθμούς ως strings."""
    payload = {"pk": "1", "code": "x", "like_count": "1500", "play_count": "99000",
               "comment_count": 5, "taken_at": 1786000000, "media_type": 2}
    post = N.post_from(payload)
    assert post.metrics.likes == 1500 and post.metrics.views == 99000
