"""Χαρτοφυλάκιο hashtags και παράγωγη δυσκολία."""
from vrgr.generation.hashtags import (build_candidates, build_sets,
                                      score_candidate, PORTFOLIOS)
from vrgr.research.collector import classify_tier, _difficulty
from vrgr.schemas import HashtagCandidate, HashtagStat, MinedPatterns, VideoContent


def _stats():
    spec = [("ελληνικοχιουμορ", 85_000, "niche", 44), ("greekmemes", 310_000, "mid", 61),
            ("σχεσεις", 48_000, "niche", 38), ("ζευγαρι", 7_200, "micro", 25),
            ("ελλαδα", 42_000_000, "broad", 97), ("greece", 95_000_000, "broad", 99),
            ("ζευγαρακια", 3_100, "micro", 22), ("αγαπη", 890_000, "mid", 74),
            ("greekcouple", 26_000, "niche", 41), ("καθημερινοτητα", 140_000, "mid", 55)]
    return {t: HashtagStat(tag=t, media_count=m, tier=tier, difficulty=d, is_greek=True)
            for t, m, tier, d in spec}


def test_tier_classification():
    assert classify_tier(3_000) == "micro"
    assert classify_tier(45_000) == "niche"
    assert classify_tier(400_000) == "mid"
    assert classify_tier(12_000_000) == "broad"
    assert classify_tier(None) == "unknown"


def test_derived_difficulty_rewards_open_hashtags():
    """Αν στα κορυφαία posts χωράνε μικροί λογαριασμοί, η πόρτα είναι ανοιχτή."""
    open_tag = HashtagStat(tag="a", media_count=85_000,
                           median_followers_top=5_500, small_account_share=1.0)
    closed_tag = HashtagStat(tag="b", media_count=95_000_000,
                             median_followers_top=1_800_000, small_account_share=0.0)
    assert _difficulty(open_tag) < 60
    assert _difficulty(closed_tag) > 90


def test_evidence_beats_raw_popularity():
    """Τεκμηριωμένο στενό tag > τεράστιο tag χωρίς απόδειξη."""
    proven = HashtagCandidate(tag="a", relevance=0.8, evidence_count=8,
                              is_greek=True, tier="niche", difficulty=42)
    popular = HashtagCandidate(tag="b", relevance=0.8, evidence_count=0,
                               is_greek=True, tier="broad", difficulty=98)
    assert score_candidate(proven) > score_candidate(popular)


def test_portfolio_spans_multiple_tiers():
    content = VideoContent(niche="χιούμορ", sub_niche="σχέσεις", mood="παιχνιδιάρικη")
    mined = MinedPatterns(outlier_sample_size=12, sample_size=60,
                          top_hashtags=[("ελληνικοχιουμορ", 3.4, 9),
                                        ("greekmemes", 2.1, 7), ("σχεσεις", 1.8, 5)])
    terms = {"hashtags_niche": ["ζευγαρακια", "greekcouple"],
             "hashtags_mid": ["σχεσεις", "ζευγαρι", "αγαπη", "καθημερινοτητα"],
             "hashtags_broad": ["ελλαδα", "greece"]}
    cands = build_candidates(content, None, mined, _stats(), planner_terms=terms)
    sets = build_sets(cands, mined=mined, target_size=13)
    assert sets
    for hset in sets:
        measured = {t: n for t, n in hset.tier_distribution.items() if t != "unknown"}
        assert len(measured) >= 3, f"{hset.strategy}: μονοδιάστατο σετ {measured}"
        # Τα άγνωστου μεγέθους δεν επιτρέπεται να κατακλύζουν το σετ
        unknown = hset.tier_distribution.get("unknown", 0)
        assert unknown <= max(2, int(len(hset.tags) * 0.45))


def test_generated_tags_are_valid_greek():
    """Κανένα «#σχεσεισ» — άκυρο hashtag που δεν υπάρχει στο Instagram."""
    content = VideoContent(niche="σχέσεις", sub_niche="έρωτας",
                           main_subject="ζευγάρι", mood="τρυφερή")
    cands = build_candidates(content, None, None, {})
    for c in cands:
        assert not c.tag.endswith("σ") or c.tag in ("greekmemes",), c.tag
        assert " " not in c.tag and "#" not in c.tag


def test_discovery_portfolio_favours_narrow_tags():
    spec = PORTFOLIOS["ανακάλυψη"]
    assert spec["niche"][1] > spec["broad"][1]
    assert spec["micro"][1] > spec["broad"][1]
