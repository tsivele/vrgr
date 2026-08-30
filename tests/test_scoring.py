"""Σκοράρισμα: πολλαπλασιαστής τεκμηρίων και κοινή βελτιστοποίηση."""
from vrgr.schemas import (CaptionCandidate, HashtagCandidate, HashtagSet,
                          MinedPatterns, ObservedPost, ResearchBundle,
                          VideoAnalysis, VideoContent, VideoTechnical,
                          ViralAngle, ViralSignals)
from vrgr.scoring.ranking import rank, synergy
from vrgr.scoring.viral_score import ViralScorer


def _fixtures():
    analysis = VideoAnalysis(
        technical=VideoTechnical(duration_s=18),
        content=VideoContent(summary="ζευγάρι μαλώνει αστεία", niche="χιούμορ"),
        signals=ViralSignals(curiosity_gap=72, emotional_trigger=60, relatability=88,
                             shareability=80, commentability=75, save_potential=30,
                             rewatch_potential=55, attention_hold=78,
                             greek_cultural_fit=85))
    angle = ViralAngle(name="Η διαπραγμάτευση", strategy="ταύτιση", strength=82,
                       caption_should_add="το ανείπωτο σκορ της σχέσης")
    caption = CaptionCandidate(
        text="Δεν είναι ότι δεν ξέρω τι θέλω. Θέλω να το προτείνεις εσύ και μετά "
             "να διαφωνήσω.", strategy="ταύτιση").fill()
    hset = HashtagSet(
        tags=[f"tag{i}" for i in range(12)], strategy="ελληνική_στόχευση",
        greek_share=1.0, evidence_share=0.5, score=62,
        candidates=[HashtagCandidate(tag=f"tag{i}", media_count=50_000,
                                     evidence_count=2, is_greek=True)
                    for i in range(12)]).fill()
    return analysis, angle, caption, hset


def test_evidence_multiplier_caps_unsupported_scores(config_dir):
    """
    Ο μηχανισμός ειλικρίνειας: ίδια ακριβώς λεζάντα, διαφορετική τεκμηρίωση.
    Χωρίς δεδομένα το σύστημα ΔΕΝ δικαιούται υψηλό σκορ.
    """
    scorer = ViralScorer(config_dir)
    analysis, angle, caption, hset = _fixtures()
    research = ResearchBundle(
        posts=[ObservedPost(media_id=str(i)) for i in range(120)],
        greek_posts=52,
        outliers=[ObservedPost(media_id=f"o{i}", hashtags=["tag1", "tag2"])
                  for i in range(14)])
    mined = MinedPatterns(sample_size=80, outlier_sample_size=14,
                          caption_len_p25=60, caption_len_median=95,
                          caption_len_p75=150, question_share=0.5, cta_share=0.4)

    with_evidence = scorer.score(analysis, angle, caption, hset, research, mined,
                                 {"score": 0.68, "coverage": 0.8,
                                  "patterns": [{"key": "k", "n": 40}]}, 18)
    without = scorer.score(analysis, angle, caption, hset, None, None, None, 0)

    assert with_evidence.evidence_multiplier == 1.0
    assert without.evidence_multiplier <= 0.75
    assert with_evidence.total - without.total > 15
    assert with_evidence.confidence == "υψηλή"
    assert without.confidence == "χαμηλή"
    assert without.notes            # πρέπει να ΕΞΗΓΕΙ τον περιορισμό


def test_confidence_interval_widens_without_data(config_dir):
    scorer = ViralScorer(config_dir)
    analysis, angle, caption, hset = _fixtures()
    poor = scorer.score(analysis, angle, caption, hset, None, None, None, 0)
    assert poor.interval[1] - poor.interval[0] >= 20


def test_weights_are_normalised(config_dir):
    scorer = ViralScorer(config_dir)
    assert abs(sum(scorer.weights.values()) - 1.0) < 1e-9
    assert len(scorer.weights) == 7


def test_synergy_is_strategy_dependent():
    """Λεζάντα περιέργειας θέλει στενά hashtags· ταύτισης αντέχει πλατιά."""
    curiosity = CaptionCandidate(text="x", strategy="περιέργεια").fill()
    relatable = CaptionCandidate(text="x", strategy="ταύτιση").fill()
    broad = HashtagSet(strategy="εμβέλεια").fill()
    narrow = HashtagSet(strategy="ανακάλυψη").fill()
    assert synergy(curiosity, narrow) > synergy(curiosity, broad)
    assert synergy(relatable, broad) > synergy(curiosity, broad)


def test_ranking_returns_distinct_captions(config_dir):
    """Οι «εναλλακτικές» πρέπει να είναι όντως διαφορετικές λεζάντες."""
    scorer = ViralScorer(config_dir)
    analysis, angle, _, hset = _fixtures()
    captions = [CaptionCandidate(text=f"Λεζάντα υποψήφια αριθμός {i} για δοκιμή",
                                 strategy="ταύτιση").fill() for i in range(4)]
    sets = [HashtagSet(tags=[f"t{j}" for j in range(10)], strategy=s,
                       greek_share=1.0, score=55 + j,
                       candidates=[HashtagCandidate(tag=f"t{j}", media_count=1000)
                                   for j in range(10)]).fill()
            for j, s in enumerate(["ισορροπημένο", "ανακάλυψη", "εμβέλεια"])]
    ranked = rank(analysis, angle, captions, sets, scorer, top_n=4)
    ids = [c.caption.id for c in ranked]
    assert len(ids) == len(set(ids))
    assert ranked[0].score.total >= ranked[-1].score.total
