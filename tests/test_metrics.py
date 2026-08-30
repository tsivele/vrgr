"""Κανονικοποίηση ως προς μέγεθος λογαριασμού — ο πυρήνας του συστήματος."""
from conftest import make_post
from vrgr.analysis import metrics as M


def test_small_account_outranks_mega_account():
    """
    Η θεμελιώδης απαίτηση #14.

    5K followers → 1.5M views (V/F 300) πρέπει να κατατάσσεται ΠΑΝΩ από
    10M followers → 2M views (V/F 0.2), παρότι το δεύτερο έχει περισσότερες
    απόλυτες προβολές.
    """
    micro = make_post("1", "micro", followers=5_000, views=1_500_000,
                      likes=62_000, comments=3_100)
    mega = make_post("2", "mega", followers=10_000_000, views=2_000_000,
                     likes=40_000, comments=900)
    M.enrich([micro, mega])
    assert micro.normalized.outlier_score > mega.normalized.outlier_score
    assert micro.normalized.outlier_score > 60
    assert mega.normalized.outlier_score < 20


def test_viral_multiplier_uses_creator_median():
    """Breakout σε σχέση με τον ΙΔΙΟ creator, όχι απόλυτα νούμερα."""
    posts = [make_post(str(i), "same", followers=45_000, views=v)
             for i, v in enumerate([20_000, 25_000, 30_000, 380_000])]
    M.enrich(posts)
    breakout = posts[-1]
    assert breakout.normalized.viral_multiplier > 10
    assert breakout.normalized.outlier_score > 55


def test_median_not_mean_for_baseline():
    """Ένα viral post δεν επιτρέπεται να «κρύψει» τα επόμενα breakouts."""
    views = [10_000, 12_000, 11_000, 5_000_000]
    posts = [make_post(str(i), "c", views=v) for i, v in enumerate(views)]
    base = M.creator_baseline(posts)
    assert base < 100_000            # διάμεσο, όχι μέσος όρος (~1.25M)


def test_missing_data_stays_none():
    """`None` σημαίνει «δεν το δίνει το API» — ποτέ σιωπηλό μηδέν."""
    p = make_post("1", followers=None, views=None, likes=None, comments=None)
    p.followers_at_observation = None
    p.metrics.views = None
    nm = M.compute(p)
    assert nm.vf_ratio is None and nm.outlier_score is None


def test_rank_outliers_filters_before_sorting():
    """Αν φιλτράραμε μετά την ταξινόμηση, θα χάναμε τα μικρά breakouts."""
    posts = [
        make_post("1", "gr_micro", followers=4_000, views=800_000),
        make_post("2", "xx", followers=4_000, views=900_000, greek=0.1),
        make_post("3", "gr_flat", followers=100_000, views=50_000),
    ]
    M.enrich(posts)
    picked = M.rank_outliers(posts, min_score=45.0, greek_only=True)
    names = [p.username for p in picked]
    assert "gr_micro" in names
    assert "xx" not in names          # μη ελληνικό, όσο viral κι αν είναι
    assert "gr_flat" not in names     # ελληνικό αλλά χωρίς breakout
