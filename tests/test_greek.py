"""Ελληνική γλώσσα — τα θεμέλια κάθε ταιριάσματος κειμένου."""
import pytest

from vrgr import greek as G


def test_normalize_strips_accents_and_final_sigma():
    assert G.normalize("Καφές") == "καφεσ"
    assert G.normalize("ΤΩΡΑ") == "τωρα"
    # Ίδια λέξη με/χωρίς τόνο πρέπει να ταυτίζεται
    assert G.normalize("ελληνικά") == G.normalize("ΕΛΛΗΝΙΚΑ")


def test_normalize_tag_keeps_final_sigma():
    """Κρίσιμο: το «#σχεσεισ» δεν υπάρχει στο Instagram, το «#σχεσεις» ναι."""
    assert G.normalize_tag("Σχέσεις") == "σχεσεις"
    assert G.normalize_tag("#ΕΛΛΑΔΑ") == "ελλαδα"
    assert G.normalize_tag("greek memes") == "greekmemes"
    assert G.normalize("σχέσεις") != G.normalize_tag("σχέσεις")


def test_word_tokens_preserve_valid_hashtags():
    tokens = G.word_tokens("Οι σχέσεις και ο έρωτας στην Ελλάδα")
    assert "σχεσεις" in tokens and "σχεσεισ" not in tokens
    assert "και" not in tokens          # stopword


def test_greek_ratio():
    assert G.greek_ratio("Καλημέρα") == 1.0
    assert G.greek_ratio("Hello") == 0.0
    # 8 ελληνικοί χαρακτήρες / 13 αλφαβητικοί συνολικά
    assert G.greek_ratio("Καλημέρα hello") == pytest.approx(8 / 13)


def test_greeklish_detected_as_greek():
    """Το μισό ελληνικό κοινό γράφει greeklish — δεν είναι «αγγλικά»."""
    assert G.greek_confidence("einai poly kalo re file") > 0.4
    assert G.greek_confidence("the quick brown fox jumps") < 0.2


def test_extract_hashtags():
    tags = G.extract_hashtags("τεστ #καφές #Athens #foodporn #καφές")
    assert tags == ["καφές", "athens", "foodporn"]      # χωρίς διπλά, πεζά


def test_caption_body_removes_hashtags():
    assert G.caption_body("Κάτι ωραίο #a #b") == "Κάτι ωραίο"


def test_char_ngrams_absorb_inflection():
    """Η ελληνική κλίση σπάει το word matching· τα n-grams την απορροφούν."""
    a, b = set(G.char_ngrams("καφέδες")), set(G.char_ngrams("καφέ"))
    assert len(a & b) >= 2


def test_is_greek_hashtag():
    assert G.is_greek_hashtag("ελληνικοχιουμορ")
    assert G.is_greek_hashtag("greekgirl")
    assert not G.is_greek_hashtag("foodporn")
