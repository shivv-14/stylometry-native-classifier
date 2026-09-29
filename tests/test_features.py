import math

import numpy as np
import pytest

from stylometry.data.clean import clean_text, truncate_words, word_count
from stylometry.features.handcrafted import HandcraftedFeatures, feature_group
from stylometry.features.punctuation import punctuation_features
from stylometry.features.text_utils import get_nlp
from stylometry.features.vocabulary import mattr, ttr, ttr_fixed_window


@pytest.fixture(scope="module")
def hf():
    return HandcraftedFeatures().fit([])


def feats(hf, text):
    return hf.transform([text]).iloc[0]


def test_sentence_counts(hf):
    f = feats(hf, "The cat sat on the mat. The dog ran away quickly. Birds sing.")
    # 3 sentences over 13 words (6 + 5 + 2)
    assert f["sent_per_100w"] == pytest.approx(100 * 3 / 13)
    assert f["sent_max_len"] == 6
    assert f["sent_min_len"] == 2
    assert f["sent_mean_len"] == pytest.approx(13 / 3)


def test_punctuation_rates():
    text = "Well, yes; no: maybe? Sure! (fine) \"ok\" it's well-known."
    f = punctuation_features(text, n_words=10)
    assert f["punct_comma"] == pytest.approx(10.0)
    assert f["punct_semicolon"] == pytest.approx(10.0)
    assert f["punct_question"] == pytest.approx(10.0)
    assert f["punct_parentheses"] == pytest.approx(20.0)
    assert f["punct_quotes"] == pytest.approx(20.0)
    assert f["punct_apostrophe"] == pytest.approx(10.0)
    assert f["punct_dash"] == pytest.approx(10.0)


def test_ttr_in_range(hf):
    f = feats(hf, "one two three one two three four five six seven " * 5)
    for c in ("vocab_ttr100", "vocab_mattr", "vocab_hapax_ratio"):
        assert 0.0 <= f[c] <= 1.0
    assert ttr(["a", "b", "a"]) == pytest.approx(2 / 3)
    assert ttr_fixed_window(["a"] * 150 + ["b"], 100) == pytest.approx(0.01)


def test_mattr_matches_bruteforce():
    rng = np.random.default_rng(0)
    words = list(rng.choice(list("abcdefghij"), size=120))
    w = 50
    brute = np.mean([len(set(words[i:i + w])) / w for i in range(len(words) - w + 1)])
    assert mattr(words, w) == pytest.approx(brute)


@pytest.mark.parametrize("text", [
    "Just one sentence without any punctuation at all",
    "ALL CAPS TEXT IS HERE AND IT KEEPS SHOUTING LOUDLY",
    "Hi.",
    "...!!!",
    "",
])
def test_no_nan_on_edge_cases(hf, text):
    f = hf.transform([text])
    assert not f.isna().any().any()
    assert np.isfinite(f.to_numpy()).all()


def test_function_words_per_100(hf):
    f = feats(hf, "The the the cat and a dog")
    assert f["fw_the"] == pytest.approx(100 * 3 / 7)
    assert f["fw_and"] == pytest.approx(100 / 7)


def test_pos_percentages_sum_below_100(hf):
    f = feats(hf, "The quick brown fox jumps over the lazy dog.")
    pos_cols = [c for c in f.index if c.startswith("pos_")]
    assert 0 < f[pos_cols].sum() <= 100 + 1e-9


def test_feature_names_and_groups(hf):
    names = hf.get_feature_names_out()
    groups = {feature_group(n) for n in names}
    assert groups == {"sentence", "vocabulary", "function_words", "pos", "punctuation"}
    assert len(names) == len(set(names))


def test_group_selection():
    only = HandcraftedFeatures(groups=("punctuation",)).fit([])
    assert all(n.startswith("punct_") for n in only.get_feature_names_out())
    length = HandcraftedFeatures(groups=("length",)).fit([])
    assert list(length.get_feature_names_out()) == ["length_word_count", "sent_mean_len"]


def test_clean_keeps_punctuation_and_case():
    raw = "Topic: smoking\n\n  “Hello,”   she   said’s.  \n\nNew   para."
    out = clean_text(raw, r"^\s*(Topic:.*)\s*$")
    assert out == "\"Hello,\" she said's.\nNew para."


def test_truncate_words():
    t = "One two three. Four five six."
    assert truncate_words(t, 3) == "One two three."
    assert word_count(truncate_words(t, 4)) == 4
    assert truncate_words(t, None) == t


def test_spacy_ner_disabled():
    assert "ner" not in get_nlp().pipe_names
    assert not math.isnan(len(get_nlp().pipe_names))
