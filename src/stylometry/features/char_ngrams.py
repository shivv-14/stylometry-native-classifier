"""Character n-gram TF-IDF."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def make_char_vectorizer(params: dict | None = None) -> TfidfVectorizer:
    p = dict(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=5000, sublinear_tf=True)
    p.update(params or {})
    p["ngram_range"] = tuple(p["ngram_range"])
    # lowercase=False: capitalisation habits are part of the signal
    return TfidfVectorizer(lowercase=False, **p)
