"""Part-of-speech distribution (spaCy universal POS tags)."""
from __future__ import annotations

from collections import Counter

from .text_utils import word_tokens

DEFAULT_POS_TAGS = ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
                    "CCONJ", "SCONJ", "AUX", "PART", "NUM"]


def names(pos_tags=DEFAULT_POS_TAGS) -> list[str]:
    return [f"pos_{t}" for t in pos_tags]


def pos_features(doc, pos_tags=DEFAULT_POS_TAGS) -> dict:
    """Percentage of word tokens with each tag."""
    tokens = word_tokens(doc)
    n = max(len(tokens), 1)
    counts = Counter(t.pos_ for t in tokens)
    return {f"pos_{tag}": 100.0 * counts[tag] / n for tag in pos_tags}
