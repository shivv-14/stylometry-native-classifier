"""Function-word frequencies per 100 words."""
from __future__ import annotations

from collections import Counter

from .text_utils import word_tokens

DEFAULT_FUNCTION_WORDS = ["the", "a", "an", "of", "to", "in", "and", "for", "with", "but", "on", "at",
                          "from", "as", "by", "that", "which", "this", "it", "is", "be", "not", "or",
                          "so", "if", "there"]


def names(function_words=DEFAULT_FUNCTION_WORDS) -> list[str]:
    return [f"fw_{w}" for w in function_words] + ["fw_total"]


def function_word_features(doc, function_words=DEFAULT_FUNCTION_WORDS) -> dict:
    tokens = [t.text.lower() for t in word_tokens(doc)]
    n = max(len(tokens), 1)
    counts = Counter(tokens)
    out = {f"fw_{w}": 100.0 * counts[w] / n for w in function_words}
    out["fw_total"] = sum(out.values())
    return out
