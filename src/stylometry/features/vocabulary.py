"""Vocabulary richness features."""
from __future__ import annotations

from collections import Counter

from .text_utils import alpha_words

NAMES = ["vocab_ttr100", "vocab_mattr", "vocab_avg_word_len", "vocab_long_word_prop",
         "vocab_hapax_ratio", "vocab_unique_per_100w"]


def ttr(words: list[str]) -> float:
    return len(set(words)) / len(words) if words else 0.0


def ttr_fixed_window(words: list[str], window: int = 100) -> float:
    """TTR of the first `window` words (all words if the text is shorter)."""
    return ttr(words[:window])


def mattr(words: list[str], window: int = 50) -> float:
    """Moving-average TTR (Covington & McFall 2010); plain TTR if shorter than the window."""
    n = len(words)
    if n == 0:
        return 0.0
    if n <= window:
        return ttr(words)
    counts = Counter(words[:window])
    total = len(counts)
    for i in range(window, n):
        out, new = words[i - window], words[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[new] += 1
        total += len(counts)
    return total / ((n - window + 1) * window)


def vocabulary_features(doc, ttr_window: int = 100, mattr_window: int = 50, long_word_len: int = 7) -> dict:
    words = alpha_words(doc)
    n = len(words)
    if n == 0:
        return dict.fromkeys(NAMES, 0.0)
    freq = Counter(words)
    return {
        "vocab_ttr100": ttr_fixed_window(words, ttr_window),
        "vocab_mattr": mattr(words, mattr_window),
        "vocab_avg_word_len": sum(map(len, words)) / n,
        "vocab_long_word_prop": sum(len(w) >= long_word_len for w in words) / n,
        "vocab_hapax_ratio": sum(1 for c in freq.values() if c == 1) / n,
        "vocab_unique_per_100w": 100.0 * len(freq) / n,
    }
