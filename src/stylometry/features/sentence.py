"""Sentence-length features."""
from __future__ import annotations

import numpy as np

from .text_utils import word_tokens

NAMES = ["sent_mean_len", "sent_median_len", "sent_std_len", "sent_min_len", "sent_max_len", "sent_per_100w"]


def sentence_features(doc) -> dict:
    lengths = [len(word_tokens(s)) for s in doc.sents]
    lengths = np.array([n for n in lengths if n > 0], dtype=float)
    n_words = len(word_tokens(doc))
    if len(lengths) == 0:
        return dict.fromkeys(NAMES, 0.0)
    return {
        "sent_mean_len": float(lengths.mean()),
        "sent_median_len": float(np.median(lengths)),
        "sent_std_len": float(lengths.std()),
        "sent_min_len": float(lengths.min()),
        "sent_max_len": float(lengths.max()),
        "sent_per_100w": 100.0 * len(lengths) / max(n_words, 1),
    }
