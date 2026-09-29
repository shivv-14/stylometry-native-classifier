"""Punctuation marks per 100 words, counted on the raw (cleaned) text."""
from __future__ import annotations

PUNCTUATION = {
    "punct_comma": [","],
    "punct_period": ["."],
    "punct_semicolon": [";"],
    "punct_colon": [":"],
    "punct_question": ["?"],
    "punct_exclamation": ["!"],
    "punct_parentheses": ["(", ")"],
    "punct_quotes": ['"'],
    "punct_dash": ["-", "–", "—"],
    "punct_apostrophe": ["'"],
}
NAMES = list(PUNCTUATION)


def punctuation_features(text: str, n_words: int) -> dict:
    n = max(n_words, 1)
    return {name: 100.0 * sum(text.count(ch) for ch in chars) / n for name, chars in PUNCTUATION.items()}
